# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from github.com/openai/CLIP
from collections import OrderedDict

import numpy as np
import timm
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

import losses
import utils

import vit


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout_prob=0.0, drop_path_prob=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)
        self.drop_path = DropPath(drop_prob=drop_path_prob)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inp):
        x, mode = inp
        if mode == 'local':
            self.dropout(x)
            x = x + self.drop_path(self.attention(self.ln_1(x)))
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return (x, mode)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, dropout_prob=0.0, drop_path_prob=0.0):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(
                width,
                heads,
                attn_mask,
                dropout_prob,
                drop_path_prob
            ) for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, mode='global'):
        return self.resblocks((x, mode))[0]


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        vision_width: int,
        vision_model: nn.Module,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int = 12,
        detach_proj: bool = False,
        no_share_token=False,
        clip_proj_type='linear',
        clip_hidden_dim=4096,
        global_text_mask_prob=1.0,
        local_text_mask_prob=0.5,
        text_dropout_prob=0.0,
        text_drop_path_prob=0.0,
        **kwargs,
    ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width
        self.transformer_width = transformer_width
        self.embed_dim = embed_dim
        self.detach_proj = detach_proj
        self.clip_proj_type = clip_proj_type

        self.visual = vision_model
        self.no_share_token = no_share_token

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            dropout_prob=text_dropout_prob,
            drop_path_prob=text_drop_path_prob,
        )

        self.vocab_size = vocab_size
        self.local_text_mask_prob = local_text_mask_prob
        self.global_text_mask_prob = global_text_mask_prob
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        if clip_proj_type == 'mlp':
            self.image_projector = self._build_mlp(
                in_dim=self.vision_width,
                mlp_dim=clip_hidden_dim,
                out_dim=embed_dim
            )
            self.text_projector = self._build_mlp(
                in_dim=self.transformer_width,
                mlp_dim=clip_hidden_dim,
                out_dim=embed_dim
            )
        else:
            self.image_projector = nn.Linear(self.vision_width, embed_dim, bias=False)
            self.text_projector = nn.Linear(self.transformer_width, embed_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def _build_mlp(self, in_dim, mlp_dim, out_dim, num_layers=3):
        mlp = [
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", utils.infer_batchnorm_class()(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True))
        ]
        i = 1
        for i in range(2, num_layers):
            mlp.extend([
                (f"layer{i}", nn.Linear(mlp_dim, mlp_dim)),
                (f"bn{i}", utils.infer_batchnorm_class()(mlp_dim)),
                (f"relu{i}", nn.ReLU(inplace=True))
            ])
        mlp.append((f"layer{i+1}", nn.Linear(mlp_dim, out_dim)))
        return nn.Sequential(OrderedDict(mlp))

    @torch.no_grad()
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.clip_proj_type == 'linear':
            nn.init.normal_(self.image_projector.weight, std=self.vision_width ** -0.5)
            nn.init.normal_(self.text_projector.weight, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image):
        feats = self.visual(image)
        z = self.image_projector(feats)
        return {'feats_image': feats, 'z_image': z}

    def encode_text(self, text, mode='global', forward_proj=True):
        range_index = torch.arange(text.size(0))
        eot_index = text.argmax(dim=-1)
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, mode=mode)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        feats = x[range_index, eot_index]

        out = {'feats_text': feats}

        if forward_proj:
            out['z_text'] = self.text_projector(feats.detach() if self.detach_proj else feats)

        return out

    def forward(self, image, text):
        out_image = self.encode_image(image)
        out_text = self.encode_text(text)
        return {**out_image, **out_text, 'logit_scale': self.logit_scale.exp()}

    @torch.no_grad()
    def predict_zeroshot(self, image_feats, text_feats):
        z_image = image_feats['z_image']
        z_text = text_feats['z_text']
    
        z_image = z_image / z_image.norm(dim=-1, keepdim=True)
        z_text = z_text / z_text.norm(dim=-1, keepdim=True)
        similarity = z_image @ z_text.t()
        return {'z_sim': similarity}

    def encode_image_val(self, image):
        out = self.encode_image(image)
        return out

    def encode_text_val(self, text):
        out = self.encode_text(text)
        return out

class CL2L(CLIP):
    def __init__(self, separate_proj=False, cl2l_txt_proj_type='mlp', cl2l_img_proj_type='mlp', **kwargs):
        super().__init__(separate_proj=False, **kwargs)
        self.separate_proj = separate_proj
        if separate_proj:
            self.l2l_logit_scale = nn.Parameter(
                torch.ones([]) * np.log(1 / 0.1))
            if cl2l_img_proj_type == 'mlp':
                self.l2l_image_projector = self._build_mlp(
                    in_dim=self.vision_width,
                    mlp_dim=4096,
                    out_dim=self.embed_dim
                )
            else:
                self.l2l_image_projector = nn.Linear(self.vision_width, self.embed_dim, bias=False)
            if cl2l_txt_proj_type == 'mlp':
                self.l2l_text_projector = self._build_mlp(
                    in_dim=self.transformer_width,
                    mlp_dim=4096,
                    out_dim=self.embed_dim
                )
            else:
                self.l2l_text_projector = nn.Linear(self.transformer_width, self.embed_dim, bias=False)

        else:
            self.l2l_image_projector = self.image_projector
            self.l2l_text_projector = self.text_projector

    def encode_image_val(self, image):
        out = self.encode_image(image)
        out['h_image'] = self.l2l_image_projector(out['feats_image'])
        return out

    def encode_text_val(self, text):
        out = super().encode_text(text)
        out['h_text'] = self.l2l_text_projector(out['feats_text'])
        return out

    def forward(self, image_global, text, *image_local):
        text_global, *text_local = text.unbind(1)
        out = super().forward(image_global, text_global)

        # forward backbone
        out['feats_image_local'] = [self.visual(l) for l in image_local]
        out['feats_text_local'] = [
            self.encode_text(t, mode='local', forward_proj=False)['feats_text']
            for t in text_local
        ]

        # forward projector
        out['h_image_local'] = [self.l2l_image_projector(l) for l in out['feats_image_local']]
        out['h_text_local'] = [self.l2l_text_projector(l) for l in out['feats_text_local']]

        # fix names
        out['z_image_global'] = out.pop('z_image')
        out['z_text_global'] = out.pop('z_text')
        out['h_logit_scale'] = self.l2l_logit_scale.exp() if self.separate_proj else out['logit_scale']

        return out

    @torch.no_grad()
    def predict_zeroshot(self, image_feats, text_feats):
        outs = super().predict_zeroshot(image_feats, text_feats)

        z_image = image_feats['h_image']
        z_text = text_feats['h_text']

        z_image = z_image / z_image.norm(dim=-1, keepdim=True)
        z_text = z_text / z_text.norm(dim=-1, keepdim=True)
        similarity = z_image @ z_text.t()
        return {**outs, 'h_sim': similarity}


class BARLIP(CL2L):
    def __init__(self, barlip_proj_dim, barlip_hidden_dim, **kwargs):
        super().__init__(**kwargs)

        self.barlip_image_projector_global = nn.Sequential(
            nn.Linear(kwargs['vision_width'], barlip_hidden_dim),
            utils.infer_batchnorm_class()(barlip_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(barlip_hidden_dim, barlip_hidden_dim),
            utils.infer_batchnorm_class()(barlip_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(barlip_hidden_dim, barlip_proj_dim),
            utils.infer_batchnorm_class()(barlip_proj_dim)
        )
        self.barlip_text_projector_global = nn.Sequential(
            nn.Linear(kwargs['transformer_width'], barlip_hidden_dim),
            utils.infer_batchnorm_class()(barlip_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(barlip_hidden_dim, barlip_hidden_dim),
            utils.infer_batchnorm_class()(barlip_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(barlip_hidden_dim, barlip_proj_dim),
            utils.infer_batchnorm_class()(barlip_proj_dim)
        )

        if 'separate_proj_child' in kwargs and kwargs['separate_proj_child']:
            self.barlip_image_projector_local = nn.Sequential(
                nn.Linear(kwargs['vision_width'], barlip_hidden_dim),
                utils.infer_batchnorm_class()(barlip_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(barlip_hidden_dim, barlip_hidden_dim),
                utils.infer_batchnorm_class()(barlip_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(barlip_hidden_dim, barlip_proj_dim),
                utils.infer_batchnorm_class()(barlip_proj_dim)
            )
            self.barlip_text_projector_local = nn.Sequential(
                nn.Linear(kwargs['transformer_width'], barlip_hidden_dim),
                utils.infer_batchnorm_class()(barlip_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(barlip_hidden_dim, barlip_hidden_dim),
                utils.infer_batchnorm_class()(barlip_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(barlip_hidden_dim, barlip_proj_dim),
                utils.infer_batchnorm_class()(barlip_proj_dim)
            )
        else:
            self.barlip_image_projector_local = self.barlip_image_projector_global
            self.barlip_text_projector_local = self.barlip_text_projector_global

    def forward(self, image, text, *image_local):
        out = super().forward(image, text, *image_local)

        out['v_image'] = self.barlip_image_projector_global(out['feats_image'])
        out['v_text'] = self.barlip_text_projector_global(out['feats_text'])

        out['v_image_local'] = [self.barlip_image_projector_local(l) for l in out['feats_image_local']]
        out['v_text_local'] = [self.barlip_text_projector_local(l) for l in out['feats_text_local']]

        return out


class SIAMLIP(CL2L):
    def __init__(self, siamlip_proj_dim, siamlip_hidden_dim, siamlip_no_last_bn, **kwargs):
        super().__init__(**kwargs)

        self.siamlip_image_projector_global = nn.Sequential(
            nn.Linear(kwargs['vision_width'], siamlip_hidden_dim, bias=False),
            utils.infer_batchnorm_class()(siamlip_hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Linear(siamlip_hidden_dim, siamlip_hidden_dim, bias=False),
            # utils.infer_batchnorm_class()(siamlip_hidden_dim),
            # nn.ReLU(inplace=True),
            nn.Linear(siamlip_hidden_dim, siamlip_proj_dim, bias=False),
        )

        self.siamlip_text_projector_global = nn.Sequential(
            nn.Linear(kwargs['transformer_width'], siamlip_hidden_dim, bias=False),
            utils.infer_batchnorm_class()(siamlip_hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Linear(siamlip_hidden_dim, siamlip_hidden_dim, bias=False),
            # utils.infer_batchnorm_class()(siamlip_hidden_dim),
            # nn.ReLU(inplace=True),
            nn.Linear(siamlip_hidden_dim, siamlip_proj_dim, bias=False),
        )

        if 'separate_proj_child' in kwargs and kwargs['separate_proj_child']:
            self.siamlip_image_projector_local = nn.Sequential(
                nn.Linear(kwargs['vision_width'], siamlip_hidden_dim, bias=False),
                utils.infer_batchnorm_class()(siamlip_hidden_dim),
                nn.ReLU(inplace=True),
                # nn.Linear(siamlip_hidden_dim, siamlip_hidden_dim, bias=False),
                # utils.infer_batchnorm_class()(siamlip_hidden_dim),
                # nn.ReLU(inplace=True),
                nn.Linear(siamlip_hidden_dim, siamlip_proj_dim, bias=False),
            )

            self.siamlip_text_projector_local = nn.Sequential(
                nn.Linear(kwargs['transformer_width'], siamlip_hidden_dim, bias=False),
                utils.infer_batchnorm_class()(siamlip_hidden_dim),
                nn.ReLU(inplace=True),
                # nn.Linear(siamlip_hidden_dim, siamlip_hidden_dim, bias=False),
                # utils.infer_batchnorm_class()(siamlip_hidden_dim),
                # nn.ReLU(inplace=True),
                nn.Linear(siamlip_hidden_dim, siamlip_proj_dim, bias=False),
            )
        else:
            self.siamlip_image_projector_local = self.siamlip_image_projector_global
            self.siamlip_text_projector_local = self.siamlip_text_projector_global


        if not siamlip_no_last_bn:
            self.siamlip_image_projector_global = nn.Sequential(
                self.siamlip_image_projector_global,
                utils.infer_batchnorm_class()(siamlip_proj_dim, affine=False)
            )
            self.siamlip_text_projector_global = nn.Sequential(
                self.siamlip_text_projector_global,
                utils.infer_batchnorm_class()(siamlip_proj_dim, affine=False)
            )
            self.siamlip_image_projector_local = nn.Sequential(
                self.siamlip_image_projector_local,
                utils.infer_batchnorm_class()(siamlip_proj_dim, affine=False)
            )
            self.siamlip_text_projector_local = nn.Sequential(
                self.siamlip_text_projector_local,
                utils.infer_batchnorm_class()(siamlip_proj_dim, affine=False)
            )

        # predictors
        self.image_text_predictor_global = nn.Sequential(
            nn.Linear(siamlip_proj_dim, siamlip_hidden_dim, bias=False),
            utils.infer_batchnorm_class()(siamlip_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(siamlip_hidden_dim, siamlip_proj_dim),
        )

        self.text_image_predictor_global = nn.Sequential(
            nn.Linear(siamlip_proj_dim, siamlip_hidden_dim, bias=False),
            utils.infer_batchnorm_class()(siamlip_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(siamlip_hidden_dim, siamlip_proj_dim),
        )

        if 'separate_proj_child' in kwargs and kwargs['separate_proj_child']:
            self.image_text_predictor_local = nn.Sequential(
                nn.Linear(siamlip_proj_dim, siamlip_hidden_dim, bias=False),
                utils.infer_batchnorm_class()(siamlip_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(siamlip_hidden_dim, siamlip_proj_dim),
            )

            self.text_image_predictor_local = nn.Sequential(
                nn.Linear(siamlip_proj_dim, siamlip_hidden_dim, bias=False),
                utils.infer_batchnorm_class()(siamlip_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(siamlip_hidden_dim, siamlip_proj_dim),
            )
        else:
            self.image_text_predictor_local = self.image_text_predictor_global
            self.text_image_predictor_local = self.text_image_predictor_global


    def forward(self, image, text, *image_local):
        out = super().forward(image, text, *image_local)

        out['v_image'] = self.siamlip_image_projector_global(out['feats_image'])
        out['p_image'] = self.image_text_predictor_global(out['v_image'])
        out['v_text'] = self.siamlip_text_projector_global(out['feats_text'])
        out['p_text'] = self.text_image_predictor_global(out['v_text'])

        out['v_image_local'] = [self.siamlip_image_projector_local(l) for l in out['feats_image_local']]
        out['p_image_local'] = [self.image_text_predictor_local(l) for l in out['v_image_local']]
        out['v_text_local'] = [self.siamlip_text_projector_local(l) for l in out['feats_text_local']]
        out['p_text_local'] = [self.text_image_predictor_local(l) for l in out['v_text_local']]

        return out


class SWALIPV1(CLIP):
    def __init__(
        self,
        swalip_proj_dim,
        swalip_hidden_dim,
        swalip_num_proto,
        swalip_no_shared_proto,
        swalip_temperature,
        swalip_learn_temperature,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.swalip_image_projector = nn.Sequential(
            nn.Linear(kwargs['vision_width'], swalip_hidden_dim),
            utils.infer_batchnorm_class()(swalip_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(swalip_hidden_dim, swalip_proj_dim)
        )
        self.swalip_text_projector = nn.Sequential(
            nn.Linear(kwargs['transformer_width'], swalip_hidden_dim),
            utils.infer_batchnorm_class()(swalip_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(swalip_hidden_dim, swalip_proj_dim)
        )

        # prototypes
        if swalip_no_shared_proto:
            self.image_prototypes = self.create_prototypes(swalip_proj_dim, swalip_num_proto)
            self.text_prototypes = self.create_prototypes(swalip_proj_dim, swalip_num_proto)
        else:
            self.image_prototypes = self.create_prototypes(swalip_proj_dim, swalip_num_proto)
            self.text_prototypes = self.image_prototypes

        self.swalip_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / swalip_temperature))
        self.swalip_logit_scale.requires_grad = swalip_learn_temperature

    def create_prototypes(self, swalip_proj_dim, swalip_num_proto):
        prototypes = nn.utils.weight_norm(nn.Linear(swalip_proj_dim, swalip_num_proto, bias=False))
        prototypes.weight_g.data.fill_(1)
        prototypes.weight_g.requires_grad = False
        return prototypes

    def encode_image(self, image):
        out = super().encode_image(image)
        h_image = self.swalip_image_projector(out['feats_image'])
        p_image = self.image_prototypes(F.normalize(h_image))
        return {**out, 'h_image': h_image, 'p_image': p_image}

    def encode_text(self, text):
        out = super().encode_text(text)
        h_text = self.swalip_text_projector(out['feats_text'])
        p_text = self.text_prototypes(F.normalize(h_text))
        return {**out, 'h_text': h_text, 'p_text': p_text}

    def forward(self, image, text):
        return {
            **super().forward(image, text),
            'swalip_logit_scale': self.swalip_logit_scale.exp(),
        }


def get_model(args, **kwargs):
    arch, model_name = args.model.rsplit('_', 1)
    model_class = {
        'BARLIP': BARLIP,
        'SWALIP': CL2L,
        'SWALIPV1': SWALIPV1,
        'SIAMLIP': SIAMLIP,
        'CLIP': CLIP,
        'CL2L': CL2L,
    }[model_name]
    model = globals()[arch](model_class, **vars(args), **kwargs)
    return model


def get_loss(args):
    if args.model.startswith('CLIP'):
        if args.model.endswith('SWALIPV1'):
            return losses.SwALIPV1Loss(
                sk_iters=args.sk_iters,
                target_epsilon=args.target_epsilon,
                swalip_weight=args.swalip_weight,
                temperature=args.swalip_temperature,
            )
        else:
            return losses.CLIPLoss()
    if args.model.startswith('CL2L'):
        if args.model.endswith('BARLIP'):
            return losses.BarLIPLoss(
                loss_avg_or_sum=args.loss_avg_or_sum,
                label_smoothing=args.label_smoothing,
                lamb=args.barlip_lamb,
                scale_loss=args.barlip_scale_loss,
            )
        elif args.model.endswith('SIAMLIP'):
            return losses.SiamLIPLoss(
                loss_avg_or_sum=args.loss_avg_or_sum,
                label_smoothing=args.label_smoothing,
            )
        elif args.model.endswith('SWALIP'):
            return losses.SwALIPLoss(
                loss_avg_or_sum=args.loss_avg_or_sum,
                label_smoothing=args.label_smoothing,
                sk_iters=args.sk_iters,
                target_epsilon=args.target_epsilon,
                swalip_weight=args.swalip_weight,
            )
        else:
            return losses.CL2LLoss(
                loss_avg_or_sum=args.loss_avg_or_sum,
                label_smoothing=args.label_smoothing
            )


def get_metric_names(model):
    parent_model, _, child_model = model.split('_')
    parent_metric_names = {
        'CL2L': ['loss', 'clip_loss', 'clip_loss_image', 'clip_loss_text', 'clip_loss_image_global', 'clip_loss_text_global', 'clip_loss_image_local', 'clip_loss_text_local', 'clip_acc', 'clip_acc_image_local', 'clip_acc_text_local', 'clip_acc_image_global', 'clip_acc_text_global', 'h_logit_scale'],
        'CLIP': ['loss', 'clip_loss', 'clip_acc'],
    }[parent_model]
    child_metric_names = {
        'BARLIP': ['barlip_loss'],
        'SWALIP': ['swalip_loss'],
        'SIAMLIP': ['siamlip_loss'],
        'CLIP': ['clip_loss', 'clip_acc'],
        'CL2L': ['clip_loss', 'clip_acc'],
    }[child_model]
    return sorted(set(parent_metric_names + child_metric_names))


@timm.models.registry.register_model
def vit_small_mocov3_patch16_224(**kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=12, **kwargs)
    model = vit._create_vision_transformer('vit_small_patch16_224', **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = vit._create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_tiny_patch16_384(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16) @ 384x384.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = vit._create_vision_transformer('vit_tiny_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_small_patch32_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32)
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = vit._create_vision_transformer('vit_small_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_small_patch32_384(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32) at 384x384.
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = vit._create_vision_transformer('vit_small_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = vit._create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_small_patch16_384(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = vit._create_vision_transformer('vit_small_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_base_patch32_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = vit._create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = vit._create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = vit._create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = vit._create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_base_patch8_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = vit._create_vision_transformer('vit_base_patch8_224', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_large_patch32_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = vit._create_vision_transformer('vit_large_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = vit._create_vision_transformer('vit_large_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = vit._create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = vit._create_vision_transformer('vit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_large_patch14_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/14)
    """
    model_kwargs = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = vit._create_vision_transformer('vit_large_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_huge_patch14_224(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model = vit._create_vision_transformer('vit_huge_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_giant_patch14_224(pretrained=False, **kwargs):
    """ ViT-Giant (little-g) model (ViT-g/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_kwargs = dict(patch_size=14, embed_dim=1408, mlp_ratio=48/11, depth=40, num_heads=16, **kwargs)
    model = vit._create_vision_transformer('vit_giant_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@timm.models.registry.register_model
def vit_gigantic_patch14_224(pretrained=False, **kwargs):
    """ ViT-Gigantic (big-G) model (ViT-G/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_kwargs = dict(patch_size=14, embed_dim=1664, mlp_ratio=64/13, depth=48, num_heads=16, **kwargs)
    model = vit.vit._create_vision_transformer('vit_gigantic_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


def CL2L_CNEXTT(model_class, **kwargs):
    vision_model = timm.create_model('convnext_tiny', num_classes=0)
    if dist.is_available() and dist.is_initialized():
        vision_model = nn.SyncBatchNorm.convert_sync_batchnorm(vision_model)
    model = model_class(vision_width=vision_model.num_features, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, **kwargs)
    return model


def CL2L_VITS16MOCO(model_class, attn_layer, **kwargs):
    vision_model = timm.create_model('vit_small_mocov3_patch16_224', num_classes=0, attn_layer=attn_layer)
    model = model_class(vision_width=384, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, **kwargs)
    return model


def CL2L_VITS16(model_class, attn_layer, **kwargs):
    vision_model = timm.create_model('vit_small_patch16_224', num_classes=0, attn_layer=attn_layer)
    model = model_class(vision_width=384, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, **kwargs)
    return model


def CL2L_VITS32(model_class, attn_layer, **kwargs):
    vision_model = timm.create_model('vit_small_patch32_224', num_classes=0, attn_layer=attn_layer)
    model = model_class(vision_width=384, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, **kwargs)
    return model


def CL2L_R50(model_class, **kwargs):
    vision_model = timm.create_model('resnet50', num_classes=0)
    if dist.is_available() and dist.is_initialized():
        vision_model = nn.SyncBatchNorm.convert_sync_batchnorm(vision_model)
    model = model_class(vision_width=2048, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, **kwargs)
    return model


def CLIP_R50(model_class, **kwargs):
    vision_model = timm.create_model('resnet50', num_classes=0)
    if dist.is_available() and dist.is_initialized():
        vision_model = nn.SyncBatchNorm.convert_sync_batchnorm(vision_model)
    model = model_class(vision_width=2048, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, **kwargs)
    return model


def CL2L_CNEXTS(model_class, **kwargs):
    vision_model = timm.create_model('convnext_small', num_classes=0)
    if dist.is_available() and dist.is_initialized():
        vision_model = nn.SyncBatchNorm.convert_sync_batchnorm(vision_model)
    model = model_class(vision_width=vision_model.num_features, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, **kwargs)
    return model


def CLIP_VITB16(model_class, attn_layer, **kwargs):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0, attn_layer=attn_layer)
    model = model_class(vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, **kwargs)
    return model


def CL2L_VITB32(model_class, attn_layer, embed_dim=512, **kwargs):
    vision_model = timm.create_model('vit_base_patch32_224', num_classes=0, attn_layer=attn_layer)
    model = model_class(embed_dim=embed_dim, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, **kwargs)
    return model


def CL2L_VITB16(model_class, attn_layer, embed_dim=512, **kwargs):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0, attn_layer=attn_layer)
    model = model_class(embed_dim=embed_dim, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, **kwargs)
    return model


def CLIP_VITL16(model_class, attn_layer, embed_dim=512, **kwargs):
    vision_model = timm.create_model('vit_large_patch16_224', num_classes=0, attn_layer=attn_layer)
    model = model_class(embed_dim=embed_dim, vision_width=1024, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, **kwargs)
    return model


def CL2L_VITL16(model_class, attn_layer, embed_dim=512, **kwargs):
    vision_model = timm.create_model('vit_large_patch16_224', num_classes=0, attn_layer=attn_layer)
    model = model_class(embed_dim=embed_dim, vision_width=1024, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, **kwargs)
    return model
