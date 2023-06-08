# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import utils

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        image_embed = outputs['z_image']
        text_embed = outputs['z_text']
        logit_scale = outputs['logit_scale']
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs
        image_embed_all, text_embed_all = \
            utils.all_gather_batch([image_embed, text_embed])

        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()

        loss = (F.cross_entropy(logits_per_image, self.labels) + \
            F.cross_entropy(logits_per_text, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'clip_loss': loss, 'clip_acc': acc}


class CL2LLoss(nn.Module):
    def __init__(self, loss_avg_or_sum, label_smoothing):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self.loss_avg_or_sum = loss_avg_or_sum
        self.label_smoothing = label_smoothing

    def forward(self, outputs):
        z_image_global = outputs['z_image_global']
        z_text_global = outputs['z_text_global']
        h_image_local = outputs['h_image_local']
        h_text_local = outputs['h_text_local']
        logit_scale = outputs['logit_scale']
        h_logit_scale = outputs['h_logit_scale']
        local_batch_size = z_image_global.size(0)
        assert len(h_image_local) == len(h_text_local)
        num_augs = len(h_image_local)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=z_image_global.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        z_image_global = F.normalize(z_image_global)
        z_text_global = F.normalize(z_text_global)
        h_image_local = [F.normalize(z) for z in h_image_local]
        h_text_local = [F.normalize(z) for z in h_text_local]

        # gather features from all GPUs
        z_image_global_all, z_text_global_all = utils.all_gather_batch([z_image_global, z_text_global])
        h_image_local_all = utils.all_gather_batch(h_image_local)
        h_text_local_all = utils.all_gather_batch(h_text_local)

        # compute global loss
        image_global_logits = logit_scale * z_image_global @ z_text_global_all.t()
        text_global_logits = logit_scale * z_text_global @ z_image_global_all.t()
        clip_loss_image_global = F.cross_entropy(image_global_logits, self.labels)
        clip_loss_text_global = F.cross_entropy(text_global_logits, self.labels)

        # compute local loss
        clip_loss_image_local, clip_loss_text_local = 0, 0
        if num_augs > 0:
            image_local_logits = []
            text_local_logits = []
            for i in range(num_augs):
                image_local_logits += [h_logit_scale * h @ h_text_local_all[i].t() for h in h_image_local]
                text_local_logits += [h_logit_scale * h @ h_image_local_all[i].t() for h in h_text_local]
                clip_loss_image_local = sum([F.cross_entropy(l, self.labels, label_smoothing=self.label_smoothing) for l in image_local_logits]) / len(image_local_logits)
                clip_loss_text_local = sum([F.cross_entropy(l, self.labels, label_smoothing=self.label_smoothing) for l in text_local_logits]) / len(text_local_logits)            

        # compute total losses
        clip_loss_image = (clip_loss_image_global + clip_loss_image_local * num_augs) / (1 + num_augs)
        clip_loss_text = (clip_loss_text_global + clip_loss_text_local * num_augs) / (1 + num_augs)
        clip_loss = (clip_loss_image + clip_loss_text) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(image_global_logits, dim=-1)
            correct = pred.eq(self.labels).sum()
            clip_acc_image_global = 100 * correct / local_batch_size

            pred = torch.argmax(text_global_logits, dim=-1)
            correct = pred.eq(self.labels).sum()
            clip_acc_text_global = 100 * correct / local_batch_size

            clip_acc_image_local, clip_acc_text_local = 0, 0
            if num_augs > 0:
                for aug_logits in image_local_logits:
                    pred = torch.argmax(aug_logits, dim=-1)
                    correct = pred.eq(self.labels).sum()
                    clip_acc_image_local += 100 * correct / local_batch_size
                clip_acc_image_local /= len(image_local_logits)

                for aug_logits in text_local_logits:
                    pred = torch.argmax(aug_logits, dim=-1)
                    correct = pred.eq(self.labels).sum()
                    clip_acc_text_local += 100 * correct / local_batch_size
                clip_acc_text_local /= len(image_local_logits)

        loss = clip_loss * (2 if self.loss_avg_or_sum == 'sum' else 1)

        clip_local_dict = {
            'clip_loss_image_local': clip_loss_image_local,
            'clip_loss_text_local': clip_loss_text_local,
            'clip_acc_image_local': clip_acc_image_local,
            'clip_acc_text_local': clip_acc_text_local,
        } if num_augs > 0 else {}

        return {
            'loss': loss,
            'clip_loss_image': clip_loss_image,
            'clip_loss_text': clip_loss_text,
            'clip_loss_image_global': clip_loss_image_global,
            'clip_loss_text_global': clip_loss_text_global,
            'clip_loss_image': clip_loss_image,
            'clip_loss': clip_loss,
            'clip_acc': clip_acc_image_global,
            'clip_acc_image_global': clip_acc_image_global,
            'clip_acc_text_global': clip_acc_text_global,
            'h_logit_scale': h_logit_scale,
            **clip_local_dict,
        }


class BarLIPLoss(CL2LLoss):
    def __init__(self, loss_avg_or_sum, label_smoothing, lamb=5e-3, scale_loss=0.025):
        super().__init__(loss_avg_or_sum, label_smoothing)
        self.lamb = lamb
        self.scale_loss = scale_loss

    def barlip_loss(self, z1, z2):

        N, D = z1.size()

        corr = torch.einsum("bi, bj -> ij", z1, z2) / N

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(corr)
            world_size = dist.get_world_size()
            corr /= world_size

        diag = torch.eye(D, device=corr.device)
        cdif = (corr - diag).pow(2)
        cdif[~diag.bool()] *= self.lamb
        loss = self.scale_loss * cdif.sum()
        return loss

    def forward(self, outputs):

        # global to global
        num_losses = 1
        barlip_loss = self.barlip_loss(outputs['v_image'], outputs['v_text'])

        # local to local
        for v_image in outputs['v_image_local']:
            for v_text in outputs['v_text_local']:
                barlip_loss += self.barlip_loss(v_image, v_text)
                num_losses += 1

        barlip_loss /= num_losses

        # online eval with clip loss
        clip_loss_out = super().forward(outputs)
        loss = barlip_loss + clip_loss_out.pop('loss')

        return {
            'loss': loss,
            'barlip_loss': barlip_loss,
            **clip_loss_out
        }


class SiamLIPLoss(CL2LLoss):
    def __init__(self, loss_avg_or_sum, label_smoothing):
        super().__init__(loss_avg_or_sum, label_smoothing)

    def negative_cosine_similarity(self, p, v):
        p = F.normalize(p, dim=-1)
        v = F.normalize(v, dim=-1)
        return 2 - 2 * (p * v.detach()).sum(dim=1).mean()

    def forward(self, outputs):
        p_image_global = outputs['p_image']
        p_text_global = outputs['p_text']
        p_image_local = outputs['p_image_local']
        p_text_local = outputs['p_text_local']

        if any('momentum' in k for k in outputs):
            v_image_global = outputs['v_image_momentum']
            v_text_global = outputs['v_text_momentum']
            v_image_local = outputs['v_image_local_momentum']
            v_text_local = outputs['v_text_local_momentum']
        else:
            v_image_global = outputs['v_image']
            v_text_global = outputs['v_text']
            v_image_local = outputs['v_image_local']
            v_text_local = outputs['v_text_local']

        # global to global
        num_losses = 2
        siamlip_loss = (
            self.negative_cosine_similarity(p_image_global, v_text_global.detach()) + \
            self.negative_cosine_similarity(p_text_global, v_image_global.detach())
        )

        # local to local
        for p in p_image_local:
            for v in v_text_local:
                siamlip_loss += self.negative_cosine_similarity(p, v.detach())
                num_losses += 1
        for p in p_text_local:
            for v in v_image_local:
                siamlip_loss += self.negative_cosine_similarity(p, v.detach())
                num_losses += 1

        siamlip_loss /= num_losses

        # online eval with clip loss
        clip_loss_out = super().forward(outputs)
        loss = siamlip_loss + clip_loss_out.pop('loss')

        return {
            'loss': loss,
            'siamlip_loss': siamlip_loss,
            **clip_loss_out
        }


class SwALIPLoss(CL2LLoss):
    def __init__(
        self,
        loss_avg_or_sum,
        label_smoothing,
        sk_iters=3,
        target_epsilon=0.05,
        swalip_weight=0.2,
    ):
        assert label_smoothing == 0.0
        super().__init__(loss_avg_or_sum, label_smoothing)

        self.sk_iters = sk_iters
        self.target_epsilon = target_epsilon
        self.swalip_weight = swalip_weight
        self.labels = None
        self.last_local_batch_size = None

        self.set_world_size()

    def set_world_size(self):
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

    @torch.no_grad()
    def sinkhorn_knopp(self, Q: torch.Tensor) -> torch.Tensor:
        """Produces assignments using Sinkhorn-Knopp algorithm.
        Applies the entropy regularization, normalizes the Q matrix and then normalizes rows and
        columns in an alternating fashion for num_iter times. Before returning it normalizes again
        the columns in order for the output to be an assignment of samples to prototypes.
        Args:
            Q (torch.Tensor): cosine similarities between the features of the
                samples and the prototypes.
        Returns:
            torch.Tensor: assignment of samples to prototypes according to optimal transport.
        """

        Q = torch.exp(Q / self.target_epsilon).t()
        B = Q.shape[1] * self.world_size
        K = Q.shape[0]  # num prototypes

        # make the matrix sum to 1
        sum_Q = torch.sum(Q)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for _ in range(self.sk_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def cross_entropy(self, logits, targets):
        return -torch.mean(torch.sum(targets * torch.log_softmax(logits, dim=1), dim=1))

    def forward(self, outputs):

        # online eval with clip loss
        clip_loss_out = super().forward(outputs)

        # cl2l
        h_image_local = [F.normalize(h) for h in outputs['h_image_local']]
        h_text_local = [F.normalize(h) for h in outputs['h_text_local']]
        h_logit_scale = outputs['h_logit_scale']
        num_augs = len(h_image_local)

        h_image_local_all = utils.all_gather_batch(h_image_local)
        h_text_local_all = utils.all_gather_batch(h_text_local)

        logits_per_image_local = [[h @ h_all.t() for h_all in h_text_local_all] for h in h_image_local]
        logits_per_text_local = [[h @ h_all.t() for h_all in h_image_local_all] for h in h_text_local]

        # generate pseudo-label
        with torch.no_grad():
            targets_per_image_local = [[self.sinkhorn_knopp(t.detach()) for t in i] for i in logits_per_image_local]
            targets_per_text_local = [[self.sinkhorn_knopp(i.detach()) for i in t] for t in logits_per_text_local]

        # compute the loss between all views
        swalip_loss = 0
        for l1 in range(2):
            for l2 in range(2):
                t1, t2 = abs(l1 - 1), abs(l2 - 1)
                swalip_loss += (
                    self.cross_entropy(logits_per_image_local[l1][l2] * h_logit_scale, targets_per_image_local[t1][t2]) + \
                    self.cross_entropy(logits_per_text_local[l1][l2] * h_logit_scale, targets_per_text_local[t1][t2])
                ) / 2
        swalip_loss /= num_augs ** 2
        loss = self.swalip_weight * swalip_loss + clip_loss_out.pop('loss')

        return {**clip_loss_out, 'loss': loss, 'swalip_loss': swalip_loss}


class SwALIPV1Loss(nn.Module):

    def __init__(self, sk_iters, target_epsilon, temperature, swalip_weight=1.0):
        super().__init__()
        self.sk_iters = sk_iters
        self.target_epsilon = target_epsilon
        self.temperature = temperature
        self.swalip_weight = swalip_weight

        self.clip_loss = CLIPLoss()

        self.set_world_size()

    def set_world_size(self):
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

    @torch.no_grad()
    def sinkhorn_knopp(self, Q: torch.Tensor) -> torch.Tensor:
        """Produces assignments using Sinkhorn-Knopp algorithm.
        Applies the entropy regularization, normalizes the Q matrix and then normalizes rows and
        columns in an alternating fashion for num_iter times. Before returning it normalizes again
        the columns in order for the output to be an assignment of samples to prototypes.
        Args:
            Q (torch.Tensor): cosine similarities between the features of the
                samples and the prototypes.
        Returns:
            torch.Tensor: assignment of samples to prototypes according to optimal transport.
        """

        Q = torch.exp(Q / self.target_epsilon).t()
        B = Q.shape[1] * self.world_size
        K = Q.shape[0]  # num prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for _ in range(self.sk_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def cross_entropy(self, logits, assign):
        return -torch.mean(torch.sum(assign * torch.log_softmax(logits, dim=1), dim=1))

    def forward(self, outputs):
        image_logits = outputs['p_image']
        text_logits = outputs['p_text']
        logit_scale = outputs['swalip_logit_scale']

        if any('momentum' in k for k in outputs):
            image_targets = outputs['p_image_momentum']
            text_targets = outputs['p_text_momentum']
        else:
            image_targets = outputs['p_image'].detach()
            text_targets = outputs['p_text'].detach()

        image_assign = self.sinkhorn_knopp(image_targets.detach())
        text_assign = self.sinkhorn_knopp(text_targets.detach())

        image_logits *= logit_scale
        text_logits *= logit_scale

        swalip_loss = (
            self.cross_entropy(image_logits, text_assign) + \
            self.cross_entropy(text_logits, image_assign)
        ) / 2

        # online eval with clip loss
        clip_loss_out = self.clip_loss(outputs)
        loss = self.swalip_weight * swalip_loss + clip_loss_out.pop('loss')

        return {
            'loss': loss,
            'swalip_loss': swalip_loss,
            'swalip_logit_scale': logit_scale,
            **clip_loss_out
        }
