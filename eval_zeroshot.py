# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import OrderedDict, defaultdict
import json
import os
from sklearn import metrics
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms

import datasets
import models
from tokenizer import SimpleTokenizer
import utils


def get_args_parser():
    parser = argparse.ArgumentParser(description='SLIP 0-shot evaluations', add_help=False)
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--batch-size', default=256, type=int, help='batch_size')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--model-name', type=str, default='')
    return parser


def main(args):
    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        ckpt_path = args.resume
    elif os.path.isfile(os.path.join(args.output_dir, 'checkpoint_best.pt')):
        ckpt_path = os.path.join(args.output_dir, 'checkpoint_best.pt')
    else:
        raise Exception('no checkpoint found')

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    
    # create model
    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    model = models.get_model(old_args, rand_embed=False)
    model.cuda()
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))

    cudnn.benchmark = True

    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, 'dataset_catalog.json')) as f:
        catalog = json.load(f)

    with open(os.path.join(cwd, 'templates.json')) as f:
        all_templates = json.load(f)

    with open(os.path.join(cwd, 'labels.json')) as f:
        all_labels = json.load(f)

    # Data loading code
    print("=> creating dataset")
    tokenizer = SimpleTokenizer(bpe_path=old_args.bpe_path)
    val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    results = {}
    for d in catalog:

        if d == 'kinetics700_frames':
            continue

        print('Evaluating {}'.format(d))
        val_dataset = datasets.get_downstream_dataset(catalog, name=d, is_train=False, transform=val_transform)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False)

        templates = all_templates[d]
        labels = all_labels[d]

        accs = validate_zeroshot(val_loader, templates, labels, model, tokenizer, d, old_args)

        results[d] = accs

        print('metric:', accs)

    print('All results:')
    for d, x in results.items():
        print('{}:\n{}'.format(d, x))
    
    res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zeroshot_results')
    os.makedirs(res_dir, exist_ok=True)
    exp_id = os.path.basename(args.output_dir)
    ckpt_name = os.path.basename(ckpt_path).rsplit('.', 1)[0]
    with open('{}/{}_{}_{}.txt'.format(res_dir, args.model_name, exp_id, ckpt_name), 'w') as f:
        f.write(json.dumps(results))

def validate_zeroshot(val_loader, templates, labels, model, tokenizer, dataset_name, args):
    # switch to evaluate mode
    model.eval()

    is_acc = dataset_name not in ['aircraft', 'pets', 'caltech101', 'flowers', 'kinetics700_frames', 'hateful_memes']
    print('is_acc:', is_acc)

    ensemble_weights = np.linspace(0.1, 0.9, 9).round(decimals=1)
    results = defaultdict(lambda: defaultdict(int if is_acc else list))

    with torch.no_grad():
        print('=> encoding captions')
        text_features = defaultdict(list)
        for label in tqdm(labels):
            if isinstance(label, list):
                texts = [t.format(l) for t in templates for l in label]
            else:
                texts = [t.format(label) for t in templates]
            texts = tokenizer(texts).cuda(non_blocking=True)
            texts = texts.view(-1, 77).contiguous()
            class_embeddings = utils.get_model_parallel(model).encode_text_val(texts)
            embed_names = {'z_text', 'h_text'}.intersection(class_embeddings.keys())
            for embed_name in embed_names:
                cls_embed = class_embeddings[embed_name]
                cls_embed = cls_embed / cls_embed.norm(dim=-1, keepdim=True)
                cls_embed = cls_embed.mean(dim=0)
                cls_embed = cls_embed / cls_embed.norm(dim=-1, keepdim=True)
                text_features[embed_name].append(cls_embed)
        text_features = {k: torch.stack(v, dim=0) for k, v in text_features.items()}

        print('=> encoding images')
        for images, target in tqdm(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # encode images
            image_features = utils.get_model_parallel(model).encode_image_val(images)

            # cosine similarity as logits
            similarities = utils.get_model_parallel(model).predict_zeroshot(image_features, text_features)

            # measure accuracy
            for name, similarity in similarities.items():
                if is_acc:
                    # measure accuracy and record loss
                    pred = similarity.argmax(dim=1)
                    correct = pred.eq(target).sum()
                    results[f'acc1_{name[0]}']['correct'] += correct.item()
                    results[f'acc1_{name[0]}']['total'] += images.size(0)
                else:
                    results[name[0]]['outputs'].append(similarity.cpu())
                    results[name[0]]['targets'].append(target.cpu())

            if is_acc and not args.model.startswith('CLIP'):
                # ensemble accuracy
                for w in ensemble_weights:
                    similarity = w * similarities['z_sim'] + (1-w) * similarities['h_sim']

                    # measure accuracy and record loss
                    pred = similarity.argmax(dim=1)
                    correct = pred.eq(target).sum()
                    results[f'acc1_zh_{w}']['correct'] += correct.item()
                    results[f'acc1_zh_{w}']['total'] += images.size(0)

    if is_acc:
        return {k: 100 * d['correct'] / d['total'] for k, d in results.items()}
    else:
        results = {
            k: (torch.cat(d['outputs']), torch.cat(d['targets'])) 
            for k, d in results.items()
        }

        results = {**results, **{
            f'zh_{w}': (w * results['z'][0] + (1-w) * results['h'][0], results['z'][1])
            for w in ensemble_weights
        }}

        if dataset_name in ['aircraft', 'pets', 'caltech101', 'flowers']:
            return {k: mean_per_class(*r) for k, r in results.items()}
        elif dataset_name == 'kinetics700_frames':
            return {k: (sum(accuracy(*r, topk=(1, 5))) / 2).item() for k, r in results.items()}
        elif dataset_name == 'hateful_memes':
            return {k: roc_auc(*r) for k, r in results.items()}


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mean_per_class(outputs, targets):
    pred = outputs.argmax(1)
    confusion_matrix = metrics.confusion_matrix(targets, pred)
    per_classes = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)

    return 100 * per_classes.mean()


def roc_auc(outputs, targets):
    pos_score = outputs[:, 1] - outputs[:, 0]
    metric = metrics.roc_auc_score(targets, pos_score)

    return 100 * metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SLIP 0-shot evaluations', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
