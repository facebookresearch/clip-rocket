# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import json
import os
import pickle
import zipfile

import numpy as np
from PIL import Image, ImageFile

import torch
from torchvision import transforms
from torchvision import datasets as t_datasets
from torchvision.datasets import ImageFolder

import utils

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def yfcc_loader(root, index):
    index = format(index, "0>8d")
    repo = index[:2]
    z = index[2: 5]
    file_img = index[5:] + '.jpg'
    path_zip = os.path.join(root, 'images', repo, z) + '.zip'
    with zipfile.ZipFile(path_zip, 'r') as myzip:
        img = Image.open(myzip.open(file_img))
    return img.convert('RGB')


class ImageCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, root, metadata, caption_sampling='single'):
        self.dataset = dataset
        self.root = root
        self.caption_sampling = caption_sampling
        if self.dataset == 'yfcc15m':
            with open(metadata, 'rb') as f:
                self.samples = pickle.load(f)
        elif self.dataset == 'coco':
            samples = defaultdict(list)
            with open(metadata) as f:
                annotations = json.load(f)['annotations']
            for ann in annotations:
                samples[ann['image_id']].append(ann['caption'])
            self.samples = [(k, v) for k, v in samples.items()]
        elif self.dataset == 'cc12m' or self.dataset == 'cc3m':
            self.samples = np.load(metadata, allow_pickle=True)
        elif self.dataset == 'merged_opendata':
            self.samples = []
            self.roots = []
            for md, r in zip(metadata.split("---"), root.split("---")):
                self.samples.append(np.load(md, allow_pickle=True))
                self.roots.append(r)
        elif self.dataset == 'redcaps':
            with open(metadata) as f:
                annotations = json.load(f)
            self.samples = [(ann['image_id'], ann['subreddit'], ann['caption']) for ann in annotations]

    def get_raw_item(self, i):
        if self.dataset == 'yfcc15m':
            index, title, desc = self.samples[i]
            caption = [c for c in [title, desc] if c != '']
            caption = [''] if len(caption) == 0 else caption
            caption = tuple(caption if self.caption_sampling == 'multi' else [np.random.choice(caption)])
            img = yfcc_loader(self.root, index)
        elif self.dataset == 'coco':
            index, captions = self.samples[i]
            path = os.path.join(self.root, 'train2017', '{:012d}.jpg'.format(index))
            img = pil_loader(path)
            caption = tuple(captions if self.caption_sampling == 'multi' else [np.random.choice(captions)])
        elif self.dataset == 'cc3m':
            ann = self.samples[i]
            filename, captions = ann['image_id'], ann['captions']
            path = os.path.join(self.root, str(filename))
            img = pil_loader(path)
            caption = tuple(captions if self.caption_sampling == 'multi' else [np.random.choice(captions)])
        elif self.dataset == 'cc12m':
            ann = self.samples[i]
            filename, captions = ann['image_name'], ann['captions']
            path = os.path.join(self.root, filename)
            img = pil_loader(path)
            caption = tuple(captions if self.caption_sampling == 'multi' else [np.random.choice(captions)])
        elif self.dataset == 'merged_opendata':
            datasets = ['cc3m', 'cc12m', 'yfcc15m']
            cum_lens = np.array([len(s) for s in self.samples]).cumsum()
            d_idx = [idx for idx, l in enumerate(cum_lens) if i < l][0]
            offset = cum_lens[d_idx - 1] if d_idx > 0 else 0
            samples_list = self.samples
            self.samples = self.samples[d_idx]
            self.dataset = datasets[d_idx]
            self.root = self.roots[d_idx]
            img, caption = self.get_raw_item(i - offset)
            self.dataset = 'merged_opendata'
            self.samples = samples_list
        elif self.dataset == 'redcaps':
            image_id, subreddit, caption = self.samples[i]
            path = os.path.join(self.root, subreddit, f"{image_id}.jpg")
            img = pil_loader(path)
        elif 'pmd' in self.dataset:
            img, captions = self.pmd[i]
            # if isinstance(captions, str):
            #     caption = captions
            assert isinstance(captions, list)
            caption = tuple(captions if self.caption_sampling == 'multi' else [np.random.choice(captions)])

        return img, caption

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        if 'pmd' in self.dataset:
            return len(self.pmd)
        elif 'merged_opendata' in self.dataset:
            return sum([len(s) for s in self.samples])
        else:
            return len(self.samples)


class ImageCaptionDatasetCLIP(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform=None, tokenizer=None):
        super().__init__(dataset, root, metadata)

        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        # apply transformation
        if self.transform is not None:
            image = self.transform(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return image, caption


class ImageCaptionDatasetCL2L(ImageCaptionDatasetBase):
    def __init__(
        self,
        dataset,
        root,
        metadata,
        transform,
        augment,
        num_augs=2,
        tokenizer=None,
        augs_only=False,
        caption_sampling='single'
    ):
        super().__init__(dataset, root, metadata, caption_sampling=caption_sampling)
        self.transform = transform
        self.num_augs = num_augs
        self.augment = augment if isinstance(augment, list) else [augment] * num_augs
        self.tokenizer = tokenizer
        self.augs_only = augs_only

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        augs = [self.augment[i](img) for i in range(self.num_augs)]

        if self.augs_only:
            return augs

        image = self.transform(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return image, caption, *augs


class FileListDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(images)
        self.labels = np.load(labels)

    def __getitem__(self, index):
        img = pil_loader(self.images[index])
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


def get_downstream_dataset(catalog, name, is_train, transform):
    entry = catalog[name]
    root = entry['path']
    if entry['type'] == 'imagefolder':
        dataset = t_datasets.ImageFolder(os.path.join(root, entry['train'] if is_train else entry['test']),
            transform=transform)
    elif entry['type'] == 'special':
        if name == 'cifar10':
            dataset = t_datasets.CIFAR10(root, train=is_train,
                transform=transform, download=True)
        elif name == 'cifar100':
            dataset = t_datasets.CIFAR100(root, train=is_train,
                transform=transform, download=True)
        elif name == 'stl10':
            dataset = t_datasets.STL10(root, split='train' if is_train else 'test',
                transform=transform, download=True)
        elif name == 'mnist':
            dataset = t_datasets.MNIST(root, train=is_train,
                transform=transform, download=True)
    elif entry['type'] == 'filelist':
        path = entry['train'] if is_train else entry['test']
        val_images = os.path.join(root, path + '_images.npy')
        val_labels = os.path.join(root, path + '_labels.npy')
        if name == 'clevr_counts':
            target_transform = lambda x: ['count_10', 'count_3', 'count_4', 'count_5', 'count_6', 'count_7', 'count_8', 'count_9'].index(x)
        else:
            target_transform = None
        dataset = FileListDataset(val_images, val_labels, transform, target_transform)
    else:
        raise Exception('Unknown dataset')

    return dataset


def get_train_dataset(args, tokenizer, metadata, augs_only=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(args.weak_min_scale, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        normalize
    ])

    augment = transforms.Compose([
        transforms.RandomResizedCrop(
            args.multicrop_resize,
            scale=(0.08, args.multicrop_max_scale),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=args.grayscale_prob),
        transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=args.blur_prob),
        transforms.RandomApply([utils.Solarization()], p=args.solarize_prob),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    if args.byol_augment:
        assert args.num_augs == 2
        augment = []
        asym_blur_prob = [1.0, 0.1]
        asym_solarize_prob = [0.0, 0.2]
        for blur_prob, solarize_prob in zip(asym_blur_prob, asym_solarize_prob):
            augment.append(transforms.Compose([
                transforms.RandomResizedCrop(
                    args.multicrop_resize,
                    scale=(0.08, args.multicrop_max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=blur_prob),
                transforms.RandomApply([utils.Solarization()], p=solarize_prob),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    assert not (args.weak_augment and args.strong_augment)
    if args.weak_augment:
        augment = train_transform
    if args.strong_augment:
        train_transform = augment
    if args.randaugment:
        train_transform = transforms.RandomChoice([train_transform, augment])

    if args.model.startswith('CLIP'):
        return ImageCaptionDatasetCLIP(args.dataset, args.root, metadata, train_transform, tokenizer)
    elif args.model.startswith('CL2L'):
        return ImageCaptionDatasetCL2L(
            args.dataset,
            args.root,
            metadata,
            train_transform,
            augment,
            args.num_augs,
            tokenizer=tokenizer,
            augs_only=augs_only,
            caption_sampling=args.caption_sampling
        )


def get_val_dataset():

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, 'dataset_catalog.json')) as f:
        root = json.load(f)['imagenet']['path']
    return ImageFolder(os.path.join(root, 'val'), val_transform)
