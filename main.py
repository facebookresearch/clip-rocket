# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import OrderedDict, defaultdict
import json
import math
import os
import sys
import time
from tkinter import E
try:
    import wandb
except ImportError:
    print("wandb not found")

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets
import models
import utils
from tokenizer import SimpleTokenizer


def get_args_parser():
    parser = argparse.ArgumentParser(description='CL2L training and evaluation', add_help=False)
    # Data
    parser.add_argument('--dataset', default='yfcc15m', type=str, choices=['yfcc15m', 'cc3m', 'cc12m', 'merged_opendata'])
    parser.add_argument('--root', default='datasets/yfcc100m', type=str,
                        help='path to dataset root')
    parser.add_argument('--metadata', default='datasets/yfcc15m_v1/yfcc15m.pkl', type=str,
                        help='path to metadata file (see README for details)')
    parser.add_argument('--metadata-unpaired', default='datasets/yfcc15m_v1/yfcc85m.pkl', type=str,
                        help='path to metadata file (see README for details)')
    parser.add_argument('--bpe-path', default='datasets/yfcc15m_v1/bpe_simple_vocab_16e6.txt.gz', type=str,
                        help='path to the bpe file (see README for details)')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')

    # Model
    parser.add_argument('--model', default='CL2L_VITB16', type=str)
    parser.add_argument('--attn-layer', default='flash', type=str, choices=["flash", "standard"])
    parser.add_argument('--no-share-token', action='store_true')
    parser.add_argument('--semi-paired', action='store_true')
    parser.add_argument('--unpaired-ratio', default=4, type=int)
    parser.add_argument('--embed-dim', default=256, type=int,
                        help='output dim for the language-image projector')
    parser.add_argument('--clip-hidden-dim', default=4096, type=int,
                        help='hidden dim of CLIP mlp projection head')
    parser.add_argument('--ssl-scale', default=1.0, type=float,
                        help='loss scale for SimCLR objective')
    parser.add_argument('--ssl-temp', default=0.1, type=float,
                        help='softmax temperature for SimCLR objective')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    parser.add_argument('--init-text-encoder', default=None, type=str, help='path to init from')
    parser.add_argument('--detach-proj', action='store_true',
                        help='whether or not to detach the clip projector')
    parser.add_argument('--momentum', action='store_true',
                        help='whether or not to use the momentum encoder')
    parser.add_argument('--momentum-tau', type=float, default=0.99,
                        help='whether or not to use the momentum encoder')
    parser.add_argument('--transformer-layers', default=12, type=int,
                        help='number of layers for the text transformer')
    parser.add_argument('--clip-proj-type', default='linear', type=str,
                        choices=['mlp', 'linear'], help='type of projector for clip')
    parser.add_argument('--cl2l-txt-proj-type', default='mlp', type=str,
                        choices=['mlp', 'linear'], help='type of text projector for cl2l')
    parser.add_argument('--cl2l-img-proj-type', default='mlp', type=str,
                        choices=['mlp', 'linear'], help='type of vision projector for cl2l')
    parser.add_argument('--separate-proj', default=False, action='store_true',
                        help='different heads')
    parser.add_argument('--separate-proj-child', default=False, action='store_true',
                        help='different heads in non-contrastive stream')
    
    # BarLIP
    parser.add_argument('--barlip-proj-dim', default=8192, type=int,
                        help='output dim for the barlip projector')
    parser.add_argument('--barlip-hidden-dim', default=3000, type=int,
                        help='hidden dim for the barlip projector')
    parser.add_argument('--barlip-lamb', default=5e-3, type=float,
                        help='lambda for BarLIP loss')
    parser.add_argument('--barlip-scale-loss', default=0.025, type=float,
                        help='loss scaling factor for BarLIP')

    # SwALIP
    parser.add_argument('--swalip-proj-dim', default=128, type=int,
                        help='output dim for the swalip projector')
    parser.add_argument('--swalip-hidden-dim', default=2048, type=int,
                        help='output dim for the swalip projector')
    parser.add_argument('--swalip-num-proto', default=3000, type=int,
                        help='number of prototypes for swalip')
    parser.add_argument('--swalip-temperature', default=0.1, type=float,
                        help='softmax temperature for swalip')
    parser.add_argument('--swalip-learn-temperature', action='store_true',
                        help='whether to learn softmax temperature for swalip')
    parser.add_argument('--sk-iters', default=3, type=int,
                        help='output dim for the swalip projector')
    parser.add_argument('--target-epsilon', default=0.05, type=float,
                        help='output dim for the swalip projector')
    parser.add_argument('--swalip-freeze-proto-iters', default=100, type=int,
                        help='number of iters to freeze swalip prototypes')
    parser.add_argument('--swalip-no-shared-proto', action='store_true',
                        help='whether or not to share prototypes between modalities')
    parser.add_argument('--swalip-weight', default=0.2, type=float,
                        help='weight for the swalip loss')
    # SiamLIP
    parser.add_argument('--siamlip-proj-dim', default=128, type=int,
                        help='output dim for the siamlip projector')
    parser.add_argument('--siamlip-hidden-dim', default=2048, type=int,
                        help='output dim for the siamlip projector')
    parser.add_argument('--siamlip-no-last-bn', action='store_true',
                        help='whether to use batchnorm at the end of the proj')

    # Image Augmentations
    parser.add_argument('--num-augs', default=2, type=int,
                        help='number of augmentations in cl2l')
    parser.add_argument('--multicrop-resize', default=224, type=int)
    parser.add_argument('--multicrop-max-scale', default=1.0, type=float)
    parser.add_argument('--weak-min-scale', default=0.5, type=float)
    parser.add_argument('--blur-prob', default=0.5, type=float)
    parser.add_argument('--solarize-prob', default=0.0, type=float)
    parser.add_argument('--grayscale-prob', default=0.2, type=float)
    parser.add_argument('--byol-augment', default=False, action='store_true',
                        help='byol-like asymmetric augment. It overrides other augment probs')
    parser.add_argument('--weak-augment', default=False, action='store_true',
                        help='make all augmentations weak, including the ones of cl2l')
    parser.add_argument('--strong-augment', default=False, action='store_true',
                        help='make all augmentations strong, including the one of baseline clip')
    parser.add_argument('--randaugment', default=False, action='store_true',
                        help='add randaugment to base augmentations')

    # Text Augmentations
    parser.add_argument('--caption-sampling', default='single', type=str,
                        choices=['single', 'multi'], help='how to sample captions')    
    parser.add_argument('--text-dropout-prob', default=0.0, type=float,
                        help='dropout probability')
    parser.add_argument('--text-drop-path-prob', default=0.0, type=float,
                        help='dropout probability')
    parser.add_argument('--label-smoothing', default=0.0, type=float,
                        help='label smoothing')
    parser.add_argument('--text-augment', default=False, action='store_true',
                        help='text augmentations')
    parser.add_argument('--clean-before-augment', default=False, action='store_true',
                        help='clean before text augmentations')
    parser.add_argument('--no-text-augment-prob', default=0.0, type=float,
                        help='prob not to augment text')
    parser.add_argument('--remove-stopwords-prob', default=0.8, type=float,
                        help='prob to remove stopwords from text')
    parser.add_argument('--synonym-replacement-prob', default=0.4, type=float,
                        help='prob to replace synonym in text')
    parser.add_argument('--random-swap-prob', default=0.4, type=float,
                        help='prob to randomly swap in text')
    parser.add_argument('--random-deletion-prob', default=0.2, type=float,
                        help='prob to randomly delete text')

    # Training
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=64, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--loss-avg-or-sum', default='avg', type=str)
    parser.add_argument('--checkpoint-grad', action='store_true',
                        help='enable gradient checkpointing')

    # System
    parser.add_argument('--print-freq', default=25, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--workers-unpaired', default=5, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument('--offline', action='store_true', help='WandB will not log online')
    parser.add_argument('--name', default='CLIP_ROCKET', type=str)
    return parser

best_acc1 = 0


def main(args):
    utils.init_distributed_mode(args)

    global best_acc1

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create model
    print("=> creating model: {}".format(args.model))
    model = models.get_model(args)
    model.visual.set_grad_checkpointing(enable=args.checkpoint_grad)
    model.cuda(args.gpu)
    print(model)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200, static_graph=True)

    if args.momentum:
        momentum_model = models.get_model(args, is_momentum=True)
        momentum_model.cuda(args.gpu)

        if args.distributed:
            momentum_model = torch.nn.parallel.DistributedDataParallel(
                momentum_model, device_ids=[args.gpu], bucket_cap_mb=200, static_graph=True)

        msg = utils.get_model_parallel(momentum_model).load_state_dict(
            utils.get_model_parallel(model).state_dict(), strict=False)
        print(msg)

        for p in momentum_model.parameters():
            p.requires_grad = False

    # define loss function (criterion) and optimizer
    criterion = models.get_loss(args).cuda(args.gpu)

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                    eps=args.eps, weight_decay=args.wd)
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    # optionally load pre-trained text encoder
    if args.init_text_encoder is not None:
        cp_text_encoder = torch.load(args.init_text_encoder)['state_dict']
        cp_text_encoder = {k.replace('module.', ''): v for k, v in cp_text_encoder.items() if 'transformer' in k}
        result = utils.get_model_parallel(model).load_state_dict(cp_text_encoder, strict=False)
        print(result)
        del cp_text_encoder

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch
            result = model.load_state_dict(
                checkpoint['state_dict'], strict=False)
            print(result)
            if args.momentum:
                print("=> loading momentum encoder from '{}'".format(args.resume))
                result = momentum_model.load_state_dict(
                    checkpoint['momentum_state_dict'], strict=False)
                print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            best_acc1 = checkpoint['best_acc1']
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(args.resume, epoch))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            if args.momentum:
                momentum_model.load_state_dict(latest_checkpoint['momentum_state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            best_acc1 = latest_checkpoint['best_acc1']
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))
            del latest_checkpoint

    cudnn.benchmark = True

    # build tokenizer
    tokenizer = SimpleTokenizer(
        bpe_path=args.bpe_path,
        text_augment=args.text_augment,
        no_text_augment_prob=args.no_text_augment_prob,
        remove_stopwords_prob=args.remove_stopwords_prob,
        synonym_replacement_prob=args.synonym_replacement_prob,
        random_swap_prob=args.random_swap_prob,
        random_deletion_prob=args.random_deletion_prob,
        clean_before_augment=args.clean_before_augment,
        num_augs=args.num_augs,
    )

    # build datasets
    print("=> creating paired datasets")
    train_dataset = datasets.get_train_dataset(args, tokenizer, metadata=args.metadata)
    val_dataset = datasets.get_val_dataset()

    # dist eval resamples data to pad uneven batch sizes
    # make sure num_samples = 0 mod num_gpus for exact acc
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    val_sampler = DistributedSampler(val_dataset) if args.distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=(val_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False
    )

    # optionally also load unpaired data
    if args.semi_paired:
        print("=> creating unpaired dataset")
        unpaired_dataset = datasets.get_train_dataset(
            args,
            tokenizer,
            metadata=args.metadata_unpaired,
            augs_only=False
        )
        unpaired_sampler = DistributedSampler(unpaired_dataset) if args.distributed else None
        unpaired_loader = DataLoader(
            unpaired_dataset,
            batch_size=args.batch_size // args.unpaired_ratio,
            shuffle=(unpaired_sampler is None),
            num_workers=args.workers_unpaired,
            pin_memory=True,
            sampler=unpaired_sampler,
            drop_last=True
        )
        unpaired_iterable = utils.cycle(unpaired_loader, unpaired_sampler)

    if args.evaluate:
        zero_stats = validate_zeroshot(val_loader, model, tokenizer, args)
        if utils.is_main_process():
            with open(os.path.join(args.output_dir, 'eval_log.txt'), 'a') as f:
                f.write(json.dumps(zero_stats) + '\n')
        return

    lr_schedule = utils.cosine_scheduler(args.lr, args.lr_end, args.epochs,
        len(train_loader) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)

    if utils.is_main_process() and args.output_dir != './':
        with open(os.path.join(args.output_dir, 'command.txt'), 'w') as f:
            f.write(' '.join(sys.argv))
        json.dump(
            vars(args),
            open(os.path.join(args.output_dir, 'args.json'), 'w'),
            default=lambda o: "<not serializable>",
            indent=4
        )
        if args.wandb:
            wandb_id = os.path.split(args.output_dir)[-1]
            wandb.init(
                project='clip-rocket',
                id=wandb_id,
                config=args,
                resume='allow',
                name=args.name,
                save_code=True,
                notes=' '.join(sys.argv),
                mode='offline' if args.offline else 'online',
                dir=args.output_dir
            )
            wandb.watch(model)

    print(args)

    print("=> beginning training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_stats = train(
            train_loader,
            model, criterion,
            optimizer,
            scaler,
            epoch,
            lr_schedule,
            args,
            momentum_model if args.momentum else None,
            unpaired_iterable if args.semi_paired else None
        )

        if (epoch + 1) % args.eval_freq != 0:
            continue

        val_stats = validate_zeroshot(val_loader, model, tokenizer, args)
        acc1 = val_stats['acc1_z']

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        print("=> saving checkpoint")
        checkpoint_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'best_acc1': best_acc1,
            'args': args,
        }
        if args.momentum:
            checkpoint_dict['momentum_state_dict'] = momentum_model.state_dict()
        utils.save_on_master(checkpoint_dict, is_best, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}

        if utils.is_main_process():
            if args.wandb:
                wandb.log(log_stats)
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    if utils.is_main_process():
        wandb.finish()


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    scaler,
    epoch,
    lr_schedule,
    args,
    momentum_model=None,
    unpaired_iterable=None,
):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = models.get_metric_names(args.model)
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    assert (momentum_model is not None) == args.momentum

    # switch to train mode
    model.train()

    if args.momentum:
        momentum_model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # optionally load unpaired data
        if args.semi_paired:
            inputs_unpaired = next(unpaired_iterable)
            inputs = [
                torch.cat([inputs[0], inputs_unpaired[0]]),
                inputs[1],
                *[torch.cat([inputs[a+2], inputs_unpaired[a+2]])
                for a in range(args.num_augs)]
            ]

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]

        inputs = [t.cuda(args.gpu, non_blocking=True) for t in inputs]

        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(*inputs)

            if args.momentum:
                with torch.no_grad():
                    momentum_outputs = momentum_model(*inputs)
                momentum_outputs = {k + '_momentum' : v for k, v in momentum_outputs.items()}
                outputs = {**outputs, **momentum_outputs}

            loss_dict = criterion(outputs)
            loss = loss_dict['loss']
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            torch.save(
                {"inputs": utils.all_gather_batch(inputs), "outputs": utils.all_gather_batch(outputs), "losses": loss_dict, "state_dict": model.state_dict()},
                os.path.join(args.output_dir, "dump_loss_nan.pgz")
            )
            print("Loss is {}, stopping training".format(loss.item()))
            time.sleep(5)
            sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        if args.model.endswith('SWALIPV1') and it < args.swalip_freeze_proto_iters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None

        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # momentum update
        if args.momentum:
            with torch.no_grad():
                m = args.momentum_tau
                for p, p_mom in zip(
                    utils.get_model_parallel(model).parameters(),
                    utils.get_model_parallel(momentum_model).parameters()
                ):
                    p_mom.data.mul_(m).add_((1 - m) * p.detach().data)

        # clamp logit scale to [0, 100]
        utils.get_model_parallel(model).logit_scale.data.clamp_(0, 4.6052)
        logit_scale = utils.get_model_parallel(model).logit_scale.exp().item()
        utils.get_model_parallel(model).l2l_logit_scale.data.clamp_(0, 4.6052)

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if utils.is_main_process() and args.wandb:
                wandb.log({**{k: v.item() for k, v in loss_dict.items()},
                        'scaler': scaler.get_scale(),
                        'logit': logit_scale})
            progress.display(optim_iter)

    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}


def validate_zeroshot(val_loader, model, tokenizer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    if args.model.startswith('SLIP') or args.model.startswith('CLIP'):
        metrics = {
            'acc1_z': AverageMeter('Acc@1_z', ':6.2f'),
            'acc5_z': AverageMeter('Acc@5_z', ':6.2f')
        }
    else:
        ensemble_weights = np.linspace(0.1, 0.9, 9).round(decimals=1)
        metric_suffixes = ['z', 'h'] + [f'zh_{w}' for w in ensemble_weights]
        metrics = {
            f'acc{k}_{s}': AverageMeter(f'Acc@{k}_{s}', ':6.2f')
            for s in metric_suffixes for k in [1, 5]
        }
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, *metrics.values()],
        prefix='Test: '
    )

    # switch to evaluate mode
    model.eval()

    print('=> encoding captions')
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, 'templates.json')) as f:
        templates = json.load(f)['imagenet']

    with open(os.path.join(cwd, 'labels.json')) as f:
        labels = json.load(f)['imagenet']

    with torch.no_grad():
        text_features = defaultdict(list)
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).cuda(args.gpu, non_blocking=True)
            class_embeddings = utils.get_model_parallel(model).encode_text_val(texts)
            embed_names = {'z_text', 'h_text', 'p_text'}.intersection(class_embeddings.keys())
            for embed_name in embed_names:
                cls_embed = class_embeddings[embed_name]
                cls_embed = cls_embed / cls_embed.norm(dim=-1, keepdim=True)
                cls_embed = cls_embed.mean(dim=0)
                cls_embed = cls_embed / cls_embed.norm(dim=-1, keepdim=True)
                text_features[embed_name].append(cls_embed)
        text_features = {k: torch.stack(v, dim=0) for k, v in text_features.items()}

        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # encode images
            image_features = utils.get_model_parallel(model).encode_image_val(images)

            # compute similarities
            similarities = utils.get_model_parallel(model).predict_zeroshot(image_features, text_features)

            # measure accuracy
            for name, similarity in similarities.items():
                acc1, acc5 = utils.accuracy(similarity, target, topk=(1, 5))
                acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
                metrics[f'acc1_{name[0]}'].update(acc1.item(), images.size(0))
                metrics[f'acc5_{name[0]}'].update(acc5.item(), images.size(0))

            if not (args.model.startswith('SLIP')) and not (args.model.startswith('CLIP')):
                # ensemble accuracy
                for w in ensemble_weights:
                    similarity = w * similarities['z_sim'] + (1-w) * similarities['h_sim']
                    acc1, acc5 = utils.accuracy(similarity, target, topk=(1, 5))
                    acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
                    metrics[f'acc1_zh_{w}'].update(acc1.item(), images.size(0))
                    metrics[f'acc5_zh_{w}'].update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    progress.synchronize()
    print(f'0-shot * Acc@1 {metrics["acc1_z"].avg:.3f} Acc@5 {metrics["acc5_z"].avg:.3f}')
    return {k: v.avg for k, v in metrics.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters=None, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            if meter.count == 0:
                continue
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SLIP training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
