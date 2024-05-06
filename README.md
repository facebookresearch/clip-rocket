# [Improved baselines for vision-language pre-training](https://arxiv.org/abs/2305.08675)

## In this repo:
 - [**CLIP** :rocket:](https://arxiv.org/abs/2305.08675), an improved recipe for training [CLIP](https://arxiv.org/abs/2103.00020)
 - **SiamLIP**, **BYOLIP**, **BarLIP**, and **SwALIP**, non-contrastive VLP baselines extending CLIP and inspired from [SimSiam](https://arxiv.org/abs/2011.10566), [BYOL](https://arxiv.org/abs/2006.07733), [Barlow Twins](https://arxiv.org/abs/2103.03230), and [SwAV](https://arxiv.org/abs/2006.09882)
 - [**repro_exps.txt**](repro_exps.txt): scripts to reproduce the experiments of our paper: [Improved baselines for vision-language pre-training](arxiv-link)

## Setup

### Requirements
The code has been tested with CUDA 11.3/CuDNN 8.3.2, PyTorch 1.12.1 and timm 0.6.11.
For a minimal environment use `conda env create -f clip_rocket_env.yaml` and optionally install wandb via pip.

conda:
- python=3.9
- pytorch=1.12.1=py3.9_cuda11.3_cudnn8.3.2_0 -c pytorch
- torchvision=0.13.1=py39_cu113 -c pytorch

pip:
- timm==0.6.11
- xformers==0.0.14.dev315+git.e23b369=py39_cu11.3_pyt1.12.1
- flash-attn==0.1
- textaugment==1.3.4
- nltk==3.7
- [optional] wandb

### Datasets

#### YFCC15M
Download the [YFCC100M dataset](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/).
Our dataloader expects the following dataset directory structure with 100 folders containing 1000 zip archives of 1000 images each.
The concatenation of the folder, archive, and file names is the index of the image (i.e. image 12345678 is stored as `678.jpg` within `12/345.zip`):

```
/path/to/yfcc100m/
├── images/
│   ├── 00/
│   │   └── 000.zip
│   │   │   ├── 000.jpg
│   │   │   │   ...
│   │   │   └── 999.jpg
│   │   ...
│   │   └── 999.zip
│   ...
│   └── 99/
...
```

Prepare the YFCC15M subset metadata pickle:
1. Download and compile a list of downloaded images to `flickr_unique_ids.npy` ([ours](https://dl.fbaipublicfiles.com/deepcluster/flickr_unique_ids.npy))
2. Download OpenAI's list of captioned YFCC100M images according to instructions [here](https://github.com/openai/CLIP/blob/8cad3a736a833bc4c9b4dd34ef12b52ec0e68856/data/yfcc100m.md)
3. Run `python make_dataset.py` to create the `yfcc15m.pkl` metadata pickle

When pre-training with YFCC15M, set `--dataset yfcc15m --root /path/to/yfcc100m --metadata /path/to/yfcc15m.pkl`.

#### Conceptual Captions
[CC3M](https://ai.google.com/research/ConceptualCaptions/download) and [CC12M](https://github.com/google-research-datasets/conceptual-12m) are published as tsv files listing original image urls and processed captions.
Download images and collect the captions of all available images (many will be missing due to broken links) into `cc3m.npy` and `cc12m.npy`.

For CC3M our dataloader expects `cc3m.npy` to contain a NumPy array of dicts in the following format:

```
{
  'image_id': 1510438788,  # local file path relative to root
  'captions': ['large field with pink tulips on a clear sunny summer day with a blue sky']
}
```

For CC12M our dataloader expects `cc12m.npy` to contain a NumPy array of dicts in the following format:

```
{
  'image_name': '0.jpg',  # local file path relative to root
  'image_id': 0,
  'captions': ['Metal Design Within Reach Ivory Slipper Chairs - a Pair For Sale - Image 7 of 10']
}
```

When pre-training on CC3M set `--dataset cc3m --root /path/to/cc3m --metadata /path/to/cc3m.npy`, and whe pre-training on CC12M set `--dataset cc12m --root /path/to/cc12m --metadata /path/to/cc12m.npy`.

#### Downstream Datasets
Zero-shot (in [main.py](main.py) and [eval_zeroshot.py](eval_zeroshot.py)) evaluations read dataset paths from [dataset_catalog.json](dataset_catalog.json).
Zero-shot evaluations read CLIP's class labels and caption templates from [labels.json](labels.json) and [templates.json](templates.json).
If just pre-training models on YFCC15M, only the ImageNet path is required for model validation between training epochs.
See Section 3 below on zero-shot transfer evaluation for dataset preparation details.

## Pre-training
We use the following pre-training recipes for CLIP :rocket: and the other improved baselines. Note that in our code the model class needed for the improved recipe is marked as `CL2L`.
See [main.py](main.py) for the full list of default arguments.
The different models can be selected by passing different strings to the `--model` argument such as `CL2L_R50_CL2L` or `CL2L_R50_BARLIP` or `CL2L_VITB16_CL2L`. As can be noted, the string is composed of three substrings: `<base model>_<vision bbone>_<model name>`:
- `<base model>` can be either `CLIP` or `CL2L`, and defines the base class of the model, where the latter is an extension of the first that allows for mutliple augmentations and architectural improvements like multiple projectors. For this reason, `CL2L` can emulate baseline `CLIP` by setting `--num-augs 0`.
- `<vision bbone>` defines the vision encoder and can assume the name of whichever class implemented in [models.py](models.py)
- `<model name>` defines the actual model we are going to train. Supported choices are defined in [`get_model()`](models.py#587)       

In our workflow we use [submitit](https://github.com/facebookincubator/submitit), which interfaces nicely with Slurm.
For local training with the [torchrun](https://pytorch.org/docs/stable/elastic/run.html) utility (supersedes `torch.distributed.launch`), replace `python run_with_submitit.py` with `torchrun --nproc_per_node=8 main.py`. 
Local multi-node training with `torchrun` should also be possible. `run.sh` provides a convenient wrapper to robustly run experiments based on the principle one commit --> one experiment.

We train most of our models on 4x 8-gpu nodes, but training with fewer gpus is possible by setting the `--update-freq` argument above 1 to enable gradient accumulation or using `--checkpoint-grad` which reduces space complexity.
Note that gradient accumulation will increase the variance of minibatch statistics and alter the training dynamics of batchnorm.

### CLIP:rocket: ViT-Base with 4-nodes each having 8 NVIDIA V100-32GB (batch size 4096) 
```
bash run.sh run_with_submitit.py \
  --model CL2L_R50_CL2L \
  --dataset yfcc15m \
  --name CL2L_R50_CLIP-YFCC \
  --separate-proj \
  --text-augment \
  --clean-before-augment \
  --loss-avg-or-sum sum \
  --label-smoothing 0.1 \
  --epochs 32 \
  --nodes 4 \
  --batch-size 128 \
```

## Evaluation: Zero-shot Transfer
First, prepare additional downstream classification datasets:
- MNIST, CIFAR-10/100, STL-10: Automatic download via [torchvision datasets](https://pytorch.org/vision/stable/datasets.html)
- HatefulMemes: Manual download from [official website](https://hatefulmemeschallenge.com/#download) and sort images according to `train.jsonl`/`dev.jsonl` into train/dev folder
- Rendered SST2, Country211: Manual download from [CLIP repo](https://github.com/openai/CLIP/tree/main/data)
- Other datasets: Use scripts from [VISSL](https://github.com/facebookresearch/vissl/tree/main/extra_scripts/datasets)

Then set all dataset paths in [dataset_catalog.json](dataset_catalog.json).

Evaluate zero-shot transfer to various classification benchmarks with [eval_zeroshot.py](eval_zeroshot.py), which reads labels and templates from [labels.json](labels.json)/[templates.json](templates.json) and dataset paths from [dataset_catalog.json](dataset_catalog.json). Inference is performed with a single gpu. By default, the script iterates through all datasets in [dataset_catalog.json](dataset_catalog.json) and evaluates zero-shot in order. Evaluation can be limited to a subset of datasets by replacing `for d in datasets:` with `for d in ['imagenet']:` on line 78.

```
python eval_zeroshot.py --output-dir /path/to/experiment --model-name clip-rocket
```

## Acknowledgements

This repo is mostly based on the [SLIP repo](https://github.com/facebookresearch/SLIP). Also, we adapted some code from the [CLIP repo](https://github.com/openai/CLIP) and the [timm repo](https://github.com/rwightman/timm) . We commend the authors of these repos for the great contribution to the community.

## Contributing to clip-rocket

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

### License

The majority of `clip-rocket` is licensed under CC-BY-NC, however portions of the project are available under separate license terms: https://github.com/facebookresearch/SLIP and https://github.com/openai/CLIP are licensed under the MIT license and https://github.com/rwightman/timm is licensed under the Apache-2.0 license.. See [LICENSE](LICENSE) for details.

### Citation
```
@article{
fini2023improved,
  title={Improved baselines for vision-language pre-training},
  author={Enrico Fini and Pietro Astolfi and Adriana Romero-Soriano and Jakob Verbeek and Michal Drozdzal},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2023},
  url={https://openreview.net/forum?id=a7nvXxNmdV},
  note={Featured Certification}
}
```
