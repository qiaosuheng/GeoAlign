# GeoAlign fine-tuning code

This repository contains the training code for GeoAlign, a geometric alignment fine-tuning framework for DNA foundation models. The public copy is intentionally focused on the runnable training workflow: model loaders, data loaders, classification/regression training scripts, and task-adaptive pre-training helpers.


## What is included

- `models/`: backbone loading utilities and downstream heads for HyenaDNA, NTv3, and Caduceus.
- `data_loader/`: single-view and dual-view dataset pipelines, including shift and reverse-complement perturbations.
- `train_classification/`: classification training scripts.
- `train_regression/`: regression training scripts.
- `data/`: preprocessing and TAPT corpus construction scripts plus placeholder directories.

## What is not included

- Local dataset files (`*.parquet`, FASTA/TXT activity files, OOD files).
- Pretrained model weights (`model.safetensors`, `*.bin`, `*.pth`, checkpoints).
- Training outputs under `checkpoints/` and `logs/`.

## Expected local layout

Before running training, place local model files and data files under the paths referenced by the configuration blocks in each script.

```text
models/
  hyenadna-small-32k-seqlen-hf/
    config.json
    modeling_hyena.py
    tokenization_hyena.py
    model.safetensors              # user-provided
  NTv3_8M_pre/
    config.json
    modeling_ntv3_pretrained.py
    tokenization_ntv3.py
    model.safetensors              # user-provided
  caduceus-ps_seqlen-131k_d_model-256_n_layer-16/
    config.json
    modeling_caduceus.py
    tokenization_caduceus.py
    model.safetensors              # user-provided

data/
  H3K27me3/{train_split,val,test}.parquet
  H3K36me3/{train_split,val,test}.parquet
  enhancers/{train_split,val,test}.parquet
  promoter_all/{train_split,val,test}.parquet
  splice_sites_acceptors/{train_split,val,test}.parquet
  splice_sites_donors/{train_split,val,test}.parquet
  splice_sites_all/{train_split,val,test}.parquet
  Drosophila/{train,val,test}.parquet
```
We provide a script for you to quickly split the training set and validation set(data_splice_to_train_val.py)
The binary classification task is mainly provided by [nucleotide_transformer_downstream_tasks_revised](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised)
The regression task is mainly provided by [deepstarr](https://zenodo.org/records/5502060)
We also provide a script to process the data provided by DeepStarr and transform it into a format suitable for the current training code(preprocess_deepstarr.py)
## Configuration style

The scripts use an editable configuration block at the top of each file. To run a different model, task, seed list, perturbation setting, learning rate, batch size, or local checkpoint path, edit the `CONFIG`, `MODEL_PATHS`, and seed-list variables directly.

This matches the internal workflow used for the manuscript and avoids long command-line argument strings.

## Main training scripts

Standard fine-tuning / data augmentation baseline for classification:

```bash
python train_classification/train_v1_seed.py
```

GeoAlign dual-view classification fine-tuning:

```bash
python train_classification/train_v2_seed.py
```

Task-adaptive pre-training (TAPT):

```bash
python train_classification/train_tapt.py
```

Standard fine-tuning / data augmentation baseline for regression:

```bash
python train_regression/train_v1_reg_seed.py
```

GeoAlign dual-view regression fine-tuning:

```bash
python train_regression/train_v2_reg_seed.py
```

## Perturbation controls

For standard single-view training scripts, perturbations are controlled by:

```python
"aug_shift": True or False
"aug_rc": True or False
```

For GeoAlign dual-view training scripts, perturbations are controlled by:

```python
"use_shift": True or False
"use_rc": True or False
"lambda_align": 1.0
```

In the combined `shift + RC` dual-view setting, the perturbed view is sampled from shift-only, RC-only, and shift+RC transformations according to the logic in `data_loader/data_loader_dual.py`.

## Local TAPT checkpoints

If a downstream run should initialize from a TAPT checkpoint, set:

```python
"foundation_ckpt": "checkpoints/.../tapt_hyenadna_last.pth"
```

Leave it as `None` for ordinary fine-tuning from the local pretrained backbone.

## Notes for Caduceus

Caduceus depends on a separate Mamba/Caduceus-compatible environment. Due to frequent compatibility issues when configuring the Mamba environment across different devices, we recommend using prebuilt .whl files to ensure reproducibility. Before setting up the environment, please make sure that the .whl files for [causal_conv1d](https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post2/causal_conv1d-1.2.0.post2+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl) and [mamba_ssm](https://github.com/state-spaces/mamba/releases/download/v1.2.0.post1/mamba_ssm-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl) have already been downloaded to the current directory.

## Environment Configuration
for hyenadna and ntv3, Please execute the following commands in sequence on the console

```bash
conda create -n hyenadna-ntv3 python=3.10 pip -y
conda activate hyenadna-ntv3

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install \
  transformers \
  peft \
  accelerate \
  safetensors \
  numpy \
  pandas \
  scipy \
  scikit-learn \
  tqdm \
  pyarrow
```
If you are unable to install CUDA 12.6, you may try installing another CUDA version that is available in your environment. We have tested CUDA 12.1, CUDA 12.6, and CUDA 12.8, and all of them are compatible.

for caduceus, Please ensure that the causal_conv1d and mamba_ssm WHL files are downloaded to the current directory, and then execute the following commands in sequence on the console

```bash
conda create --name caduceus python=3.8
conda activate caduceus

conda install -y pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8  -c pytorch -c nvidia

pip install \
  transformers \
  peft \
  accelerate \
  safetensors \
  numpy \
  pandas \
  scipy \
  scikit-learn \
  tqdm \
  pyarrow

pip install causal_conv1d-1.2.0.post2+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```
If you encounter any issues during environment setup or reproduction, please feel free to contact us.
