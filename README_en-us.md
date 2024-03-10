# Undergraduate Dissertation: A Study and Application of Prompt Learning-based Vision-Language Models in Image Generation

 [English](./README_en-us.md) | [中文](./README.md)

The implementation of this dissertation is based on [IPL](https://arxiv.org/pdf/2304.03119.pdf), this repository is mostly forked from [IPL-Zero-Shot-Generative-Model-Adaptation](https://github.com/Picsart-AI-Research/IPL-Zero-Shot-Generative-Model-Adaptation) and several changes have been made.

## Dependencies

### Create anaconda environment

```powershell
conda create -n ipl python=3.8
conda activate ipl
```

### Install dependencies

Make sure the version of NVIDIA driver, CUDA and PyTorch are compatible to each other.

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install ftfy regex tqdm ninja
pip install git+https://github.com/openai/CLIP.git
```

### Download pretrained generator

Pretrained **source** StyleGAN2 generators can be downloaded (via [Google Drive](https://drive.google.com/drive/folders/1FW8XfDbTg9MLEodEeIl6zJEaCVyZ053L?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/dbd0955d9a9547dc99f2/)) and place it/them in `./pre_stylegan` folder.

## Abstract

## Details

## Quantitative Evaluation Metrics

