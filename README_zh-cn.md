# 本科毕业论文：基于Prompt Learning的视觉-语言大模型在图像生成中的应用与研究

[English](./README.md) | [简体中文](./README_zh-cn.md)

本篇论文主要基于[IPL](https://arxiv.org/pdf/2304.03119.pdf)的思想实现。本仓库大部分从[IPL-Zero-Shot-Generative-Model-Adaptation](https://github.com/Picsart-AI-Research/IPL-Zero-Shot-Generative-Model-Adaptation)fork而来并做出了一定修改。

## 依赖

### 创建Anaconda虚拟环境

```powershell
conda create -n ipl python=3.8
conda activate ipl
```

### 安装依赖

请确保NVIDIA驱动、CUDA以及PyTorch之间版本互相匹配。

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install ftfy regex tqdm ninja
pip install git+https://github.com/openai/CLIP.git
```

### 下载预训练生成器

预训练的源域生成器可以通过[Google Drive](https://drive.google.com/drive/folders/1FW8XfDbTg9MLEodEeIl6zJEaCVyZ053L?usp=sharing)或者[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/dbd0955d9a9547dc99f2/)下载，并将其置于`./pre_stylegan`文件夹中。

## 概述

## 主要方法

## 评价指标

