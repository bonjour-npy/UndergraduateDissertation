# 本科毕业论文：基于Prompt Learning的视觉-语言大模型在图像生成中的应用与研究

[中文](./README.md) | [English](./README_en-us.md)

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

## 技术细节

### prompts的初始化

`ctx_init`参数用于初始化prompts，官方提供的演示context是`a photo of a`。

```python
source_prompts = [prompt_prefix + " " + args.source_class]
    target_prompts = [prompt_prefix + " " + args.target_class]
```

源域的初始提示词`source_prompts`是ctx_init与源域标签的组合。若源域标签为`photo`，则源域的初始提示词是`a photo of a photo`。目标域的初始提示词同理。

### prompts的tokenize与embedding

源域以及目标域的初始提示词接下来会进行tokenize：

```python
source_tokenized_prompts = torch.cat([clip.tokenize(p) for p in source_prompts]).to(device)
# (1, 77) 'sot a photo of a photo eot' 在经过tokenize后为tensor [[49406, 320, 1125, 539, 320, 1125, 49407, etc]]
# 77是CLIP在tokenize方法中缺省的context_length，超过context_length将被truncate，不足的将用0补齐
target_tokenized_prompts = torch.cat([clip.tokenize(p) for p in target_prompts]).to(device)
# (1, 77) 'sot a photo of a disney' 在经过tokenize后为tensor [[49406, 320, 1125, 539, 320, 4696, 49407, etc]]
# 77是CLIP在tokenize方法中缺省的context_length，超过context_length将被truncate，不足的将用0补齐
```

tokenize是CLIP对送入的prompt字符串进行标记化处理，在头部和尾部添加startoftext以及endoftext标记，最终为两个首尾标记和全部单词生成int标记。其中CLIP模型缺省的`context_length`是77，若prompt大于77会进行截断（truncate），若小于77会进行补零，因此`source_tokenized_prompts`与`target_tokenized_prompts`的形状均为(1, 77)。

在提示词标记化之后，将进行嵌入表示embedding：

```python
source_embedding = clip_model.token_embedding(source_tokenized_prompts).type(clip_model.dtype)
# (1, 77, 512) 其中512是CLIP中的n_dim，token_embedding层的词嵌入的维度
target_embedding = clip_model.token_embedding(target_tokenized_prompts).type(clip_model.dtype)
# (1, 77, 512) 其中512是CLIP中的n_dim，token_embedding层的词嵌入的维度
```

### 训练stage 1

#### Z空间与W空间

```python
# Z空间到W空间的变换
sample_z = mixing_noise(args.batch_mapper, 512, args.mixing, device)
# (batch_size, 512)
sample_w = net.generator_frozen.style(sample_z)
# (batch_size, 512)
```

Z 空间和 W 空间是 StyleGAN 模型中两种不同的隐变量空间，分别用于控制生成图像的随机特征和样式信息。W 空间通过对 Z 空间的映射得到。

1. **Z 空间（Latent Space Z）**：

   - Z 空间是随机噪声空间，通常由随机噪声向量组成，表示了图像的随机特征。
   - 在 StyleGAN 中，Z 空间的维度通常为 512 维。这意味着一个 Z 向量由 512 个数字组成，每个数字表示了图像的一个随机特征的强度或者方向。

2. **W 空间（Style Space W）**：

   - W 空间经过特征解耦的隐空间，与 Z 空间相比更加解耦合。

   - 在 StyleGAN 中，W 空间的维度也通常为 512 维，是通过mapping network进行映射得到的，mapping network由PixelNorm层与EqualLinear层构成。以下代码节选自`sg2_model.py`

     ```python
     '''mapping network'''
     layers = [PixelNorm()]
     
     for i in range(n_mlp):
         layers.append(
             EqualLinear(
                 style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
             )
         )
     
     self.style = nn.Sequential(*layers)
     ```

3. **Z 空间与 W 空间的关系**：

   - 在 StyleGAN 中，通常会先将一个 Z 向量映射到 W 空间，然后再将 W 向量输入到生成器网络中生成图像。
   - Z空间提供了初始随机噪声，而W空间则通过特征解耦提供更多控制图像风格的灵活性。通过对Z和W之间的映射以及W在生成器中的应用，StyleGan实现了高度可控且具有良好生成效果的图像合成。

### 训练stage 2

## 评价指标

