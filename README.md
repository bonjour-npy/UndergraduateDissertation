# 本科毕业论文：基于 Prompt Learning 的视觉-语言大模型在图像生成中的应用与研究

[中文](./README.md) | [English](./README_en-us.md)

本篇论文主要基于 [IPL](https://arxiv.org/pdf/2304.03119.pdf)的思想实现。本仓库大部分从 [IPL-Zero-Shot-Generative-Model-Adaptation](https://github.com/Picsart-AI-Research/IPL-Zero-Shot-Generative-Model-Adaptation) fork 而来并做出了一定修改。

## 依赖

### 创建 Anaconda 虚拟环境

```powershell
conda create -n ipl python=3.8
conda activate ipl
```

### 安装依赖

请确保 NVIDIA 驱动、CUDA 以及 PyTorch 之间版本互相匹配。

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install ftfy regex tqdm ninja
pip install git+https://github.com/openai/CLIP.git
```

### 下载预训练生成器

预训练的源域生成器可以通过 [Google Drive ](https://drive.google.com/drive/folders/1FW8XfDbTg9MLEodEeIl6zJEaCVyZ053L?usp=sharing)或者 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/dbd0955d9a9547dc99f2/) 下载，并将其置于 `./pre_stylegan` 文件夹中。

## 概述

## 技术细节

### prompts 的初始化

`ctx_init `参数用于初始化 prompts，官方提供的演示 context 是`a photo of a`。

```python
source_prompts = [prompt_prefix + " " + args.source_class]
    target_prompts = [prompt_prefix + " " + args.target_class]
```

源域的初始提示词 `source_prompts` 是 ctx_init 与源域标签的组合。若源域标签为 `photo`，则源域的初始提示词是 `a photo of a photo`。目标域的初始提示词同理。

### prompts 的 tokenize 与 embedding

源域以及目标域的初始提示词接下来会进行 tokenize：

```python
source_tokenized_prompts = torch.cat([clip.tokenize(p) for p in source_prompts]).to(device)
# (1, 77) 'sot a photo of a photo eot' 在经过tokenize后为tensor [[49406, 320, 1125, 539, 320, 1125, 49407, etc]]
# 77是CLIP在tokenize方法中缺省的context_length，超过context_length将被truncate，不足的将用0补齐
target_tokenized_prompts = torch.cat([clip.tokenize(p) for p in target_prompts]).to(device)
# (1, 77) 'sot a photo of a disney' 在经过tokenize后为tensor [[49406, 320, 1125, 539, 320, 4696, 49407, etc]]
# 77是CLIP在tokenize方法中缺省的context_length，超过context_length将被truncate，不足的将用0补齐
```

tokenize 是 CLIP 对送入的 prompt 字符串进行标记化处理，在头部和尾部添加 startoftext 以及 endoftext 标记，最终为两个首尾标记和全部单词生成 int 标记。其中 CLIP 模型缺省的 `context_length` 是77，若 prompt 大于 77 会进行截断（truncate），若小于 77 会进行补零，因此 `source_tokenized_prompts` 与 `target_tokenized_prompts` 的形状均为 (1, 77)。

在提示词标记化之后，将进行嵌入表示 embedding：

```python
source_embedding = clip_model.token_embedding(source_tokenized_prompts).type(clip_model.dtype)
# (1, 77, 512) 其中512是CLIP中的n_dim，token_embedding层的词嵌入的维度
target_embedding = clip_model.token_embedding(target_tokenized_prompts).type(clip_model.dtype)
# (1, 77, 512) 其中512是CLIP中的n_dim，token_embedding层的词嵌入的维度
```

### compute_text_features 的实现细节

在 Mapper 生成 prompts 后进行 prompts 的特征提取时，需要传入 tokenize 之后的人工初始化 prompt（‘a photo of a photo.’或‘a photo of a disney.’），用于选择 eot 符号对应的维度来进行特征投影（**因为 eot 作为整个句子的结尾，被认为该维度包含更多的信息**。具体做法：由于在 tokenize 之后，eot 符号对应的维度的值最大，因此可使用 argmax 来定位），以保证最后得到的特征形状与图像特征提取的输出形状相同，使得后续可以进行对比学习的损失计算。

### 训练 stage 1

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
   - Z 空间提供了初始随机噪声，而 W 空间则通过特征解耦提供更多控制图像风格的灵活性。通过对 Z 和 W 之间的映射以及 W 在生成器中的应用，StyleGan 实现了高度可控且具有良好生成效果的图像合成。

#### 损失函数

在代码中，stage 1 的损失函数是 `global_clip_loss`，该损失由三部分组成：

1. 对比学习损失：Mapper 生成的源域 prompts 的特征**（注意，这里的 prompts 特征是与人工初始化的 prompt （带源域标签）的特征做过 element-wise 相加后的特征）**与源域图像特征的余弦相似度组成的对比学习损失；
2. 目标域正则化损失：Mapper 生成的目标域 prompts 的特征与目标域文本标签特征的余弦相似度，这里生成的目标域 prompts 特征同样也是与人工初始化的 prompts 做过加法的。注意该损失有权重 `lambda_l`。

### 训练 stage 2

#### 确定目标域生成域需要更新的层

在训练的第二阶段进行前向传播时，需要先对目标域生成器（generator_trainable）的所有层进行 unfreeze，然后对更新哪些层做出选择，承担选择任务的功能函数：model.ZSSGAN.ZSSGAN.determine_opt_layers，最后 freeze 所有层后再 unfreeze 选择的网络层。

```python
if self.training and self.auto_layer_iters > 0:
    self.generator_trainable.unfreeze_layers()  # unfreeze
    train_layers = self.determine_opt_layers()  # layer to train

    if not isinstance(train_layers, list):
        train_layers = [train_layers]

    self.generator_trainable.freeze_layers()
    self.generator_trainable.unfreeze_layers(train_layers)  # unfreeze
```

具体选择带更新网络层的策略：

将 W 空间的隐向量送入目标域生成器（SG2Generator）中，并进行反向传播，此时可以通过反向传播后 W 空间隐向量不同维度的更新幅度来衡量不同网络层的影响力，因此选出更新幅度最大的维度就可以确定在 Model Adaption 中需要更新的网络层。

**之所以 W 空间编码在 n_latent 维度上的序号就代表着对应的网络层数的序号，是因为 StyleGAN 生成器的结构决定了这一点：StyleGAN 生成器中，W 空间编码的不同维度会被送入生成器网络的不同层，控制这些层的特征映射 (feature mapping)。具体来说，W 空间编码的每个维度会被重复 n_latent 次，作为该层的风格向量 (style vector)，通过 AdaIN (Adaptive Instance Normalization) 层控制该层的特征映射。因此，W 空间编码的第 i 个维度会影响生成器网络中第 i 层的特征映射。当某个维度的 W 值被更新的程度较大时，就意味着该维度对应的层在生成目标图像时起到了重要作用，需要被优化。**

#### 损失函数



## 定量分析指标

1. Inception Score（IS）

   评估图像的质量和多样性

2. Style Fréchet Inception Distance（SFID）

   评估目标域的风格

3. Structural Consistency Score（SCS）

   评估图像的结构保存能力

4. Identity Similarity（ID）

   评估图像的特征保存能力
