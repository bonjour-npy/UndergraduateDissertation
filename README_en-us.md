# Undergraduate Thesis: Application and Research of Visual-Language Models in Image Generation Based on Prompt Learning

[中文](./README.md) | [English](./README_en-us.md)

This thesis is mainly based on the idea of [IPL](https://arxiv.org/pdf/2304.03119.pdf) with certain modifications and additions.

Thanks to the authors of IPL for their open-source work. The BibTeX citation is as follows:

```latex
@InProceedings{Guo_2023_CVPR,
    author    = {Guo, Jiayi and Wang, Chaofei and Wu, You and Zhang, Eric and Wang, Kai and Xu, Xingqian and Song, Shiji and Shi, Humphrey and Huang, Gao},
    title     = {Zero-Shot Generative Model Adaptation via Image-Specific Prompt Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {11494-11503}
}
```

## Dependencies

### Create Anaconda Virtual Environment

```powershell
conda create -n ipl python=3.8
conda activate ipl
```

### Install Dependencies

Please ensure the versions of NVIDIA drivers, CUDA, and PyTorch are compatible with each other.

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install ftfy regex tqdm ninja
pip install git+https://github.com/openai/CLIP.git
```

### Download Pre-trained Generator

The pre-trained source domain generator can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1FW8XfDbTg9MLEodEeIl6zJEaCVyZ053L?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/dbd0955d9a9547dc99f2/), and placed in the `./pre_stylegan` folder.

## Usage

### Web UI System

Run `web_ui/app.py`, with the default port set to `3000`.

### Image Inversion Function

The image inversion function allows users to invert selected images to W space. Two inversion interfaces are provided, using `e4e` and `pSp` encoders respectively.

- e4e Encoder: Run `./inference_e4e.py`
- pSp Encoder: Run `./inference_psp.py`

### Training

Run `./train.py` or `./train_improved.py` with specific network settings to train your own zero-shot generation model.

With batch_size set to 32 and iteration to 300 in the first stage, and batch_size set to 2 and iteration to 300 in the second stage, training on the following mobile GPU takes approximately 12 hours.

    NVIDIA GeForce RTX 3060 Laptop GPU
    
    Driver Version: 32.0.15.5585
    Driver Date: 2024/5/13
    DirectX Version: 12 (FL 12.1)
    Physical Location: PCI Bus 1, Device 0, Function 0
    
    Dedicated GPU Memory: 6.0 GB
    Shared GPU Memory: 15.9 GB

A possible training script parameter is as follows:

```python
--frozen_gen_ckpt
./pre_stylegan/stylegan2-ffhq-config-f.pt
--source_model_type
"ffhq"
--output_dir
"./outputs"
--n_sample
5
--sample_truncation
0.7
--ckpt
None
--lr
0.002
--iter
300
--batch
32
--test_batch
32
--size
256
--path
"./data"
--project_name
"StyleGAN2"
```
