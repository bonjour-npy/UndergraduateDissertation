import os
import torch
import torchvision
from PIL import Image
from munch import Munch
from torchvision import transforms

from model.ZSSGAN import SG2Generator

net = Munch()
transform_list = []
args = Munch()


def init():
    args = Munch({
        "ckpt_tar": "../output/disney_improved/checkpoint/generator/000300.pt",
        "ckpt_src": "../pre_stylegan/stylegan2-ffhq-config-f.pt",
        "img_size": 1024,
        "channel_multiplier": 2,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })
    generator_tar = SG2Generator(args.ckpt_tar, img_size=args.img_size, channel_multiplier=args.channel_multiplier,
                                 device=args.device)
    generator_tar.freeze_layers()
    generator_tar.eval()

    generator_src = SG2Generator(args.ckpt_src, img_size=args.img_size, channel_multiplier=args.channel_multiplier,
                                 device=args.device)
    generator_src.freeze_layers()
    generator_src.eval()

    global net
    net = Munch(generator_src=generator_src,
                generator_tar=generator_tar)

    print(f"StyleGAN model initialized")

    return net


def controller(request):
    mode = request.form['mode']  # mode == "latent" or "reference"
    seed = request.form['seed']
    if mode == 'reference':  # 使用参考图像生成
        ref_img = Image.open(request.files['ref_img'])
        res = styleganv2(mode, seed, ref_img)
    else:  # mode="latent" 使用随机种子生成
        res = styleganv2(mode, seed)

    return res


def styleganv2(mode, seed, ref_img=None):
    # 为了返回res，只用当res.success为true时，才会在前端更新图片，data存储输出图片的路径
    res = Munch({
        "success": False,
        "message": "default message",
        "data": None
    })

    if mode not in ['latent', 'reference']:
        res.message = f"no such mode: {mode}"

    if mode == 'latent':
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # np.random.seed(seed)
        # np.random.seed(seed)

        z = torch.randn(1, 512, device="cuda" if torch.cuda.is_available() else "cpu")
        w = net.generator_tar.style([z])

        target_img = net.generator_tar(w, input_is_latent=True, truncation=0.7, randomize_noise=False)[0]
        source_img = net.generator_src(w, input_is_latent=True, truncation=0.7, randomize_noise=False)[0]

        cache_path = "cache"
        os.makedirs(cache_path, exist_ok=True)

        source_img_path = f"cache/source_img_with_seed_{seed}.png"
        target_img_path = f"cache/target_img_with_seed_{seed}.png"
        torchvision.utils.save_image(source_img, source_img_path, normalize=True)
        torchvision.utils.save_image(target_img, target_img_path, normalize=True)

        res.success = True
        res.data = [source_img_path, target_img_path]

        return res
