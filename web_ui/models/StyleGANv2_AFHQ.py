import os
import torch
import torchvision
from PIL import Image
import time
from munch import Munch
from torchvision import transforms

from model.ZSSGAN import SG2Generator
from inference_e4e import run_alignment, get_average_image, run_loop, load_e4e

net = Munch()
transform_list = []
args = Munch()


def init():
    args = Munch({
        "ckpt_dict": {"src": "../pre_stylegan/afhqdog.pt",
                      "cartoon": "../adapted_generator/dog/cartoon.pt",
                      "cubism": "../adapted_generator/dog/cubism.pt",
                      "pointillism": "../adapted_generator/dog/pointillism.pt"},
        "img_size": 512,
        "channel_multiplier": 2,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })

    # Create a dictionary to store generators
    generators = {}
    for model_name in args.ckpt_dict.keys():
        generators[model_name] = SG2Generator(args.ckpt_dict[model_name], img_size=args.img_size,
                                              channel_multiplier=args.channel_multiplier,
                                              device=args.device)
        generators[model_name].freeze_layers()
        generators[model_name].eval()

    global net
    net = Munch(generators=generators)  # Store generators in net
    print(f"StyleGAN_AFHQ models initialized")
    return net


def controller(request):
    mode = request.form['mode']  # mode == "latent" or "reference"
    seed = request.form['seed']
    tar_style = request.form['tar_style']
    res = styleganv2(mode, seed, tar_style)
    return res


def styleganv2(mode, seed, tar_style, ref_img_path=None):
    # 为了返回res，只用当res.success为true时，才会在前端更新图片，data存储输出图片的路径
    res = Munch({
        "success": False,
        "message": "default message",
        "data": None
    })

    generator_src = net.generators["src"]
    generator_tar = net.generators[tar_style]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    z = torch.randn(1, 512, device="cuda" if torch.cuda.is_available() else "cpu")
    w = generator_tar.style([z])

    target_img = generator_tar(w, input_is_latent=True, truncation=0.7, randomize_noise=False)[0]
    source_img = generator_src(w, input_is_latent=True, truncation=0.7, randomize_noise=False)[0]
    cache_path = "cache"
    os.makedirs(cache_path, exist_ok=True)

    source_img_path = f"cache/source_img_{int(time.time())}.png"
    target_img_path = f"cache/target_img_{int(time.time())}.png"
    torchvision.utils.save_image(source_img, source_img_path, normalize=True)
    torchvision.utils.save_image(target_img, target_img_path, normalize=True)

    res.success = True
    res.data = [source_img_path, target_img_path]

    return res
