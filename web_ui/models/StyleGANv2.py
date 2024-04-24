import os
import torch
import torchvision
from PIL import Image
import time
from munch import Munch
from torchvision import transforms

from model.ZSSGAN import SG2Generator
from inference_e4e import run_alignment, get_average_image, run_loop, load_e4e
from model.encoder.e4e import e4e

net = Munch()
transform_list = []
args = Munch()


def init():
    args = Munch({
        "ckpt_tar": "../output/disney_improved/checkpoint/generator/000300.pt",
        "ckpt_src": "../pre_stylegan/stylegan2-ffhq-config-f.pt",
        "restyle_e4e_ckpt": "../pretrained_models/restyle_e4e_ffhq_encode.pt",
        # "shape_predictor_GTX_ckpt": "../pretrained_models/shape_predictor_68_face_landmarks_GTX.dat",
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

    restyle_e4e = load_e4e(args.restyle_e4e_ckpt)

    global net
    net = Munch(generator_src=generator_src,
                generator_tar=generator_tar,
                restyle_e4e=restyle_e4e)

    print(f"StyleGAN model initialized")

    return net


def controller(request):
    mode = request.form['mode']  # mode == "latent" or "reference"
    seed = request.form['seed']
    if mode == 'reference':  # 使用参考图像生成
        ref_img = Image.open(request.files['ref_img'])
        ref_img_path = "cache/ref_img.png"
        ref_img.save(ref_img_path)
        res = styleganv2(mode, seed, ref_img_path)
    else:  # mode="latent" 使用随机种子生成
        res = styleganv2(mode, seed)

    return res


def styleganv2(mode, seed, ref_img_path=None):
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

        z = torch.randn(1, 512, device="cuda" if torch.cuda.is_available() else "cpu")
        w = net.generator_tar.style([z])

        target_img = net.generator_tar(w, input_is_latent=True, truncation=0.7, randomize_noise=False)[0]
        source_img = net.generator_src(w, input_is_latent=True, truncation=0.7, randomize_noise=False)[0]
    if mode == 'reference':
        aligned_image = run_alignment(ref_img_path,
                                      shape_predictor_GTX_path="../pretrained_models/shape_predictor_68_face_landmarks_GTX.dat")
        print(1)
        # aligned_image.resize((256, 256))
        transform_inference = transforms.Compose([transforms.Resize((256, 256)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        with torch.no_grad():
            transformed_image = transform_inference(aligned_image).cuda()
            avg_image = get_average_image(net.restyle_e4e)
            codes = run_loop(transformed_image.unsqueeze(0), net.restyle_e4e, avg_image)
            target_img = net.generator_tar([codes], input_is_latent=True, randomize_noise=False)[0]
            source_img = net.generator_src([codes], input_is_latent=True, randomize_noise=False)[0]
    cache_path = "cache"
    os.makedirs(cache_path, exist_ok=True)

    source_img_path = f"cache/source_img_{int(time.time())}.png"
    target_img_path = f"cache/target_img_{int(time.time())}.png"
    torchvision.utils.save_image(source_img, source_img_path, normalize=True)
    torchvision.utils.save_image(target_img, target_img_path, normalize=True)

    res.success = True
    res.data = [source_img_path, target_img_path]

    return res
