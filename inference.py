import os
import torch
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from argparse import Namespace
from torchvision.utils import save_image as save_generated_images
import warnings
import dlib

from model.ZSSGAN import SG2Generator
from utils.align_faces_parallel import align_face  # face alignment with FFHQ method (https://github.com/NVlabs/ffhq-dataset)
from model.encoder.e4e import e4e
from model.encoder.psp import pSp

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_alignment(image_path):
    shape_predictor_path = "./pretrained_models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(shape_predictor_path):
        print('Downloading files for aligning face image...')
        os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        print('Done.')
    print(f"Loading shape predictor from {shape_predictor_path}\n")
    predictor = dlib.shape_predictor(shape_predictor_path)
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def load_psp(checkpoint_path, device=device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    opts = ckpt['opts']
    opts['checkpoint_path'] = checkpoint_path
    opts['load_w_encoder'] = True

    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net.to(device)

    return net


def load_e4e(checkpoint_path, device=device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    opts = ckpt['opts']
    opts['checkpoint_path'] = checkpoint_path
    opts['load_w_encoder'] = True

    opts = Namespace(**opts)

    net = e4e(opts)
    net.eval()
    net.to(device)

    return net


def get_average_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    return avg_image


def run_loop(inputs, net, avg_image, n_iters=5):
    y_hat, latent, codes = None, None, None
    results_batch = {idx: [] for idx in range(inputs.shape[0])}
    results_latent = {idx: [] for idx in range(inputs.shape[0])}
    for iter in range(n_iters + 1):
        if iter == 0:
            avg_image_for_batch = avg_image.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
            x_input = torch.cat([inputs, avg_image_for_batch], dim=1)
        else:
            x_input = torch.cat([inputs, y_hat], dim=1)

        if not (iter == n_iters):  # 正常循环
            y_hat, latent = net.forward(x_input,
                                        latent=latent,
                                        randomize_noise=False,
                                        return_latents=True,
                                        resize=True)

            # store intermediate outputs
            for idx in range(inputs.shape[0]):
                results_batch[idx].append(y_hat[idx])
                results_latent[idx].append(latent[idx].cpu().numpy())

            # resize input to 256 before feeding into next iteration
            y_hat = net.face_pool(y_hat)
        else:  # 直接返回codes
            codes = net.forward(x_input,
                                latent=latent,
                                randomize_noise=False,
                                return_latents=True,
                                resize=True,
                                return_codes=True)

    return codes


def inference():
    dataset_size = 1024  # args.size

    print(f"Checking directory.\n")
    output_dir = "./inference_output"
    os.makedirs(output_dir, exist_ok=True)

    adapted_gen_ckpt = "./adapted_generator/ffhq/disney.pt"
    print(f"Loading pre-trained target generator: {adapted_gen_ckpt}\n")
    generator_ema = SG2Generator(adapted_gen_ckpt, img_size=dataset_size, channel_multiplier=2, device=device)
    generator_ema.freeze_layers()
    generator_ema.eval()

    frozen_gen_ckpt = "./pre_stylegan/stylegan2-ffhq-config-f.pt"
    print(f"Loading pre-trained source generator: {frozen_gen_ckpt}\n")
    generator_frozen = SG2Generator(frozen_gen_ckpt, img_size=dataset_size, channel_multiplier=2, device=device)
    generator_frozen.freeze_layers()
    generator_frozen.eval()

    restyle_psp_ckpt_path = "pretrained_models/restyle_psp_ffhq_encode.pt"
    restyle_e4e_ckpt_path = "pretrained_models/restyle_e4e_ffhq_encode.pt"
    # restyle_psp = load_psp(restyle_psp_ckpt_path)
    restyle_e4e = load_e4e(restyle_e4e_ckpt_path)

    # 从本地选择图片
    # 创建一个tkinter窗口
    root = tk.Tk()
    root.withdraw()
    # 使用文件对话框选择本地文件夹
    image_path = filedialog.askopenfilename(title="select image file")

    print(f"\nLoading image from {image_path}\n")
    image = run_alignment(image_path)
    image.resize((256, 256))
    transform_inference = transforms.Compose([transforms.Resize((256, 256)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    with torch.no_grad():
        transformed_image = transform_inference(image).cuda()
        avg_image = get_average_image(restyle_e4e)
        codes = run_loop(transformed_image.unsqueeze(0), restyle_e4e, avg_image)
        print(f"\nGenerating images to {output_dir}\n")
        target_image = generator_ema([codes], input_is_latent=True, randomize_noise=False)[0]
        source_image = generator_frozen([codes], input_is_latent=True, randomize_noise=False)[0]
        image_src_path = os.path.join(output_dir, "source_image.jpg")
        image_tar_path = os.path.join(output_dir, "target_image.jpg")
        save_generated_images(source_image, image_src_path, normalize=True)
        save_generated_images(target_image, image_tar_path, normalize=True)

    # 调用系统应用显示图片
    # 打开图片文件
    img1 = Image.open(image_path)
    img2 = Image.open(image_src_path)
    img3 = Image.open(image_tar_path)
    # 创建一个新的图片对象
    new_img = Image.new('RGB', (img1.width + img2.width + img3.width, max(img1.height, img2.height, img3.height)))
    # 将图片粘贴到新的图片对象中
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    new_img.paste(img3, (img1.width + img2.width, 0))
    # 显示图片
    new_img.save(os.path.join(output_dir, f"{image_path.split('/')[-1].split('.')[0]}_comparison.jpg"))
    new_img.show()


if __name__ == "__main__":
    inference()
