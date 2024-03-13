import os
import torch
import numpy as np
from model.ZSSGAN import ZSSGAN, SG2Generator
from torchvision.utils import save_image as save_generated_images
from model.w_encoder import WEncoder
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference_desired_photo():
    dataset_size = 1024  # args.size

    output_dir = "./inference_output"
    image_src_path = os.path.join(output_dir, "source_image.jpg")
    image_tar_path = os.path.join(output_dir, "target_image.jpg")
    os.makedirs(output_dir, exist_ok=True)

    adapted_gen_ckpt = "./pre_stylegan/stylegan2-ffhq-config-f.pt"
    generator_ema = SG2Generator(adapted_gen_ckpt, img_size=dataset_size, channel_multiplier=2, device=device)
    generator_ema.freeze_layers()
    generator_ema.eval()

    frozen_gen_ckpt = "./pre_stylegan/stylegan2-ffhq-config-f.pt"
    generator_frozen = SG2Generator(frozen_gen_ckpt, img_size=dataset_size, channel_multiplier=2, device=device)
    generator_frozen.freeze_layers()
    generator_frozen.eval()

    image = torch.imread("./desired_images/selfie.jpg")

    with torch.no_grad():
        w_codes = 1
        target_image = generator_ema(w_codes, input_is_latent=True, truncation=0.7, randomize_noise=False)[0]
        source_image = generator_frozen(w_codes, input_is_latent=True, truncation=0.7, randomize_noise=False)[0]
        save_generated_images(source_image, image_src_path, normalize=True)
        save_generated_images(target_image, image_tar_path, normalize=True)


if __name__ == "__main__":
    inference_desired_photo()
