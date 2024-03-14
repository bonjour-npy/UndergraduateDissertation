import os
import torch
import torchvision
from PIL import Image
from torchvision.utils import save_image as save_generated_images
import warnings

from model.ZSSGAN import SG2Generator
from model.encoder.w_encoder import WEncoder

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


def inference_desired_photo():
    dataset_size = 1024  # args.size

    print(f"Checking directory.\n")
    output_dir = "./inference_output"
    image_src_path = os.path.join(output_dir, "source_image.jpg")
    image_tar_path = os.path.join(output_dir, "target_image.jpg")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading pre-trained target generator.\n")
    adapted_gen_ckpt = "./pre_stylegan/stylegan2-ffhq-config-f.pt"
    generator_ema = SG2Generator(adapted_gen_ckpt, img_size=dataset_size, channel_multiplier=2, device=device)
    generator_ema.freeze_layers()
    generator_ema.eval()

    print(f"Loading pre-trained source generator.\n")
    frozen_gen_ckpt = "./pre_stylegan/stylegan2-ffhq-config-f.pt"
    generator_frozen = SG2Generator(frozen_gen_ckpt, img_size=dataset_size, channel_multiplier=2, device=device)
    generator_frozen.freeze_layers()
    generator_frozen.eval()

    """
    参考hyperstyle的 models.hyperstyle.HyperStyle.__get_pretrained_w_encoder
                    models.encoders.psp.pSp.load_weights
    """
    print(f"Loading pre-trained w-encoder.\n")
    w_encoder_ckpt = "./pre_stylegan/faces_w_encoder.pt"
    ckpt = torch.load(w_encoder_ckpt, map_location=device)
    w_encoder = WEncoder(num_layers=50, mode='ir_se')
    w_encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
    w_encoder.eval()
    w_encoder.cuda()

    image = Image.open("./desired_image/photo_0037.png")
    image = torchvision.transforms.ToTensor()(image).unsqueeze(0).cuda()
    image = image / 255.

    with torch.no_grad():
        w_codes = w_encoder(image)
        target_image = generator_ema(w_codes, input_is_latent=True, truncation=0.7, randomize_noise=False)[0]
        source_image = generator_frozen(w_codes, input_is_latent=True, truncation=0.7, randomize_noise=False)[0]
        save_generated_images(source_image, image_src_path, normalize=True)
        save_generated_images(target_image, image_tar_path, normalize=True)


if __name__ == "__main__":
    inference_desired_photo()
