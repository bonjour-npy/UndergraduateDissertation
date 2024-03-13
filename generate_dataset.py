"""
generate photo to disney evaluation dataset:

python generate_dataset.py  --src_label photo \
                            --tar_label disney \
                            --frozen_gen_ckpt ./pre_stylegan/stylegan2-ffhq-config-f.pt \
                            --adapted_gen_ckpt ./adapted_generator/ffhq/disney.pt \
                            --source_model_type "ffhq" \
                            --auto_compute \
                            --n_generate 10\
"""
import os
import torch
import numpy as np
from model.ZSSGAN import ZSSGAN, SG2Generator
from options.train_options import TrainOptions
from utils.file_utils import save_images
from torchvision.utils import save_image as save_generated_images
import warnings

warnings.filterwarnings("ignore")

dataset_sizes = {
    "ffhq": 1024,
    "dog": 512,
}


def generate(args):
    """generate IPL dataset for quantitative evaluation

    """
    if args.auto_compute:
        args.size = dataset_sizes[args.source_model_type]

    dataset_dir_src = os.path.join(args.dataset_output_dir, args.src_label + "_to_" + args.tar_label, args.src_label)
    dataset_dir_tar = os.path.join(args.dataset_output_dir, args.src_label + "_to_" + args.tar_label, args.tar_label)
    os.makedirs(dataset_dir_src, exist_ok=True)
    os.makedirs(dataset_dir_tar, exist_ok=True)

    generator_ema = SG2Generator(args.adapted_gen_ckpt, img_size=args.size, channel_multiplier=args.channel_multiplier,
                                 device=device)
    generator_ema.freeze_layers()
    generator_ema.eval()

    generator_frozen = SG2Generator(args.frozen_gen_ckpt, img_size=args.size,
                                    channel_multiplier=args.channel_multiplier, device=device)
    generator_frozen.freeze_layers()
    generator_frozen.eval()

    # torch.manual_seed(args.seed2)
    # torch.cuda.manual_seed_all(args.seed2)
    # np.random.seed(args.seed2)

    fixed_z = torch.randn(args.n_generate, 512, device=device)  # number of generated images

    with torch.no_grad():
        sample_w = generator_ema.style([fixed_z])
        sample = \
            generator_ema(sample_w, input_is_latent=True, truncation=args.sample_truncation, randomize_noise=False)[0]
        sample_src = \
            generator_frozen(sample_w, input_is_latent=True, truncation=args.sample_truncation, randomize_noise=False)[
                0]

    # grid_rows = int(args.n_sample ** 0.5)
    # save_images(sample, sample_dir, "iter", grid_rows, 300)
    # save_images(sample_src, sample_dir, "src", grid_rows, 0)

    for i in range(args.n_generate):
        image_src = sample_src[i]
        image_tar = sample[i]
        # image_src = sample_src[i].cpu().numpy()
        # image_tar = sample[i].cpu().numpy()
        image_src_path = os.path.join(dataset_dir_src, f"{args.src_label}_{i:04d}.png")
        image_tar_path = os.path.join(dataset_dir_tar, f"{args.tar_label}_{i:04d}.png")
        save_generated_images(image_src, image_src_path, normalize=True)
        save_generated_images(image_tar, image_tar_path, normalize=True)


if __name__ == "__main__":
    device = "cuda"

    args = TrainOptions().parse()
    generate(args)