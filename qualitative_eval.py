"""
photo to disney evaluation
python eval.py  --src_ckpt ./pre_stylegan/stylegan2-ffhq-config-f.pt \
                --nada_ckpt ./pre_stylegan/disney_princess_nada.pt \
                --ipl_ckpt ./adapted_generator/ffhq/disney.pt \
                --ipl_improved_ckpt ./output/disney_improved/V1_checkpoint/generator/000300.pt \
                --source_model_type "ffhq" \
                --auto_compute \
                --output_dir ./eval/disney
"""
import os
import torch
import numpy as np
from model.ZSSGAN import ZSSGAN, SG2Generator
from options.train_options import TrainOptions
from utils.file_utils import save_images
import warnings

warnings.filterwarnings("ignore")

dataset_sizes = {
    "ffhq": 1024,
    "dog": 512,
}


def eval(args):
    if args.auto_compute:
        args.size = dataset_sizes[args.source_model_type]

    sample_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)

    generator_src = SG2Generator(args.src_ckpt, img_size=args.size, channel_multiplier=args.channel_multiplier,
                                 device=device)
    generator_src.freeze_layers()
    generator_src.eval()

    generator_nada = SG2Generator(args.nada_ckpt, img_size=args.size,
                                  channel_multiplier=args.channel_multiplier, device=device)
    generator_nada.freeze_layers()
    generator_nada.eval()

    generator_ipl_improved = SG2Generator(args.ipl_improved_ckpt, img_size=args.size,
                                          channel_multiplier=args.channel_multiplier, device=device)
    generator_ipl_improved.freeze_layers()
    generator_ipl_improved.eval()

    generator_ipl_improved = SG2Generator(args.ipl_improved_ckpt, img_size=args.size,
                                          channel_multiplier=args.channel_multiplier, device=device)
    generator_ipl_improved.freeze_layers()
    generator_ipl_improved.eval()

    # torch.manual_seed(args.seed1)
    # torch.cuda.manual_seed_all(args.seed1)
    # np.random.seed(args.seed1)

    torch.manual_seed(args.seed2)
    torch.cuda.manual_seed_all(args.seed2)
    np.random.seed(args.seed2)

    fixed_z = torch.randn(args.n_sample, 512, device=device)

    with torch.no_grad():
        sample_w = generator_src.style([fixed_z])
        sample_src = \
            generator_src(sample_w, input_is_latent=True, truncation=args.sample_truncation, randomize_noise=False)[0]
        sample_nada = \
            generator_nada(sample_w, input_is_latent=True, truncation=args.sample_truncation, randomize_noise=False)[
                0]
        sample_ipl_improved = \
            generator_ipl_improved(sample_w, input_is_latent=True, truncation=args.sample_truncation,
                                   randomize_noise=False)[0]

    grid_rows = int(3)
    # sample = torch.concat(
    #     [sample_src[0].unsqueeze(0), sample_nada[0].unsqueeze(0), sample_ipl_improved[0].unsqueeze(0)], dim=0)
    # for i in range(args.n_sample - 1):
    #     sample = torch.concat([sample, sample_src[i + 1].unsqueeze(0), sample_nada[i + 1].unsqueeze(0),
    #                            sample_ipl_improved[i + 1].unsqueeze(0)], dim=0)
    samples = torch.stack([sample_src, sample_nada, sample_ipl_improved], dim=1)
    samples = samples.view(-1, *samples.shape[2:])
    save_images(samples, sample_dir, "iter", grid_rows, 300)


if __name__ == "__main__":
    device = "cuda"

    args = TrainOptions().parse()
    eval(args)
