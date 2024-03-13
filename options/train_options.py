from argparse import ArgumentParser


class TrainOptions(object):
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument(
            "--frozen_gen_ckpt",
            type=str,
            help="Path to a pre-trained StyleGAN2 generator for use as the initial frozen network. " \
                 "If train_gen_ckpt is not provided, will also be used for the trainable generator initialization.",
            required=True
        )

        self.parser.add_argument(
            "--train_gen_ckpt",
            type=str,
            help="Path to a pre-trained StyleGAN2 generator for use as the initial trainable network."
        )

        self.parser.add_argument(
            "--adapted_gen_ckpt",
            type=str,
            help="Path to an adapted StyleGAN2 generator for evaluation."
        )

        self.parser.add_argument(
            "--output_dir",
            type=str,
            required=False,
            help="Path to output directory",
        )

        self.parser.add_argument(
            "--lambda_direction",
            type=float,
            default=1.0,
            help="Strength of directional clip loss",
        )

        ######################################################################################################
        # Non direction losses are unused in the paper. They are left here for those who want to experiment. #
        ######################################################################################################
        self.parser.add_argument(
            "--lambda_patch",
            type=float,
            default=0.0,
            help="Strength of patch-based clip loss",
        )

        self.parser.add_argument(
            "--lambda_global",
            type=float,
            default=0.0,
            help="Strength of global clip loss",
        )

        self.parser.add_argument(
            "--lambda_texture",
            type=float,
            default=0.0,
            help="Strength of texture preserving loss",
        )

        self.parser.add_argument(
            "--lambda_manifold",
            type=float,
            default=0.0,
            help="Strength of manifold constraint term"
        )
        ################################
        # End of Non-direction losses. #
        ################################

        self.parser.add_argument(
            "--save_interval",
            type=int,
            help="How often to save a model checkpoint. No checkpoints will be saved if not set.",
        )

        self.parser.add_argument(
            "--output_interval",
            type=int,
            default=300,
            help="How often to save an output image",
        )

        self.parser.add_argument(
            "--source_class",
            default="photo",
            help="Textual description of the source class.",
        )

        self.parser.add_argument(
            "--target_class",
            default="anime painting",
            help="Textual description of the target class.",
        )

        # Used for manual layer choices. Leave as None to use paper layers.
        self.parser.add_argument(
            "--phase",
            help="Training phase flag"
        )

        self.parser.add_argument(
            "--sample_truncation",
            default=0.7,
            type=float,
            help="Truncation value for sampled test images."
        )

        self.parser.add_argument(
            "--auto_layer_iters",
            type=int,
            default=1,
            help="Number of optimization steps when determining ideal training layers." \
                 "Set to 0 to disable adaptive layer choice (and revert to manual / all)"
        )

        self.parser.add_argument(
            "--auto_layer_k",
            type=int,
            default=1,
            help="Number of layers to train in each optimization step."
        )

        self.parser.add_argument(
            "--auto_layer_batch",
            type=int,
            default=8,
            help="Batch size for use in automatic layer selection step."
        )

        self.parser.add_argument(
            "--clip_models",
            nargs="+",
            type=str,
            default=["ViT-B/32"],
            help="Names of CLIP models to use for losses"
        )

        self.parser.add_argument(
            "--clip_model_weights",
            nargs="+",
            type=float,
            default=[1.0],
            help="Relative loss weights of the clip models"
        )

        #######################################################
        # Arguments for image style targets (instead of text) #
        #######################################################
        self.parser.add_argument(
            "--style_img_dir",
            type=str,
            help="Path to a directory containing images (png, jpg or jpeg) with a specific style to mimic"
        )

        self.parser.add_argument(
            "--img2img_batch",
            type=int,
            default=16,
            help="Number of images to generate for source embedding calculation."
        )
        #################################
        # End of image-style arguments. #
        #################################

        # Original rosinality args. Most of these are not needed and should probably be removed.

        self.parser.add_argument(
            "--iter", type=int, default=1000, help="total training iterations"
        )
        self.parser.add_argument(
            "--batch", type=int, default=2, help="batch sizes for each gpus"
        )

        self.parser.add_argument(
            "--n_sample",
            type=int,
            default=35,
            help="number of the samples generated during training",
        )

        self.parser.add_argument(
            "--size", type=int, default=256, help="image sizes for the model"
        )

        self.parser.add_argument(
            "--r1", type=float, default=10, help="weight of the r1 regularization"
        )

        self.parser.add_argument(
            "--path_regularize",
            type=float,
            default=2,
            help="weight of the path length regularization",
        )

        self.parser.add_argument(
            "--path_batch_shrink",
            type=int,
            default=2,
            help="batch size reducing factor for the path length regularization (reduce memory consumption)",
        )

        self.parser.add_argument(
            "--d_reg_every",
            type=int,
            default=16,
            help="interval of the applying r1 regularization",
        )

        self.parser.add_argument(
            "--g_reg_every",
            type=int,
            default=4,
            help="interval of the applying path length regularization",
        )

        self.parser.add_argument(
            "--mixing", type=float, default=0.0, help="probability of latent code mixing"
        )

        self.parser.add_argument(
            "--ckpt",
            type=str,
            default=None,
            help="path to the checkpoints to resume training",
        )

        self.parser.add_argument("--lr", type=float, default=0.002, help="learning rate")

        self.parser.add_argument(
            "--channel_multiplier",
            type=int,
            default=2,
            help="channel multiplier factor for the model. config-f = 2, else = 1",
        )

        self.parser.add_argument(
            "--augment", action="store_true", help="apply non leaking augmentation"
        )

        self.parser.add_argument(
            "--augment_p",
            type=float,
            default=0,
            help="probability of applying augmentation. 0 = use adaptive augmentation",
        )

        self.parser.add_argument(
            "--ada_target",
            type=float,
            default=0.6,
            help="target augmentation probability for adaptive augmentation",
        )

        self.parser.add_argument(
            "--ada_length",
            type=int,
            default=500 * 1000,
            help="target duraing to reach augmentation probability for adaptive augmentation",
        )

        self.parser.add_argument(
            "--ada_every",
            type=int,
            default=256,
            help="probability update interval of the adaptive augmentation",
        )

        self.parser.add_argument(
            "--source_model_type",
            type=str,
            default="ffhq",
            help="",
        )

        self.parser.add_argument(
            "--auto_compute",
            action="store_true",
            help="auto compute num of auto_layer_k and size",
        )

        self.parser.add_argument(
            '--iter_mapper',
            type=int,
            default=300,
            help="iterations for mapper",
        )

        self.parser.add_argument(
            '--batch_mapper',
            type=int,
            default=32,
            help="batchsize for mapper",
        )

        self.parser.add_argument(
            '--lr_mapper',
            type=float,
            default=0.05,
            help="lr for mapper",
        )

        self.parser.add_argument(
            '--run_stage1',
            action="store_true",  # 带这个参数就指定为True
            help="training mapper",
        )

        self.parser.add_argument(
            '--run_stage2',
            action="store_true",
            help="training generator",
        )

        self.parser.add_argument(
            '--lambda_l',
            type=float,
            default=1.,
            help="control the intensity of domain regularization loss",
        )

        self.parser.add_argument(
            '--seed1',
            type=int,
            default=3,
            help="random seed for stage 1",
        )

        self.parser.add_argument(
            '--seed2',
            type=int,
            default=2,
            help="random seed for stage 2",
        )

        self.parser.add_argument(
            "--ema_decay",
            type=float,
            default=0.99,
            help="ema decay"
        )

        self.parser.add_argument(
            '--lambda_src',
            type=float,
            default=0.,
            help="optional, for source domain regularization"
        )

        ########################
        # Arguments from CoOp. #
        ########################
        self.parser.add_argument(
            "--n_ctx",
            type=int,
            default=4,
            help="number of context tokens prompt"
        )

        self.parser.add_argument(
            "--ctx_init",
            type=str,
            default="",
            help="initialization words"
        )

        #########################################
        # Arguments for quantitative evaluation #
        #########################################
        self.parser.add_argument(
            "--dataset_output_dir",
            type=str,
            default="./dataset/ipl",
            help="output dir for evaluation dataset"
        )

        self.parser.add_argument(
            "--n_generate",
            type=int,
            default=30,
            help="number of generated images to form evaluation dataset for quantitative analysis"
        )

        self.parser.add_argument(
            "--src_label",
            type=str,
            help="text label of source domain"
        )

        self.parser.add_argument(
            "--tar_label",
            type=str,
            help="text label of target domain"
        )

    def parse(self):
        opts = self.parser.parse_args()

        if len(opts.clip_models) != len(opts.clip_model_weights):
            raise ValueError("Number of clip model names must match number of model weights")

        opts.train_gen_ckpt = opts.train_gen_ckpt or opts.frozen_gen_ckpt

        return opts
