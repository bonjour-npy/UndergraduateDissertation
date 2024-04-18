"""
Example arguments:
train_improved.py   --frozen_gen_ckpt
                    ./pre_stylegan/stylegan2-ffhq-config-f.pt
                    --source_model_type
                    "ffhq"
                    --output_interval
                    100
                    --save_interval
                    100
                    --auto_compute
                    --source_class
                    "photo"
                    --target_class
                    "disney"
                    --batch_mapper
                    32
                    --lr_mapper
                    0.05
                    --iter_mapper
                    300
                    --ctx_init
                    "a photo of a"
                    --n_ctx
                    4
                    --lambda_l
                    1
                    --run_stage1
                    --run_stage2
                    --batch
                    2
                    --lr
                    0.002
                    --iter
                    300
                    --output_dir
                    ./output/disney_improved

prompts for generating prompts in text_templates:
    中文提示词：
    “针对将普通人像转换成迪士尼风格人物画像的任务，给出60个描述迪士尼人像特有特征的文字prompt。
    将上述生成的60个prompts放在同一个Python列表中，即每一个prompt作为该列表的字符串元素，输出整个Python列表。”
    English prompts:
    "For the task of converting a {source class} photo into a {target_class} photo,
    provide some text prompts describing the distinctive features of Disney character portraits.
    Put the generated 60 prompts into the same Python list, with each prompt as a string element of the list,
    and output the entire Python list."
"""
import os
import numpy as np
import torch
from tqdm import tqdm
import clip
import warnings

from model.ZSSGAN import ZSSGAN, SG2Generator
from utils.file_utils import save_images
from utils.training_utils import mixing_noise
from mapper import latent_mappers
import utils.text_templates as text_templates
from options.train_options import TrainOptions

warnings.filterwarnings("ignore")

dataset_sizes = {
    "ffhq": 1024,
    "dog": 512,
}


def text_encoder(source_prompts, source_tokenized_prompts, clip_model):
    """
    :param source_prompts: W空间随机生成的n_ctx长度的prompts与前缀、后缀latent code的concat结果
    :param source_tokenized_prompts: 人工输入prompts + 域标签的标记化结果 (1, 77)
    :param clip_model:
    :return:
    该text-encoder函数来自CLIP类的encode_text方法
    https://blog.csdn.net/m0_47623548/article/details/123056243
    """
    x = source_prompts + clip_model.positional_embedding.type(clip_model.dtype)  # element-wise加法
    x = x.permute(0, 2, 1, 3)  # NLD -> LND
    # (batch_size, 1, 77, n_dim) -> (batch_size, 77, 1, n_dim)
    for j in range(len(x)):  # 每个batch
        x[j] = clip_model.transformer(x[j])  # 为了满足Transformer对输入的要求：(seq_len, batch_size, n_dim)
    x = x.permute(0, 2, 1, 3)  # LND -> NLD
    # (batch_size, 77, 1, n_dim) -> (batch_size, 1, 77, n_dim)
    x = clip_model.ln_final(x).type(clip_model.dtype)  # layer normalization
    """
    text_projection: project the text embedding dimension to the dimension we get from the image encoder.
    It's a nn.Parameter with shape (transformer_width, embed_dim)
    x[:, torch.arange(x.shape[1]), source_tokenized_prompts.argmax(dim=-1)]: (batch_size, 1, n_dim)
    text_projection is initialized by nn.init.normal_, with shape of (transformer_width, embed_dim)
    """
    text_features = x[:, torch.arange(x.shape[1]), source_tokenized_prompts.argmax(dim=-1)] @ clip_model.text_projection
    # 这里选择人工初始化的prompt的eot层作为随机初始化prompt特征投影的输入，意味着将随机初始化的prompt以人工初始化prompt为目标
    # text_features = x[:, 1, source_tokenized_prompts.argmax(dim=-1)] @ clip_model.text_projection

    return text_features  # (batch_size, 1, n_dim)


def compute_text_features(prompts, source_prefix, source_suffix, source_tokenized_prompts, clip_model, batch):
    """
    :param prompts: Mapper生成的prompts的嵌入表示，(batch_size, n_ctx, n_dim)
    :param source_prefix: sot符号的嵌入表示，(1, 1, 512)
    :param source_suffix: n_ctx符号之后的所有符号（包括source_class、eot符号以及补足位）的的嵌入表示，(1, 77-n_ctx-1, 512)
    :param source_tokenized_prompts: sot + ctx_init + 域标签 + eot，(1, 77) 只用于选中eot符号层，不参与特征的计算
    :param clip_model:
    :param batch:
    :return: [sot + 学习到的prompts + class label + eot + etc]的嵌入表示的文字特征，即论文中提到的将学习到的
             image-specific prompts与class label进行concat
    """
    source_ctx = prompts.unsqueeze(1)  # (batch_size, 1, n_ctx, n_dim)
    source_prefix = source_prefix.expand(batch, -1, -1, -1)  # 对应维度复制输入参数的倍数，(batch_size, 1, 1, 512)
    source_suffix = source_suffix.expand(batch, -1, -1, -1)  # 对应维度复制输入参数的倍数，(batch_size, 1, 77-n_ctx-1, 512)
    source_prompts = torch.cat(
        [
            source_prefix,  # (batch, n_cls, 1, dim)
            source_ctx,  # (batch, n_cls, n_ctx, dim)
            source_suffix,  # (batch, n_cls, 77-n_ctx-1, dim)
        ],
        dim=2,
    )
    # source_prompts：将随机初始化的嵌入表示与前缀（sot符号）、后缀（n_ctx之后的class label + 补足位 + eot符号）的嵌入表示进行concat
    # (batch_size, 1, 77, n_dim)
    text_features = text_encoder(source_prompts, source_tokenized_prompts, clip_model)  # 返回随机初始化prompts的特征
    # (batch_size, 1, n_dim)
    return text_features


def ema(source, target, decay):
    """
    指数移动平均，将source模型中的参数更新至target模型中
    即在stage 2中初始化目标域生成器参数
    """
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


def train(args):
    if args.auto_compute:
        # auto compute num of auto_layer_k and size
        args.auto_layer_k = int(2 * (2 * np.log2(dataset_sizes[args.source_model_type]) - 2) / 3)
        args.size = dataset_sizes[args.source_model_type]

    # Set up networks, optimizers.
    print("Initializing networks...")
    net = ZSSGAN(args)
    with torch.no_grad():
        clip_loss_models, clip_model_weights = net.get_clip_loss_models()

    # Using original SG2 params. Not currently using r1 regularization, may need to change.
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)

    # Set up output directories.
    sample_dir = os.path.join(args.output_dir, "sample")
    ckpt_dir = os.path.join(args.output_dir, "checkpoint")
    ckpt_dir_m = os.path.join(ckpt_dir, "mapper")
    ckpt_dir_g = os.path.join(ckpt_dir, "generator")

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(ckpt_dir_m, exist_ok=True)
    os.makedirs(ckpt_dir_g, exist_ok=True)

    # set seed after all networks have been initialized. Avoids change of outputs due to model changes.
    # 确保每次运行实验时，生成的随机数序列都相同，从而保证实验的可重复性
    torch.manual_seed(args.seed1)
    torch.cuda.manual_seed_all(args.seed1)
    np.random.seed(args.seed1)

    # pre
    clip_model = clip_loss_models[args.clip_models[0]].model
    n_dim = clip_model.ln_final.weight.shape[0]  # 512
    if args.ctx_init != "":  # 如果提供了初始化字符串，重新计算n_ctx
        ctx_init = args.ctx_init.replace("_", " ")
        args.n_ctx = len(ctx_init.split(" "))
        prompt_prefix = ctx_init
    else:
        prompt_prefix = " ".join(["X"] * args.n_ctx)  # default n_ctx=4, prompt_prefix: "X X X X"
    source_prompts = [prompt_prefix + " " + args.source_class]  # 源域prompts ['a photo of a photo']
    target_prompts = [prompt_prefix + " " + args.target_class]  # 目标域prompts ['a photo of a disney']
    source_tokenized_prompts = torch.cat(
        [clip.tokenize(p) for p in source_prompts]).to(device)
    # (1, 77) 'sot a photo of a photo eot' 在经过tokenize后为tensor [[49406, 320, 1125, 539, 320, 1125, 49407, etc]]
    # 77是CLIP在tokenize方法中缺省的context_length，超过context_length将被truncate，不足的将用0补齐
    target_tokenized_prompts = torch.cat(
        [clip.tokenize(p) for p in target_prompts]).to(device)
    # (1, 77) 'sot a photo of a disney' 在经过tokenize后为tensor [[49406, 320, 1125, 539, 320, 4696, 49407, etc]]
    source_embedding = clip_model.token_embedding(source_tokenized_prompts).type(clip_model.dtype)
    # (1, 77, 512) 其中512是CLIP中token_embedding层的词嵌入的维度n_dim
    target_embedding = clip_model.token_embedding(target_tokenized_prompts).type(clip_model.dtype)
    source_prefix = source_embedding[:, :1, :].detach()
    # 即sot符号在潜在空间的嵌入表示，(1, 1, n_dim)
    source_suffix = source_embedding[:, 1 + args.n_ctx:, :].detach()
    # n_ctx之后的所有符号（包括source class、eot符号及补足位）在潜在空间的嵌入表示，(1, 77-n_ctx-1, n_dim)
    target_prefix = target_embedding[:, :1, :].detach()
    target_suffix = target_embedding[:, 1 + args.n_ctx:, :].detach()
    # n_ctx之后的所有符号（target class、eot符号及补足位）在潜在空间的嵌入表示，(1, 77-n_ctx-1, n_dim)

    if args.run_stage1:
        # stage 1
        print("stage 1: training mapper")
        mapper = latent_mappers.TransformerMapperV1(args, n_dim)
        # 由PixelNorm以及四层EqualLinear构成的Mapper，最终输出n_dim * n_ctx
        m_optim = torch.optim.Adam(mapper.mapping.parameters(), lr=args.lr_mapper)

        for i in tqdm(range(args.iter_mapper)):  # stage 1的每个epoch
            mapper.train()
            sample_z = mixing_noise(args.batch_mapper, 512, args.mixing, device)
            # (batch_size, 512)
            sample_w = net.generator_frozen.style(sample_z)  # Z空间到W空间的变换
            # (batch_size, 512)
            prompts = torch.reshape(mapper(sample_w[0]), (args.batch_mapper, args.n_ctx, n_dim)).type(clip_model.dtype)
            # 通过W空间的随机分布初始化需要学习的prompts (batch_size, n_ctx, n_dim)
            source_text_features = compute_text_features(prompts, source_prefix, source_suffix,
                                                         source_tokenized_prompts,
                                                         clip_model, args.batch_mapper)
            # image-specific prompts与源域标签concat之后得到的文字编码
            """
            在compute_text_features中最后一步使用text_projection进行投影的目的是为了与图像编码器的输出可以进行对比学习
            text_features的形状：(batch_size, 1, n_dim)，最终获得的是每个图像对应的mapper通过随机分布生成的prompt的特征
            """
            target_text_features = compute_text_features(prompts, target_prefix, target_suffix,
                                                         target_tokenized_prompts,
                                                         clip_model, args.batch_mapper)
            # image-specific prompts与目标域标签concat之后得到的文字编码
            with torch.no_grad():
                imgs = net.generator_frozen(sample_z,  # 使用Z空间随机初始化向量生成图像
                                            input_is_latent=False,
                                            truncation=1,
                                            randomize_noise=True)[0].detach()
                # (32, 3, 1024, 1024)
            # loss = clip_loss_models[args.clip_models[0]].global_clip_loss(
            #     img=imgs,
            #     text=args.source_class,  # 源域标签str
            #     delta_features=source_text_features,
            #     # (batch_size, 1, n_dim)
            #     is_contrastive=1,
            #     logit_scale=clip_model.logit_scale,
            #     prompt_prefix=prompt_prefix,
            #     target_text=args.target_class,
            #     target_delta_features=target_text_features,
            #     lambda_l=args.lambda_l,
            #     lambda_src=args.lambda_src)
            loss = clip_loss_models[args.clip_models[0]].global_clip_loss_v3(img=imgs, text=args.source_class,
                                                                             prompts=text_templates.ffhq_disney_templates,
                                                                             delta_features=source_text_features,
                                                                             is_contrastive=1,
                                                                             logit_scale=clip_model.logit_scale,
                                                                             prompt_prefix=prompt_prefix,
                                                                             target_text=args.target_class,
                                                                             target_delta_features=target_text_features,
                                                                             lambda_l=args.lambda_l)
            """
            由三部分组成：
            1. 对比学习损失：计算生成的源域prompts与源域图像之间的余弦相似度
            2. 目标域正则化：计算生成的目标域prompts与目标域标签之间的余弦相似度由--lambda_l控制
            3. 源域正则化：计算生成的源域prompts与源域标签之间的余弦相似度，由--lambda_src控制，默认是0
            """
            m_optim.zero_grad()
            clip_model.zero_grad()
            net.zero_grad()
            loss.backward()
            m_optim.step()

        # stage 1的全部epoch结束后保存mapper的参数
        torch.save({"m": mapper.state_dict(), "m_optim": m_optim.state_dict()},
                   f"{ckpt_dir_m}/mapper.pt")

    generator_ema = (SG2Generator(args.frozen_gen_ckpt, img_size=args.size, channel_multiplier=args.channel_multiplier)
                     .to(device))
    generator_ema.freeze_layers()
    generator_ema.eval()

    # reset seed
    torch.manual_seed(args.seed2)
    torch.cuda.manual_seed_all(args.seed2)
    np.random.seed(args.seed2)

    if args.run_stage2:
        # stage 2
        print("stage 2: training generator")
        if not args.run_stage1:
            print("loading mapper...")
            checkpoint_path = os.path.join(ckpt_dir_m, "mapper.pt")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            mapper = latent_mappers.TransformerMapperV1(args, n_dim)
            mapper.load_state_dict(checkpoint["m"], strict=True)
        mapper.eval()
        g_optim = torch.optim.Adam(
            net.generator_trainable.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )

        # Training loop
        fixed_z = torch.randn(args.n_sample, 512, device=device)  # random vectors
        # n_sample: 35
        for i in tqdm(range(1, args.iter + 1)):  # i starts with 1

            net.train()
            sample_z = mixing_noise(args.batch, 512, args.mixing, device)  # (batch_size, latent_dim)

            with torch.no_grad():
                sample_w = net.generator_frozen.style(sample_z)  # 从Z空间到W空间
                prompts = torch.reshape(mapper(sample_w[0]), (args.batch, args.n_ctx, n_dim)).type(clip_model.dtype)
                # shape: (batch_size, n_ctx, n_dim)
                source_text_features = compute_text_features(prompts, source_prefix, source_suffix,
                                                             source_tokenized_prompts, clip_model, args.batch)
                # 将image-specific prompts与源域class label concat起来送入text encoder
                target_text_features = compute_text_features(prompts, target_prefix, target_suffix,
                                                             target_tokenized_prompts, clip_model, args.batch)
                # 将image-specific prompts与目标域class label concat起来送入text encoder

            [sampled_src, sampled_dst], loss = net(sample_w, input_is_latent=True,
                                                   source_text_features=source_text_features,
                                                   target_text_features=target_text_features,
                                                   templates=prompt_prefix)  # "a photo of a"
            # 这里的loss默认是clip_directional_loss: criteria.clip_loss.CLIPLoss.clip_directional_loss

            net.zero_grad()
            loss.backward()
            g_optim.step()

            ema(net.generator_trainable.generator, generator_ema.generator, args.ema_decay)

            if i % args.output_interval == 0:
                net.eval()

                with torch.no_grad():
                    sample_w = generator_ema.style([fixed_z])
                    sample = generator_ema(sample_w, input_is_latent=True, truncation=args.sample_truncation,
                                           randomize_noise=False)[0]
                    sample_src = net.generator_frozen(sample_w, input_is_latent=True, truncation=args.sample_truncation,
                                                      randomize_noise=False)[0]

                grid_rows = int(args.n_sample ** 0.5)
                save_images(sample, sample_dir, "iter", grid_rows, i)
                save_images(sample_src, sample_dir, "src", grid_rows, 0)

            if (args.save_interval is not None) and (i > 0) and (i % args.save_interval == 0):
                torch.save(
                    {
                        "g_ema": generator_ema.generator.state_dict(),
                        "g_optim": g_optim.state_dict(),
                    },
                    f"{ckpt_dir_g}/{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    args = TrainOptions().parse()
    train(args)
