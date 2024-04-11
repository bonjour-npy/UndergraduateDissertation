import torch
from torch import nn
from torch.nn import Module
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from mapper.stylegan2.model import EqualLinear, PixelNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mapper(Module):
    def __init__(self, opts, n_dim):
        super(Mapper, self).__init__()

        self.opts = opts
        layers = [PixelNorm()]  # 将每个点归一化（除以latent code的模长），避免输入noise的极端权重，改善稳定性
        style_dim = n_dim

        for i in range(3):
            layers.append(EqualLinear(512, 512, lr_mul=0.01, activation='fused_lrelu'))

        layers.append(EqualLinear(512, style_dim * opts.n_ctx, lr_mul=0.01, activation='fused_lrelu'))

        self.mapping = nn.Sequential(*layers)  # 解压后加入sequential中

    def forward(self, x):
        x = self.mapping(x)
        return x


class SingleMapper(Module):
    def __init__(self, opts, n_dim):
        super(SingleMapper, self).__init__()

        self.opts = opts

        # self.mapping = Mapper(opts, n_dim).to("cuda")
        self.mapping = Mapper(opts, n_dim).to(device)

    def forward(self, x):
        out = self.mapping(x)
        return out


class LevelsMapper(Module):
    def __init__(self, opts):
        super(LevelsMapper, self).__init__()

        self.opts = opts

        if not opts.no_coarse_mapper:
            self.course_mapping = Mapper(opts)
        if not opts.no_medium_mapper:
            self.medium_mapping = Mapper(opts)
        if not opts.no_fine_mapper:
            self.fine_mapping = Mapper(opts)

    def forward(self, x):
        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]

        if not self.opts.no_coarse_mapper:
            x_coarse = self.course_mapping(x_coarse)
        else:
            x_coarse = torch.zeros_like(x_coarse)

        if not self.opts.no_medium_mapper:
            x_medium = self.medium_mapping(x_medium)
        else:
            x_medium = torch.zeros_like(x_medium)

        if not self.opts.no_fine_mapper:
            x_fine = self.fine_mapping(x_fine)
        else:
            x_fine = torch.zeros_like(x_fine)

        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)

        return out


###############################
# Add Transformer Mapper here #
###############################
class TransformerMapperV1(nn.Module):
    def __init__(self, opts, n_dim):
        super(TransformerMapperV1, self).__init__()
        self.opts = opts
        self.n_dim = n_dim

        layers = [PixelNorm()]  # 将每个点归一化（除以模长），避免输入noise的极端权重，改善稳定性

        # 自定义Transformer编码器层配置
        transformer_layer = TransformerEncoderLayer(d_model=512, nhead=2, dim_feedforward=1024, dropout=0.1)

        # 构建Transformer编码器
        self.transformer_encoder = TransformerEncoder(transformer_layer, num_layers=3)
        layers.append(self.transformer_encoder)

        # 最后一个全连接层，输出维度保持不变
        self.final_linear = EqualLinear(512, n_dim * opts.n_ctx, lr_mul=0.01, activation='fused_lrelu')
        layers.append(self.final_linear)

        self.mapping = nn.Sequential(*layers).to(device)

    def forward(self, x):
        out = self.mapping(x)
        return out


class TransformerMapperV2(nn.Module):
    """
    改良版transformer mapper，增加多头注意力，减小transformer encoder的层数，防止学习到的源域图像细节过拟合
    同时去掉开头的PixelNorm，防止与transformer中的layer normalization冲突
    并在transformer encoder之后加入Pixel Norm以及全连接层
    """
    def __init__(self, opts, n_dim):
        super(TransformerMapperV2, self).__init__()
        self.opts = opts
        self.n_dim = n_dim

        layers = []  # transformer中有layer normalization，不需要进行PixelNorm

        # 自定义Transformer编码器层配置
        transformer_layer = TransformerEncoderLayer(d_model=512, nhead=3, dim_feedforward=1024, dropout=0.1)

        # 构建Transformer编码器
        self.transformer_encoder = TransformerEncoder(transformer_layer, num_layers=2)
        layers.append(self.transformer_encoder)

        # 再过一次PixelNorm以及全连接层，将每个点归一化（除以模长），避免输入noise的极端权重，改善稳定性
        layers.append(PixelNorm())
        self.linear = EqualLinear(512, 512, lr_mul=0.01, activation='fused_lrelu')
        layers.append(self.Linear)

        # 最后一个全连接层，输出维度保持不变
        self.final_linear = EqualLinear(512, n_dim * opts.n_ctx, lr_mul=0.01, activation='fused_lrelu')
        layers.append(self.final_linear)

        self.mapping = nn.Sequential(*layers).to(device)

    def forward(self, x):
        out = self.mapping(x)
        return out
