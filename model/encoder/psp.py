"""
This file defines the core research contribution
"""
import math
import torch
from torch import nn

from model.sg2_model import Generator
from model.encoder import fpn_encoders, restyle_psp_encoders

model_paths = {
    # models for backbones and losses
    'ir_se50': 'pretrained_models/model_ir_se50.pth',
    'resnet34': 'pretrained_models/resnet34-333f7ec4.pth',
    'moco': 'pretrained_models/moco_v2_800ep_pretrain.pt',
    # stylegan2 generators
    'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
    'stylegan_cars': 'pretrained_models/stylegan2-car-config-f.pt',
    'stylegan_ada_wild': 'pretrained_models/afhqwild.pt',
    # model for face alignment
    'shape_predictor': 'pretrained_models/shape_predictor_68_face_landmarks.dat',
    # models for ID similarity computation
    'curricular_face': 'pretrained_models/CurricularFace_Backbone.pth',
    'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
    'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
    'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
    # WEncoders for training on various domains
    'faces_w_encoder': 'pretrained_models/faces_w_encoder.pt',
    'cars_w_encoder': 'pretrained_models/cars_w_encoder.pt',
    'afhq_wild_w_encoder': 'pretrained_models/afhq_wild_w_encoder.pt',
    # models for domain adaptation
    'restyle_e4e_ffhq': 'pretrained_models/restyle_e4e_ffhq_encode.pt',
    'stylegan_pixar': 'pretrained_models/pixar.pt',
    'stylegan_toonify': 'pretrained_models/ffhq_cartoon_blended.pt',
    'stylegan_sketch': 'pretrained_models/sketch.pt',
    'stylegan_disney': 'pretrained_models/disney_princess.pt'
}

# specify the encoder types for pSp and e4e - this is mainly used for the inference scripts
ENCODER_TYPES = {
    'pSp': ['GradualStyleEncoder', 'ResNetGradualStyleEncoder', 'BackboneEncoder', 'ResNetBackboneEncoder'],
    'e4e': ['ProgressiveBackboneEncoder', 'ResNetProgressiveBackboneEncoder']
}

RESNET_MAPPING = {
    'layer1.0': 'body.0',
    'layer1.1': 'body.1',
    'layer1.2': 'body.2',
    'layer2.0': 'body.3',
    'layer2.1': 'body.4',
    'layer2.2': 'body.5',
    'layer2.3': 'body.6',
    'layer3.0': 'body.7',
    'layer3.1': 'body.8',
    'layer3.2': 'body.9',
    'layer3.3': 'body.10',
    'layer3.4': 'body.11',
    'layer3.5': 'body.12',
    'layer4.0': 'body.13',
    'layer4.1': 'body.14',
    'layer4.2': 'body.15',
}


class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.set_opts(opts)
        self.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(self.opts.output_size, 512, 8, channel_multiplier=2)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = fpn_encoders.GradualStyleEncoder(50, 'ir_se', self.n_styles, self.opts)
        elif self.opts.encoder_type == 'ResNetGradualStyleEncoder':
            encoder = fpn_encoders.ResNetGradualStyleEncoder(self.n_styles, self.opts)
        elif self.opts.encoder_type == 'BackboneEncoder':  # pSp: pretrained_models/restyle_psp_ffhq_encode.pt
            encoder = restyle_psp_encoders.BackboneEncoder(50, 'ir_se', self.n_styles, self.opts)
        elif self.opts.encoder_type == 'ResNetBackboneEncoder':
            encoder = restyle_psp_encoders.ResNetBackboneEncoder(self.n_styles, self.opts)
        else:
            raise Exception(f'{self.opts.encoder_type} is not a valid encoders')
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print(f'Loading ReStyle pSp from checkpoint: {self.opts.checkpoint_path}')
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(self.__get_keys(ckpt, 'encoder'), strict=False)
            self.decoder.load_state_dict(self.__get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            encoder_ckpt = self.__get_encoder_checkpoint()
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print(f'Loading decoder weights from pretrained path: {self.opts.stylegan_weights}')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=True)
            self.__load_latent_avg(ckpt, repeat=self.n_styles)

    def forward(self, x, return_codes=False, latent=None, resize=True, latent_mask=None, input_code=False,
                randomize_noise=True, inject_latent=None, return_latents=False, alpha=None, average_code=False,
                input_is_full=False):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # residual step
            if x.shape[1] == 6 and latent is not None:
                # learn error with respect to previous iteration
                codes = codes + latent
            else:
                # first iteration is with respect to the avg latent code
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        if average_code:
            input_is_latent = True
        else:
            input_is_latent = (not input_code) or (input_is_full)

        if return_codes:
            return codes
        # return codes

        ################################
        # Following is original return #
        ################################

        images, result_latent = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

    def __get_encoder_checkpoint(self):
        if "ffhq" in self.opts.dataset_type:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            # Transfer the RGB input of the irse50 network to the first 3 input channels of pSp's encoder
            if self.opts.input_nc != 3:
                shape = encoder_ckpt['input_layer.0.weight'].shape
                altered_input_layer = torch.randn(shape[0], self.opts.input_nc, shape[2], shape[3], dtype=torch.float32)
                altered_input_layer[:, :3, :, :] = encoder_ckpt['input_layer.0.weight']
                encoder_ckpt['input_layer.0.weight'] = altered_input_layer
            return encoder_ckpt
        else:
            print('Loading encoders weights from resnet34!')
            encoder_ckpt = torch.load(model_paths['resnet34'])
            # Transfer the RGB input of the resnet34 network to the first 3 input channels of pSp's encoder
            if self.opts.input_nc != 3:
                shape = encoder_ckpt['conv1.weight'].shape
                altered_input_layer = torch.randn(shape[0], self.opts.input_nc, shape[2], shape[3], dtype=torch.float32)
                altered_input_layer[:, :3, :, :] = encoder_ckpt['conv1.weight']
                encoder_ckpt['conv1.weight'] = altered_input_layer
            mapped_encoder_ckpt = dict(encoder_ckpt)
            for p, v in encoder_ckpt.items():
                for original_name, psp_name in RESNET_MAPPING.items():
                    if original_name in p:
                        mapped_encoder_ckpt[p.replace(original_name, psp_name)] = v
                        mapped_encoder_ckpt.pop(p)
            return encoder_ckpt

    @staticmethod
    def __get_keys(d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt
