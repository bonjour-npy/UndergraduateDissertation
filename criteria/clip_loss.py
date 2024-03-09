import torch
import torchvision.transforms as transforms
import numpy as np
import clip
from PIL import Image
from utils.text_templates import imagenet_templates, part_templates


class DirectionLoss(torch.nn.Module):
    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse': torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae': torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)

        return self.loss_func(x, y)


class CLIPLoss(torch.nn.Module):
    def __init__(self, device, lambda_direction=1., lambda_patch=0., lambda_global=0., lambda_manifold=0.,
                 lambda_texture=0., patch_loss_type='mae', direction_loss_type='cosine',
                 clip_model='ViT-B/32'):
        super(CLIPLoss, self).__init__()

        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess

        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0,
                                                                                                 2.0])] +  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                             clip_preprocess.transforms[:2] +  # to match CLIP input scale assumptions
                                             clip_preprocess.transforms[4:])  # + skip convert PIL to tensor

        self.target_direction = None
        self.target_direction_fewshot = None
        self.patch_text_directions = None

        self.patch_loss = DirectionLoss(patch_loss_type)
        self.direction_loss = DirectionLoss(direction_loss_type)
        self.patch_direction_loss = torch.nn.CosineSimilarity(dim=2)
        self.loss_img = torch.nn.CrossEntropyLoss()
        self.loss_txt = torch.nn.CrossEntropyLoss()

        self.lambda_global = lambda_global
        self.lambda_patch = lambda_patch
        self.lambda_direction = lambda_direction
        self.lambda_manifold = lambda_manifold
        self.lambda_texture = lambda_texture

        self.src_text_features = None
        self.target_text_features = None
        self.angle_loss = torch.nn.L1Loss()

        self.model_cnn, preprocess_cnn = clip.load("RN50", device=self.device)
        self.preprocess_cnn = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0,
                                                                                                     2.0])] +  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                                 preprocess_cnn.transforms[
                                                 :2] +  # to match CLIP input scale assumptions
                                                 preprocess_cnn.transforms[4:])  # + skip convert PIL to tensor

        self.texture_loss = torch.nn.MSELoss()

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def encode_images_with_cnn(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess_cnn(images).to(self.device)
        return self.model_cnn.encode_image(images)

    def distance_with_templates(self, img: torch.Tensor, class_str: str, templates=imagenet_templates) -> torch.Tensor:

        text_features = self.get_text_features(class_str, templates)
        image_features = self.get_image_features(img)

        similarity = image_features @ text_features.T  # matrix multiplication

        return 1. - similarity

    def get_text_features(self, class_str: str, templates=imagenet_templates, norm: bool = True) -> torch.Tensor:

        template_text = self.compose_text_with_templates(class_str, templates)

        tokens = clip.tokenize(template_text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def simple_get_text_features(self, class_str: str, templates=imagenet_templates, norm: bool = True) -> torch.Tensor:

        tokens = clip.tokenize(class_str).to(self.device)

        text_features = self.encode_text(tokens).detach()  # 编码

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:

        image_features = self.encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def compute_text_direction(self, source_class, target_class) -> torch.Tensor:

        source_features = self.get_text_features(source_class)
        target_features = self.get_text_features(target_class)
        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        text_direction = text_direction / text_direction.norm(dim=-1, keepdim=True)

        return text_direction

    def compute_img2img_direction(self, source_images: torch.Tensor, target_images: list) -> torch.Tensor:

        with torch.no_grad():
            src_encoding = self.get_image_features(source_images)
            src_encoding = src_encoding.mean(dim=0, keepdim=True)

            target_encodings = []
            for target_img in target_images:
                preprocessed = self.clip_preprocess(Image.open(target_img)).unsqueeze(0).to(self.device)

                encoding = self.model.encode_image(preprocessed)
                encoding /= encoding.norm(dim=-1, keepdim=True)

                target_encodings.append(encoding)

            target_encoding = torch.cat(target_encodings, axis=0)
            target_encoding = target_encoding.mean(dim=0, keepdim=True)

            direction = target_encoding - src_encoding
            direction /= direction.norm(dim=-1, keepdim=True)

        return direction

    def set_text_features(self, source_class: str, target_class: str) -> None:
        source_features = self.get_text_features(source_class).mean(axis=0, keepdim=True)
        self.src_text_features = source_features / source_features.norm(dim=-1, keepdim=True)

        target_features = self.get_text_features(target_class).mean(axis=0, keepdim=True)
        self.target_text_features = target_features / target_features.norm(dim=-1, keepdim=True)

    def clip_angle_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor,
                        target_class: str) -> torch.Tensor:
        """
        clip_angle_loss
        :param src_img:
        :param source_class:
        :param target_img:
        :param target_class:
        :return:
        """
        if self.src_text_features is None:
            self.set_text_features(source_class, target_class)

        cos_text_angle = self.target_text_features @ self.src_text_features.T
        text_angle = torch.acos(cos_text_angle)

        src_img_features = self.get_image_features(src_img).unsqueeze(2)
        target_img_features = self.get_image_features(target_img).unsqueeze(1)

        cos_img_angle = torch.clamp(target_img_features @ src_img_features, min=-1.0, max=1.0)
        img_angle = torch.acos(cos_img_angle)

        text_angle = text_angle.unsqueeze(0).repeat(img_angle.size()[0], 1, 1)
        cos_text_angle = cos_text_angle.unsqueeze(0).repeat(img_angle.size()[0], 1, 1)

        return self.angle_loss(cos_img_angle, cos_text_angle)

    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:

        return [template.format(text) for template in templates]

    def clip_directional_loss(self, src_img: torch.Tensor, source_class, target_img: torch.Tensor, target_class,
                              source_delta_features=None, target_delta_features=None, templates=None) -> torch.Tensor:
        """
        论文中提到的改进版Directional CLIP Loss
        对于生成的源域和目标域prompts的特征，都向其中加入了人工初始化的prompt特征（即a photo of a {label}.的特征）
        target_direction（text_direction）即生成的源域和目标域prompts特征的差值
        edit_direction是生成的源域和目标域图像特征的差值
        最后再送入directional_loss中，计算余弦相似度并1-cosine_sim作为最终损失函数
        也就是使源域与目标域图像之间的差异逼近二者prompts文字之间的差异
        :param src_img:
        :param source_class:
        :param target_img:
        :param target_class:
        :param source_delta_features:
        :param target_delta_features:
        :param templates:
        :return:
        """
        if source_delta_features is None or target_delta_features is None:
            self.target_direction = self.compute_text_direction(source_class, target_class)  # non-ipl

        # 论文中提到的改进版directional clip loss
        else:
            # source_class
            source_text_features = self.get_text_features(source_class, templates=[templates], norm=False)
            for i in range(len(source_delta_features)):
                source_delta_features[i] = torch.add(source_delta_features[i], source_text_features)
            source_text_features = source_delta_features / source_delta_features.clone().norm(dim=-1, keepdim=True)

            # target_class
            target_text_features = self.get_text_features(target_class, templates=[templates], norm=False)
            for i in range(len(target_delta_features)):
                target_delta_features[i] = torch.add(target_delta_features[i], target_text_features)
            target_text_features = target_delta_features / target_delta_features.clone().norm(dim=-1, keepdim=True)

            text_direction = target_text_features - source_text_features
            text_direction = text_direction.squeeze(1)

            self.target_direction = text_direction / text_direction.clone().norm(dim=-1, keepdim=True)

        src_encoding = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)

        edit_direction = (target_encoding - src_encoding)
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)

        return self.direction_loss(edit_direction, self.target_direction).mean()

    def global_clip_loss(self, img, text, delta_features=None, is_contrastive=0, logit_scale=None, prompt_prefix=None,
                         target_text=None, target_delta_features=None, lambda_l=0, lambda_src=1) -> torch.Tensor:
        """
        :param img:
        :param text:
        :param delta_features: 生成prompts的文字特征，(batch_size, 1, n_dim)，无域标签
        :param is_contrastive:
        :param logit_scale:
        :param prompt_prefix:
        :param target_text:
        :param target_delta_features: 生成prompts的文字特征，(batch_size, 1, n_dim)，无域标签
        :param lambda_l:
        :param lambda_src:
        :return:
        """
        if is_contrastive:

            # image features
            image = self.preprocess(img)  # resize为(batch_size, channel, 224, 224)
            """
            0：Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])
            1：Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
            2：CenterCrop(size=(224, 224))
            3：Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            """
            image_features = self.model.encode_image(image).detach()  # VisionTransformer对image编码(batch_size, n_dim)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 归一化，方便后续计算余弦相似度(batch_size, n_dim)

            # text features
            prompt_prefix = prompt_prefix + " {}."  # 方便后续使用string的format方法向{}中加入域标签

            # a photo of a {}. + 源域/目标域标签
            init_source_features = self.get_text_features(text, templates=[prompt_prefix], norm=False)  # (1, n_dim)
            init_target_features = self.get_text_features(target_text, templates=[prompt_prefix],
                                                          norm=False)  # (1, n_dim)

            delta_source_features = torch.empty_like(delta_features)  # (batch_size, 1, n_dim)
            delta_target_features = torch.empty_like(target_delta_features)
            for i in range(len(delta_features)):  # 将生成的prompt沿batch_size维度与init prompt做element-wise相加
                delta_source_features[i] = torch.add(delta_features[i], init_source_features)
                delta_target_features[i] = torch.add(target_delta_features[i], init_target_features)
            # (batch_size, 1, n_dim)

            text_source_features = delta_source_features / delta_source_features.clone().norm(dim=-1, keepdim=True)
            text_target_features = delta_target_features / delta_target_features.clone().norm(dim=-1, keepdim=True)
            # (batch_size, n_dim)

            templates_source_features = self.get_text_features(text).mean(dim=0)  # 源域标签特征
            templates_target_features = self.get_text_features(target_text).mean(dim=0)  # 目标域标签特征

            text_target_features = text_target_features.squeeze(1)  # (batch_size, n_dim)
            text_source_features = text_source_features.squeeze(1)
            templates_source_features = templates_source_features.unsqueeze(0).expand(len(text_target_features), -1)
            templates_target_features = templates_target_features.unsqueeze(0).expand(len(text_target_features), -1)

            # 将生成的prompts（纯生成，无域标签）与域标签做direction_loss（这里生成的prompts是与初始化的prompts+域标签做过element-wise加法的）
            target_loss = self.direction_loss(text_target_features, templates_target_features).mean()
            source_loss = self.direction_loss(text_source_features, templates_source_features).mean()

            # cosine similarity as logits
            logit_scale = logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_source_features.squeeze(1).t()
            # (batch_size, batch_size)
            logits_per_text = torch.transpose(logits_per_image, dim0=0, dim1=1)
            ground_truth = torch.arange(len(image_features), dtype=torch.long, device=self.device)  # 每一个元素代表每行的正确标签索引

            total_loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text,
                                                                                        ground_truth)) / 2
            """
            total_loss计算了对比学习的损失函数
            source_loss是paper中的domain_loss
            lambda_src是paper公式6的lambda，域损失函数的正则项系数
            """
            return total_loss + lambda_l * (target_loss + lambda_src * source_loss)

        else:
            if not isinstance(text, list):
                text = [text]

            tokens = clip.tokenize(text).to(self.device)
            image = self.preprocess(img)

            logits_per_image, _ = self.model(image, tokens)

            return (1. - logits_per_image / 100).mean()

    def random_patch_centers(self, img_shape, num_patches, size):
        batch_size, channels, height, width = img_shape

        half_size = size // 2
        patch_centers = np.concatenate(
            [np.random.randint(half_size, width - half_size, size=(batch_size * num_patches, 1)),
             np.random.randint(half_size, height - half_size, size=(batch_size * num_patches, 1))], axis=1)

        return patch_centers

    def generate_patches(self, img: torch.Tensor, patch_centers, size):
        batch_size = img.shape[0]
        num_patches = len(patch_centers) // batch_size
        half_size = size // 2

        patches = []

        for batch_idx in range(batch_size):
            for patch_idx in range(num_patches):
                center_x = patch_centers[batch_idx * num_patches + patch_idx][0]
                center_y = patch_centers[batch_idx * num_patches + patch_idx][1]

                patch = img[batch_idx:batch_idx + 1, :, center_y - half_size:center_y + half_size,
                        center_x - half_size:center_x + half_size]

                patches.append(patch)

        patches = torch.cat(patches, axis=0)

        return patches

    def patch_scores(self, img: torch.Tensor, class_str: str, patch_centers, patch_size: int) -> torch.Tensor:

        parts = self.compose_text_with_templates(class_str, part_templates)
        tokens = clip.tokenize(parts).to(self.device)
        text_features = self.encode_text(tokens).detach()

        patches = self.generate_patches(img, patch_centers, patch_size)
        image_features = self.get_image_features(patches)

        similarity = image_features @ text_features.T

        return similarity

    def clip_patch_similarity(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor,
                              target_class: str) -> torch.Tensor:
        patch_size = 196  # TODO remove magic number

        patch_centers = self.random_patch_centers(src_img.shape, 4, patch_size)  # TODO remove magic number

        src_scores = self.patch_scores(src_img, source_class, patch_centers, patch_size)
        target_scores = self.patch_scores(target_img, target_class, patch_centers, patch_size)

        return self.patch_loss(src_scores, target_scores)

    def patch_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor,
                               target_class: str) -> torch.Tensor:

        if self.patch_text_directions is None:
            src_part_classes = self.compose_text_with_templates(source_class, part_templates)
            target_part_classes = self.compose_text_with_templates(target_class, part_templates)

            parts_classes = list(zip(src_part_classes, target_part_classes))

            self.patch_text_directions = torch.cat(
                [self.compute_text_direction(pair[0], pair[1]) for pair in parts_classes], dim=0)

        patch_size = 510  # TODO remove magic numbers

        patch_centers = self.random_patch_centers(src_img.shape, 1, patch_size)

        patches = self.generate_patches(src_img, patch_centers, patch_size)
        src_features = self.get_image_features(patches)

        patches = self.generate_patches(target_img, patch_centers, patch_size)
        target_features = self.get_image_features(patches)

        edit_direction = (target_features - src_features)
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)

        cosine_dists = 1. - self.patch_direction_loss(edit_direction.unsqueeze(1),
                                                      self.patch_text_directions.unsqueeze(0))

        patch_class_scores = cosine_dists * (edit_direction @ self.patch_text_directions.T).softmax(dim=-1)

        return patch_class_scores.mean()

    def cnn_feature_loss(self, src_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:

        src_features = self.encode_images_with_cnn(src_img)
        target_features = self.encode_images_with_cnn(target_img)

        return self.texture_loss(src_features, target_features)

    def compute_img2img_feature_match(self, trainable_img, few_shot_feature) -> torch.Tensor:

        src_encoding = self.get_image_features(trainable_img)
        src_encoding = src_encoding.mean(dim=0, keepdim=True)

        loss = 0
        for i in range(len(few_shot_feature)):
            loss = loss + (few_shot_feature[i] - src_encoding).norm(p=1, dim=-1)

        match_loss = loss[0] / len(few_shot_feature)

        return match_loss

    def forward(self, src_img: torch.Tensor, source_class, target_img: torch.Tensor, target_class,
                texture_image: torch.Tensor = None,
                source_delta_features=None, target_delta_features=None, templates=None):
        clip_loss = 0.0

        if self.lambda_global:
            clip_loss += self.lambda_global * self.global_clip_loss(target_img, [f"a {target_class}"],
                                                                    delta_features=source_delta_features,
                                                                    is_contrastive=0,
                                                                    logit_scale=self.model.logit_scale,
                                                                    prompt_prefix=templates,
                                                                    target_text=None,
                                                                    target_delta_features=target_delta_features,
                                                                    threshold=0.)

        if self.lambda_patch:
            clip_loss += self.lambda_patch * self.patch_directional_loss(src_img, source_class, target_img,
                                                                         target_class)

        if self.lambda_direction:  # default
            clip_loss += self.lambda_direction * self.clip_directional_loss(src_img, source_class, target_img,
                                                                            target_class,
                                                                            source_delta_features=source_delta_features,
                                                                            target_delta_features=target_delta_features,
                                                                            templates=templates)

        if self.lambda_manifold:
            clip_loss += self.lambda_manifold * self.clip_angle_loss(src_img, source_class, target_img, target_class)

        if self.lambda_texture and (texture_image is not None):
            clip_loss += self.lambda_texture * self.cnn_feature_loss(texture_image, target_img)

        return clip_loss
