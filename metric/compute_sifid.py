import torch

from single_image_frechet_inception_distance import calculate_sifid_given_paths

# 准备真实数据分布和生成模型的图像数据
real_images_folder = '../dataset/fid_reference/test1'
generated_images_folder = '../dataset/fid_reference/test2'

# 计算FID距离值
fid_value = calculate_sifid_given_paths(real_images_folder,
                                        generated_images_folder,
                                        batch_size=10,
                                        cuda=True,
                                        dims=2048,
                                        suffix="jpg")
print('FID value:', fid_value)
