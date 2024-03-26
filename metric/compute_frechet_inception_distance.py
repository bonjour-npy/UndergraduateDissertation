from frechet_inception_score import calculate_fid_given_paths
# from pytorch_fid.fid_score import calculate_fid_given_paths
from compute_nomalization_config import compute_mean_and_std

# 准备真实数据分布和生成模型的图像数据
real_images_folder = '../dataset/nada/photo_to_disney/photo'
generated_images_folder = '../dataset/nada/photo_to_disney/disney'

# 计算FID距离值
fid_value = calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                      batch_size=10,
                                      device='cuda',
                                      dims=2048,
                                      num_workers=0)
print('FID value:', fid_value)
