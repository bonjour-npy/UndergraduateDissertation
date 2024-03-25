from pytorch_fid import fid_score

# 准备真实数据分布和生成模型的图像数据
real_images_folder = '../dataset/fid_reference/disney'
generated_images_folder = '../dataset/nada/photo_to_disney/disney'

# 计算FID距离值
fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                batch_size=3,
                                                device='cuda',
                                                dims=2048,
                                                num_workers=0)
print('FID value:', fid_value)
