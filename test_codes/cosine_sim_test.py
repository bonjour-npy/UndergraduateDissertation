import torch
import numpy as np


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim


a = torch.tensor([1.0, 2.0, 3.0, 4.0])
b = torch.tensor([4.0, 3.0, 2.0, 1.0])

a_norm = a / torch.norm(a, keepdim=True)
b_norm = b / torch.norm(b, keepdim=True)

cosine_sim_1 = torch.dot(a_norm, b_norm)

dot_product = torch.dot(a, b)
cosine_sim_2 = dot_product / (torch.norm(a) * torch.norm(b))

print(cosine_sim_1, cos_sim(a, b), cosine_sim_2)
