import torch
import numpy
import clip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model, _ = clip.load('ViT-B/32', device)

clip_model.eval()

text_1 = clip.tokenize('I am happy', truncate=True).to(device)
feature_1 = clip_model.encode_text(text_1)

text_2 = clip.tokenize(['I am happy', 'I am happy'], truncate=True).to(device)
feature_2 = clip_model.encode_text(text_2)

text_3 = clip.tokenize(['I am happy', 'I am happy', 'I am happy'], truncate=True).to(device)
feature_3 = clip_model.encode_text(text_3)

text_4 = clip.tokenize(['I am happy', 'I am sad', 'I am angry'], truncate=True).to(device)
feature_4 = clip_model.encode_text(text_4)

print((text_1 - text_2[0]).sum(), '\n', (text_1[0] - text_3[0]).sum(), '\n', (text_1[0] - text_4[0]).sum())
print((feature_1 - feature_2[0]).sum(), '\n', (feature_1[0] - feature_3[0]).sum(), '\n', (feature_1[0] - feature_4[0]).sum())
