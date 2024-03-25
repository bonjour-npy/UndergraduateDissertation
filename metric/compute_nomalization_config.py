import random
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os


class IplDataset(torch.utils.data.Dataset):
    def __init__(self, root="../dataset/ipl/photo_to_disney/disney", transform=None):
        self.images = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".png") or f.endswith(".jpg")]
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)


def compute_mean_and_std(ImageDataSet, root="../dataset/ipl/photo_to_disney/disney", device=torch.device("cuda")):
    # 图像预处理
    train_transform = transforms.Compose([transforms.Resize((299, 299)),
                                          transforms.ToTensor()])

    ipl_dataset = ImageDataSet(root=root, transform=train_transform)
    train_loader = DataLoader(dataset=ipl_dataset, batch_size=500, shuffle=True, num_workers=8)
    print(f"Computing mean and std of {root}")
    data = next(iter(train_loader)).to(device)
    mean = torch.mean(data, dim=(0, 2, 3))
    std = torch.std(data, dim=(0, 2, 3))
    # mean = np.mean(data.numpy(), axis=(0, 2, 3))
    # std = np.std(data.numpy(), axis=(0, 2, 3))
    return mean, std


if __name__ == '__main__':
    train_mean, train_std = compute_mean_and_std(IplDataset, root="../dataset/ipl/photo_to_disney/disney")
    print("train_mean:", train_mean)
    print("train_std:", train_std)
