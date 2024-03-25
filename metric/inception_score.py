import os.path
import torch
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models.inception import inception_v3, Inception_V3_Weights
import numpy as np
from scipy.stats import entropy

from compute_nomalization_config import compute_mean_and_std


class ImageDataset(torch.utils.data.Dataset):
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


def inception_score(dataset, batch_size=32, resize=False, splits=1):
    """
    Computes the inception score of the generated images
    dataset: Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    batch_size: batch size for feeding into Inception v3
    splits: number of splits
    """
    N = len(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').to(device)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.to(device)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    # root = "../dataset/ipl/photo_to_disney/disney"
    root = "../dataset/nada/photo_to_disney/disney"
    mean, std = compute_mean_and_std(ImageDataset, root=root)
    transform = transforms.Compose([transforms.Resize(299),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    ipl_dataset = ImageDataset(root=root, transform=transform)
    print(inception_score(ipl_dataset, batch_size=64, resize=True, splits=10))
