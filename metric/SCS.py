from json.tool import main
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys

from torchvision.models.inception import inception_v3
from PIL import Image
import numpy as np
from scipy.stats import entropy
import argparse


class HED_Network(torch.nn.Module):

    def __init__(self):
        super().__init__()
        arguments_strModel = 'bsds500'
        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                              torch.hub.load_state_dict_from_url(
                                  url='http://content.sniklaus.com/github/pytorch-hed/network-' + arguments_strModel + '.pytorch',
                                  file_name='hed-' + arguments_strModel).items()})


    def forward(self, tenInput):
        tenBlue = (tenInput[:, 0:1, :, :] * 255.0) - 104.00698793
        tenGreen = (tenInput[:, 1:2, :, :] * 255.0) - 116.66876762
        tenRed = (tenInput[:, 2:3, :, :] * 255.0) - 122.67891434

        tenInput = torch.cat([tenBlue, tenGreen, tenRed], 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]),
                                                      mode='bilinear', align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]),
                                                      mode='bilinear', align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]),
                                                      mode='bilinear', align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]),
                                                      mode='bilinear', align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]),
                                                      mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv], 1))


def HED_estimate(tenInput, HED_net):
    with torch.no_grad():
        return HED_net(tenInput.cuda()).cpu()


def read_Image(img_name):
    return torch.from_numpy(numpy.ascontiguousarray(
        numpy.array(PIL.Image.open(img_name))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))


def SCS_eval(args, HED_net):
    # img_dir = args.img_pth
    # imgs_source = os.listdir(os.path.join(img_dir, 'source'))
    # imgs_target = os.listdir(os.path.join(img_dir, 'target'))
    imgs_source = args.imgs_source_path
    imgs_target = args.imgs_target_path
    imgs_source_HED = os.path.join(imgs_source, 'source_HED')
    imgs_target_HED = os.path.join(imgs_target, 'target_HED')
    if not os.path.exists(imgs_source_HED):
        os.makedirs(imgs_source_HED)
    if not os.path.exists(imgs_target_HED):
        os.makedirs(imgs_target_HED)

    img_source_names = [os.path.join(imgs_source, f) for f in os.listdir(imgs_source) if f.endswith(".png") or f.endswith(".jpg")]
    img_list = []
    for image_name in img_source_names:
        img_list.append(read_Image(image_name).unsqueeze(0))
    img_list = torch.cat(img_list, dim=0)
    tenOutput_list = []
    for i in range(20):
        tenOutput = HED_estimate(img_list[i * 25:(i + 1) * 25], HED_net)
        tenOutput_list.append(tenOutput)
        tenOutput = torch.cat(tenOutput_list)
    for num, item in enumerate(tenOutput):
        PIL.Image.fromarray((item.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(
            f'%s/%s' % (imgs_source_HED, imgs_source[num]))

    imgs_target_names = [os.path.join(imgs_target, f) for f in os.listdir(imgs_target) if f.endswith(".png") or f.endswith(".jpg")]
    img_list = []
    for image_name in imgs_target_names:
        img_list.append(read_Image(image_name).unsqueeze(0))
    tenOutput_list = []
    img_list = torch.cat(img_list, dim=0)
    for i in range(20):
        tenOutput = HED_estimate(img_list[i * 25:(i + 1) * 25], HED_net)
        tenOutput_list.append(tenOutput)
        tenOutput = torch.cat(tenOutput_list)
    for num, item in enumerate(tenOutput):
        PIL.Image.fromarray((item.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(
            f'%s/%s' % (imgs_target_HED, imgs_target[num]))

    score = 0

    for i in range(500):
        img_s = np.array(Image.open(f'%s/img{i}.png' % imgs_source_HED))
        img_t = np.array(Image.open(f'%s/img{i}.png' % imgs_target_HED))
        img_s = torch.from_numpy(img_s).cuda().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) / 255
        img_t = torch.from_numpy(img_t).cuda().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) / 255

        sim = 2 * (img_s * img_t).sum() / (img_s ** 2 + img_t ** 2).sum()
        score += sim

    print('SCS Score: %.3f' % (score / 500))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_source_path', type=str, default="../dataset/ipl/photo_to_disney/photo")
    parser.add_argument('--imgs_target_path', type=str, default="../dataset/ipl/photo_to_disney/disney")
    args = parser.parse_args()

    HED_net = HED_Network().cuda().eval()
    SCS_eval(args, HED_net)
