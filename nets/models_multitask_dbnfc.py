import torch
import torch.nn as nn
import torch.nn.functional as func
from collections import OrderedDict
from itertools import chain
import copy
import torch.nn.functional as F


class DigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits.
    """

    def __init__(self, num_classes=10, **kwargs):
        super(DigitModel, self).__init__()
        # features_conv
        self.add_module('conv1', nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2))
        self.add_module('conv2', nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        self.add_module('conv3', nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2))

        self.add_module('bn0', nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True)))
        self.add_module('bn1', nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True)))
        self.add_module('bn2', nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(inplace=True)))
        self.add_module('bn3', nn.Sequential(nn.BatchNorm1d(2048), nn.ReLU(inplace=True)))
        self.add_module('bn4', nn.Sequential(nn.BatchNorm1d(512), nn.ReLU(inplace=True)))

        self.add_module('bn0local', nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True)))
        self.add_module('bn1local', nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True)))
        self.add_module('bn2local', nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(inplace=True)))
        self.add_module('bn3local', nn.Sequential(nn.BatchNorm1d(2048), nn.ReLU(inplace=True)))
        self.add_module('bn4local', nn.Sequential(nn.BatchNorm1d(512), nn.ReLU(inplace=True)))

        # features_fc
        self.add_module('fc1', nn.Linear(128 * 7 * 7, 2048))
        self.add_module('fc2', nn.Linear(2048, 512))

        # classifier
        self.add_module('fc3', nn.Linear(512, 10))
        self.add_module('fc3local', nn.Linear(512, 10))

    def forward(self, x, mode="global"):
        conv1 = self.conv1(x)
        if mode == 'global':
            conv1 = self.bn0(conv1)
        else:
            conv1 = self.bn0local(conv1)
        x = F.max_pool2d(conv1, 2)

        conv2 = self.conv2(x)
        if mode == 'global':
            conv2 = self.bn1(conv2)
        else:
            conv2 = self.bn1local(conv2)
        x = F.max_pool2d(conv2, 2)

        conv3 = self.conv3(x)
        if mode == 'global':
            x = self.bn2(conv3)
        else:
            x = self.bn2local(conv3)

        out = x.view(x.size(0), -1)

        fc1 = self.fc1(out)
        if mode == 'global':
            fc1 = self.bn3(fc1)
        else:
            fc1 = self.bn3local(fc1)

        fc2 = self.fc2(fc1)
        if mode == 'global':
            fc2 = self.bn4(fc2)
        else:
            fc2 = self.bn4local(fc2)
        out = self.fc3(fc2)

        return out, x

