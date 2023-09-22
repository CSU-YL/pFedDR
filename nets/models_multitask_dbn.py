import torch
import torch.nn as nn
import torch.nn.functional as func
from collections import OrderedDict
import torch.nn.functional as F


class AlexNet(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        # features_conv
        self.add_module('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2))
        self.add_module('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2))
        self.add_module('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1))
        self.add_module('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1))
        self.add_module('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1))

        self.add_module('bn1', nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True)))
        self.add_module('bn2', nn.Sequential(nn.BatchNorm2d(192), nn.ReLU(inplace=True)))
        self.add_module('bn3', nn.Sequential(nn.BatchNorm2d(384), nn.ReLU(inplace=True)))
        self.add_module('bn4', nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(inplace=True)))
        self.add_module('bn5', nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(inplace=True)))
        self.add_module('bn6', nn.Sequential(nn.BatchNorm1d(4096), nn.ReLU(inplace=True)))
        self.add_module('bn7', nn.Sequential(nn.BatchNorm1d(4096), nn.ReLU(inplace=True)))

        self.add_module('bn1local', nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True)))
        self.add_module('bn2local', nn.Sequential(nn.BatchNorm2d(192), nn.ReLU(inplace=True)))
        self.add_module('bn3local', nn.Sequential(nn.BatchNorm2d(384), nn.ReLU(inplace=True)))
        self.add_module('bn4local', nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(inplace=True)))
        self.add_module('bn5local', nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(inplace=True)))
        self.add_module('bn6local', nn.Sequential(nn.BatchNorm1d(4096), nn.ReLU(inplace=True)))
        self.add_module('bn7local', nn.Sequential(nn.BatchNorm1d(4096), nn.ReLU(inplace=True)))

        # features_fc
        self.add_module('fc1', nn.Linear(256 * 6 * 6, 4096))
        self.add_module('fc2', nn.Linear(4096, 4096))

        # classifier
        self.add_module('fc3', nn.Linear(4096, num_classes))
        # self.add_module('fc3local', nn.Linear(4096, num_classes))

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x, mode="global"):
        conv1 = self.conv1(x)
        if mode == 'global':
            conv1 = self.bn1(conv1)
        else:
            conv1 = self.bn1local(conv1)
        x = F.max_pool2d(conv1, 3, 2)

        conv2 = self.conv2(x)
        if mode == 'global':
            conv2 = self.bn2(conv2)
        else:
            conv2 = self.bn2local(conv2)
        x = F.max_pool2d(conv2, 3, 2)

        conv3 = self.conv3(x)
        if mode == 'global':
            conv3 = self.bn3(conv3)
        else:
            conv3 = self.bn3local(conv3)

        conv4 = self.conv4(conv3)
        if mode == 'global':
            conv4 = self.bn4(conv4)
        else:
            conv4 = self.bn4local(conv4)

        conv5 = self.conv5(conv4)
        if mode == 'global':
            conv5 = self.bn5(conv5)
        else:
            conv5 = self.bn5local(conv5)
        x = F.max_pool2d(conv5, 3, 2)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        fc1 = self.fc1(x)
        if mode == 'global':
            fc1 = self.bn6(fc1)
        else:
            fc1 = self.bn6local(fc1)

        fc2 = self.fc2(fc1)
        if mode == 'global':
            fc2 = self.bn7(fc2)
            out = self.fc3(fc2)
        else:
            fc2 = self.bn7local(fc2)
            # out = self.fc3local(fc2)
            out = self.fc3(fc2)
        return out
