#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn


class Feature_extractor(nn.Module):
    def __init__(self):
        super(Feature_extractor, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=25, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=15)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=5)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.LeakyReLU(inplace=True)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=5)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.LeakyReLU(inplace=True)
        self.pool4 = nn.AdaptiveMaxPool1d(4)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)

        return x