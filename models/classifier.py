#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout())

        self.layer6 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout())

        self.layer7 = nn.Linear(256, num_classes)


    def forward(self, x):
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        return x