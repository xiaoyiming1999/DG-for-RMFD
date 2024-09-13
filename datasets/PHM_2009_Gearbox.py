#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import pandas as pd
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *

num_cls = 8
sig_size = 1024
cls_names = {0: 'spur 1', 1: 'spur 2', 2: 'spur 3', 3: 'spur 4',
             4: 'spur 5', 5: 'spur 6', 6: 'spur 7', 7: 'spur 8'}
total_conditions = {0: '30hz_High_1', 1: '30hz_High_2', 2: '30hz_Low_1', 3: '30hz_Low_2',
                    4: '35hz_High_1', 5: '35hz_High_2', 6: '35hz_Low_1', 7: '35hz_Low_2',
                    8: '40hz_High_1', 9: '40hz_High_2', 10: '40hz_Low_1', 11: '40hz_Low_2',
                    12: '45hz_High_1', 13: '45hz_High_2', 14: '45hz_Low_1', 15: '45hz_Low_2',
                    16: '50hz_High_1', 17: '50hz_High_2', 18: '50hz_Low_1', 19: '50hz_Low_2', }

def data_load(root, N, num_samples):
    data = []
    label = []

    for lab in range(num_cls):
        name = cls_names[lab]
        num_sample = 0
        path = os.path.join(root, name, name + '_' + total_conditions[N], name + '_' + total_conditions[N] + '.txt')
        data_temp = np.loadtxt(path)
        data_temp = data_temp[:, 0]
        data_temp = np.expand_dims(data_temp, axis=1)

        start, end = 0, sig_size
        while end <= data_temp.shape[0] and num_sample < num_samples:
            x = data_temp[start:end]
            # FFT
            # x = np.fft.fft(x)
            # x = np.abs(x) / len(x)
            # x = x[range(int(x.shape[0] / 2))]
            # x = x.reshape(-1, 1)
            data.append(x)
            label.append(lab)
            start += sig_size
            end += sig_size
            num_sample += 1

    return [data, label]

class PHM2009_DG(object):
    def __init__(self, transfer_task, normalizetype, num_train_samples, num_test_samples):
        self.data_dir = 'D:/datasets/PHM2009/PHM2009 Gearbox/PHM_Society_2009_Competition_Expanded_txt'
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normalizetype = normalizetype
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples

        self.data_transforms = {
            'train': Compose([
                Reshape(),
                # AddWhiteGaussian(),
                Normalize(self.normalizetype),
                # AddGaussian(),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'test': Compose([
                Reshape(),
                # AddWhiteGaussian(),
                Normalize(self.normalizetype),
                Retype(),
                # Scale(1)
            ])
        }

    def train_test_data_split(self):
        # get source_1
        list_data = data_load(self.data_dir, self.source_N[0], self.num_train_samples)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        print('source_1:\n', data_pd)
        source_1 = dataset(list_data=data_pd, transform=self.data_transforms['train'])

        # get source_2
        list_data = data_load(self.data_dir, self.source_N[1], self.num_train_samples)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        print('source_2:\n', data_pd)
        source_2 = dataset(list_data=data_pd, transform=self.data_transforms['train'])

        # get source_3
        list_data = data_load(self.data_dir, self.source_N[2], self.num_train_samples)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        print('source_3:\n', data_pd)
        source_3 = dataset(list_data=data_pd, transform=self.data_transforms['train'])

        # get target
        list_data = data_load(self.data_dir, self.target_N, self.num_test_samples)
        data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        print('target:\n', data_pd)
        target = dataset(list_data=data_pd, transform=self.data_transforms['test'])

        return source_1, source_2, source_3, target, num_cls



