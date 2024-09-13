#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import pandas as pd
import scipy

from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *

num_cls = 14
sig_size = 1024
cls_names = {0: 'KA04', 1: 'KA15', 2: 'KA16', 3: 'KA22', 4: 'KA30',
              5: 'KI04', 6: 'KI14', 7: 'KI16', 8: 'KI17', 9: 'KI18', 10: 'KI21',
              11: 'KB27', 12: 'KB23', 13: 'KB24'}

total_conditions = {0: 'N09_M07_F10_', 1: 'N15_M01_F10_', 2: 'N15_M07_F04_', 3: 'N15_M07_F10_'}

def data_load(root, N, num_samples):
    data = []
    label = []

    for lab in range(num_cls):
        num_sample = 0
        name = cls_names[lab]
        class_path = os.path.join(root, name, name)
        all_data_temp = []
        for i in range(1, 20):
            current_name =total_conditions[N] + name + '_' + str(i)
            path = os.path.join(class_path, current_name + '.mat')
            data_temp = scipy.io.loadmat(path)
            data_temp = data_temp[current_name][0][0][2][0][6][2]
            data_temp = np.array(data_temp)
            data_temp = data_temp.reshape(-1, 1)
            data_temp = data_temp[0:sig_size * 100, 0]
            all_data_temp.append(data_temp)
        all_data_temp = np.concatenate(all_data_temp)
        all_data_temp = np.expand_dims(all_data_temp, axis=1)

        start, end = 0, sig_size
        while end <= all_data_temp.shape[0] and num_sample < num_samples:
            x = all_data_temp[start:end]
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

class PU_DG(object):
    def __init__(self, transfer_task, normalizetype, num_train_samples, num_test_samples):
        self.data_dir = 'D:/datasets/PU'
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


