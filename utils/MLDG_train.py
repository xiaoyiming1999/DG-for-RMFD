#!/usr/bin/python
# -*- coding:utf-8 -*-
import copy
import random
import logging
import warnings
import numpy as np
import torch
from torch import nn
from torch import optim
from datasets import PHM2009_DG, PU_DG
from models import Feature_extractor, Classifier

class MLDG_train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # Load the datasets
        self.datasets = {}
        if args.data_name == 'PHM2009':
            self.datasets['src_1'], self.datasets['src_2'], self.datasets['src_3'], self.datasets['tar'], \
            self.datasets['num_cls'] = PHM2009_DG(transfer_task=args.transfer_task,
                                                  normalizetype=args.normalizetype,
                                                  num_train_samples=args.num_train_samples,
                                                  num_test_samples=args.num_test_samples).train_test_data_split()

        elif args.data_name == 'PU':
            self.datasets['src_1'], self.datasets['src_2'], self.datasets['src_3'], self.datasets['tar'], \
            self.datasets['num_cls'] = PU_DG(transfer_task=args.transfer_task,
                                                  normalizetype=args.normalizetype,
                                                  num_train_samples=args.num_train_samples,
                                                  num_test_samples=args.num_test_samples).train_test_data_split()

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[0] == 'src' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=True,
                                                           drop_last=False) for x in
                            ['src_1', 'src_2', 'src_3', 'tar']}

        # define model
        self.Gf = Feature_extractor()
        self.Gc = Classifier(num_classes=self.datasets['num_cls'])

        # Define the learning parameters
        parameter_list = [{"params": self.Gf.parameters(), "lr": args.lr},
                          {"params": self.Gc.parameters(), "lr": args.lr}]

        # Define the optimizer
        self.optimizer = optim.Adam(parameter_list, lr=args.lr, weight_decay=args.weight_decay)

        # Define the learning rate decay
        steps = [int(step) for step in args.steps.split(',')]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)

        self.Gf.to(self.device)
        self.Gc.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

    def train(self):

        args = self.args

        for epoch in range(0, args.epoch):
            logging.info('-' * 10 + 'Epoch {}/{}'.format(epoch, args.epoch - 1) + '-' * 10)

            iter_src_2 = iter(self.dataloaders['src_2'])
            iter_src_3 = iter(self.dataloaders['src_3'])

            #  learning rate
            logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))

            # Each epoch has a training and val phase
            for phase in ['src_1', 'tar']:
                # Define the temp variable
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0

                # Set model to train mode or test mode
                if phase == 'src_1':
                    self.Gf.train()
                    self.Gc.train()
                else:
                    self.Gf.eval()
                    self.Gc.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):

                    self.optimizer.zero_grad()
                    for p in self.Gf.parameters():
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                    for p in self.Gc.parameters():
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)

                    if phase == 'src_1':
                        src_1_inputs = inputs
                        src_1_labels = labels
                        src_2_inputs, src_2_labels = iter_src_2.next()
                        src_3_inputs, src_3_labels = iter_src_3.next()

                        src_1_inputs = src_1_inputs.to(self.device)
                        src_1_labels = src_1_labels.to(self.device)
                        src_2_inputs = src_2_inputs.to(self.device)
                        src_2_labels = src_2_labels.to(self.device)
                        src_3_inputs = src_3_inputs.to(self.device)
                        src_3_labels = src_3_labels.to(self.device)

                        all_domains = {0: [src_1_inputs, src_1_labels], 1: [src_2_inputs, src_2_labels],
                                       2: [src_3_inputs, src_3_labels]}
                        domain_index = [0, 1, 2]
                        meta_train_index = copy.deepcopy(domain_index)
                        meta_test_index = random.choice(domain_index)
                        meta_train_index.remove(meta_test_index)

                        meta_train_inputs_1 = all_domains[meta_train_index[0]][0]
                        meta_train_labels_1 = all_domains[meta_train_index[0]][1]
                        meta_train_inputs_2 = all_domains[meta_train_index[1]][0]
                        meta_train_labels_2 = all_domains[meta_train_index[1]][1]
                        meta_train_inputs = torch.cat((meta_train_inputs_1, meta_train_inputs_2), dim=0)
                        meta_train_labels = torch.cat((meta_train_labels_1, meta_train_labels_2), dim=0)

                        meta_test_inputs = all_domains[meta_test_index][0]
                        meta_test_labels = all_domains[meta_test_index][1]

                        with torch.set_grad_enabled(phase == 'src_1'):
                            # forward

                            inner_Gf = copy.deepcopy(self.Gf)
                            inner_Gc = copy.deepcopy(self.Gc)

                            inner_parameter_list = [{"params": inner_Gf.parameters(), "lr": args.inner_lr},
                                                    {"params": inner_Gc.parameters(), "lr": args.inner_lr}]
                            inner_optimizer = optim.Adam(inner_parameter_list, lr=args.inner_lr,
                                                         weight_decay=args.weight_decay)

                            features = inner_Gf(meta_train_inputs)
                            outputs = inner_Gc(features)
                            meta_train_loss = self.criterion(outputs, meta_train_labels)

                            # inner_backward
                            inner_optimizer.zero_grad()
                            meta_train_loss.backward()
                            inner_optimizer.step()

                            for p, q in zip(self.Gf.parameters(), inner_Gf.parameters()):
                                if q.grad is not None:
                                    p.grad.data.add_(q.grad.data)
                            for p, q in zip(self.Gc.parameters(), inner_Gc.parameters()):
                                if q.grad is not None:
                                    p.grad.data.add_(q.grad.data)

                            # calculate meta objective loss grad
                            meta_test_outputs = inner_Gc(inner_Gf(meta_test_inputs))
                            meta_test_loss = self.criterion(meta_test_outputs, meta_test_labels)

                            # inner_backward
                            inner_optimizer.zero_grad()
                            meta_test_loss.backward()

                            for p, q in zip(self.Gf.parameters(), inner_Gf.parameters()):
                                if q.grad is not None:
                                    p.grad.data.add_(args.trade_off * q.grad.data)
                            for p, q in zip(self.Gc.parameters(), inner_Gc.parameters()):
                                if q.grad is not None:
                                    p.grad.data.add_(args.trade_off * q.grad.data)

                            loss = meta_train_loss + meta_test_loss

                            # Updating the running mean and running variance in BN
                            _ = self.Gf(meta_train_inputs)
                            # Updating the learnable parameters in model
                            self.optimizer.step()

                        loss_temp = loss.item() * meta_train_labels.size(0)
                        epoch_loss += loss_temp
                        epoch_length += meta_train_labels.size(0)

                    else:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        features = self.Gf(inputs)
                        outputs = self.Gc(features)

                        loss = self.criterion(outputs, labels)

                        loss_temp = loss.item() * labels.size(0)
                        epoch_loss += loss_temp
                        epoch_length += labels.size(0)

                        pred = outputs.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        epoch_acc += correct

                epoch_loss = np.divide(epoch_loss, epoch_length)
                epoch_acc = np.divide(epoch_acc, epoch_length)

                if phase == 'src_1':
                    logging.info('Epoch: {} train-Loss: {:.4f}'.format(epoch, epoch_loss))

                if phase == 'tar':
                    logging.info('Epoch: {} tar-Loss: {:.4f} tar-Acc: {:.4f}'.format(
                        epoch, epoch_loss, epoch_acc))

            self.lr_scheduler.step()
