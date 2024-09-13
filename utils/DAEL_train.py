#!/usr/bin/python
# -*- coding:utf-8 -*-


import logging
import warnings
import numpy as np
import torch
from torch import nn
from torch import optim
from datasets import PHM2009_DG, PU_DG
from models import Feature_extractor, Classifier

class DAEL_train_utils(object):
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
        self.Gc_1 = Classifier(num_classes=self.datasets['num_cls'])
        self.Gc_2 = Classifier(num_classes=self.datasets['num_cls'])
        self.Gc_3 = Classifier(num_classes=self.datasets['num_cls'])

        # Define the learning parameters
        parameter_list = [{"params": self.Gf.parameters(), "lr": args.lr},
                          {"params": self.Gc_1.parameters(), "lr": args.lr},
                          {"params": self.Gc_2.parameters(), "lr": args.lr},
                          {"params": self.Gc_3.parameters(), "lr": args.lr}]

        # Define the optimizer
        self.optimizer = optim.Adam(parameter_list, lr=args.lr, weight_decay=args.weight_decay)

        # Define the learning rate decay
        steps = [int(step) for step in args.steps.split(',')]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)

        self.Gf.to(self.device)
        self.Gc_1.to(self.device)
        self.Gc_2.to(self.device)
        self.Gc_3.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()

    def train(self):

        args = self.args

        for epoch in range(0, args.epoch):
            logging.info('-' * 10 + 'Epoch {}/{}'.format(epoch, args.epoch - 1) + '-' * 10)

            #  learning rate
            logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            iter_src_2 = iter(self.dataloaders['src_2'])
            iter_src_3 = iter(self.dataloaders['src_3'])

            # Each epoch has a training and val phase
            for phase in ['src_1', 'tar']:
                # Define the temp variable
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0

                # Set model to train mode or test mode
                if phase == 'src_1':
                    self.Gf.train()
                    self.Gc_1.train()
                    self.Gc_2.train()
                    self.Gc_3.train()
                else:
                    self.Gf.eval()
                    self.Gc_1.eval()
                    self.Gc_2.eval()
                    self.Gc_3.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):

                    if phase == 'src_1':
                        src_1_inputs = inputs
                        src_1_labels = labels
                        src_2_inputs, src_2_labels = iter_src_2.next()
                        src_3_inputs, src_3_labels = iter_src_3.next()

                    else:
                        src_1_inputs = src_2_inputs = src_3_inputs = inputs
                        src_1_labels = src_2_labels = src_3_labels = labels

                    src_1_inputs = src_1_inputs.to(self.device)
                    src_1_labels = src_1_labels.to(self.device)
                    src_2_inputs = src_2_inputs.to(self.device)
                    src_2_labels = src_2_labels.to(self.device)
                    src_3_inputs = src_3_inputs.to(self.device)
                    src_3_labels = src_3_labels.to(self.device)

                    with torch.set_grad_enabled(phase == 'src_1'):
                        # forward
                        features_1 = self.Gf(src_1_inputs)
                        outputs_1 = self.Gc_1(features_1)
                        loss_1 = self.criterion(outputs_1, src_1_labels)
                        outputs_1 = torch.unsqueeze(outputs_1, dim=1)

                        features_2 = self.Gf(src_2_inputs)
                        outputs_2 = self.Gc_2(features_2)
                        loss_2 = self.criterion(outputs_2, src_2_labels)
                        outputs_2 = torch.unsqueeze(outputs_2, dim=1)

                        features_3 = self.Gf(src_3_inputs)
                        outputs_3 = self.Gc_3(features_3)
                        loss_3 = self.criterion(outputs_3, src_3_labels)
                        outputs_3 = torch.unsqueeze(outputs_3, dim=1)

                        outputs = torch.cat((outputs_1, outputs_2, outputs_3), dim=1)
                        outputs = torch.mean(outputs, dim=1)

                        loss = 1/3 * (loss_1 + loss_2 + loss_3)

                        if phase == 'src_1':

                            # Gc_1 outputs
                            Gc_1_for_src_3_outputs = torch.unsqueeze(self.Gc_1(features_3), dim=1)
                            Gc_1_for_src_2_outputs = torch.unsqueeze(self.Gc_1(features_2), dim=1)

                            # Gc_2 outputs
                            Gc_2_for_src_3_outputs = torch.unsqueeze(self.Gc_2(features_3), dim=1)
                            Gc_2_for_src_1_outputs = torch.unsqueeze(self.Gc_2(features_1), dim=1)

                            # Gc_3 outputs
                            Gc_3_for_src_1_outputs = torch.unsqueeze(self.Gc_3(features_1), dim=1)
                            Gc_3_for_src_2_outputs = torch.unsqueeze(self.Gc_3(features_2), dim=1)

                            # ensemble learning loss
                            Gc_2_and_3_for_src_1_outputs = torch.mean(
                                torch.cat((Gc_2_for_src_1_outputs, Gc_3_for_src_1_outputs), dim=1), dim=1)

                            Gc_1_and_3_for_src_2_outputs = torch.mean(
                                torch.cat((Gc_1_for_src_2_outputs, Gc_3_for_src_2_outputs), dim=1), dim=1)

                            Gc_1_and_2_for_src_3_outputs = torch.mean(
                                torch.cat((Gc_1_for_src_3_outputs, Gc_2_for_src_3_outputs), dim=1), dim=1)

                            el_loss_1 = self.MSE(Gc_2_and_3_for_src_1_outputs, outputs_1)

                            el_loss_2 = self.MSE(Gc_1_and_3_for_src_2_outputs, outputs_2)

                            el_loss_3 = self.MSE(Gc_1_and_2_for_src_3_outputs, outputs_3)

                            el_loss = 1/3 * (el_loss_1 + el_loss_2 + el_loss_3)

                            loss = loss + args.trade_off * el_loss

                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                        # Calculate the loss of a batch
                        loss_temp = loss.item() * labels.size(0)
                        # Calculate the loss for an epoch at a phase
                        epoch_loss += loss_temp
                        # Calculate the number of samples in an epoch at a phase
                        epoch_length += labels.size(0)

                        if phase == 'tar':
                            # get predicted labels
                            pred = outputs.argmax(dim=1)
                            # Calculate the number of correct predictions for a batch
                            correct = torch.eq(pred, src_1_labels).float().sum().item()
                            # Calculate the number of correct predictions for an epoch in each phase.
                            epoch_acc += correct

                epoch_loss = np.divide(epoch_loss, epoch_length)
                epoch_acc = np.divide(epoch_acc, epoch_length)

                if phase == 'src_1':
                    logging.info('Epoch: {} train-Loss: {:.4f}'.format(epoch, epoch_loss))

                if phase == 'tar':
                    logging.info('Epoch: {} tar-Loss: {:.4f} tar-Acc: {:.4f}'.format(
                        epoch, epoch_loss, epoch_acc))

            self.lr_scheduler.step()
