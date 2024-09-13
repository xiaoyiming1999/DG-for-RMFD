#!/usr/bin/python
# -*- coding:utf-8 -*-


import logging
import warnings
import numpy as np
import torch
from torch import nn
from torch import optim
from datasets import PHM2009_DG, PU_DG
from models import Feature_extractor, Classifier, Discriminator, MMD, CORAL, TripletLoss, mixup_augmentation

class Train_utils(object):
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
        self.Gd = Discriminator(output_size=len(args.transfer_task[0]))

        # Define the learning parameters
        if args.method == 'DANN':
            parameter_list = [{"params": self.Gf.parameters(), "lr": args.lr},
                              {"params": self.Gc.parameters(), "lr": args.lr},
                              {"params": self.Gd.parameters(), "lr": args.lr}]
        else:
            parameter_list = [{"params": self.Gf.parameters(), "lr": args.lr},
                              {"params": self.Gc.parameters(), "lr": args.lr}]

        # Define the optimizer
        self.optimizer = optim.Adam(parameter_list, lr=args.lr, weight_decay=args.weight_decay)

        # Define the learning rate decay
        steps = [int(step) for step in args.steps.split(',')]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)

        self.Gf.to(self.device)
        self.Gc.to(self.device)
        if args.method == 'DANN':
            self.Gd.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.Tripletloss = TripletLoss(margin=2)

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
                    if args.method == 'DANN':
                        self.Gd.train()

                else:
                    self.Gf.eval()
                    self.Gc.eval()
                    if args.method == 'DANN':
                        self.Gd.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):

                    if phase == 'src_1':
                        src_1_inputs = inputs
                        src_1_labels = labels
                        src_2_inputs, src_2_labels = iter_src_2.next()
                        src_3_inputs, src_3_labels = iter_src_3.next()
                        inputs = torch.cat((src_1_inputs, src_2_inputs, src_3_inputs), dim=0)
                        labels = torch.cat((src_1_labels, src_2_labels, src_3_labels), dim=0)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        interval = int(labels.size(0) / len(args.transfer_task[0]))

                        if args.method == 'Mixup':
                            src_1_labels = src_1_labels.to(self.device)
                            src_2_labels = src_2_labels.to(self.device)
                            src_3_labels = src_3_labels.to(self.device)
                            aug_inputs, lamda = mixup_augmentation(inputs, interval, self.device)

                    else:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                    with torch.set_grad_enabled(phase == 'src_1'):
                        # forward
                        features = self.Gf(inputs)
                        outputs = self.Gc(features)

                        loss = self.criterion(outputs, labels)

                        if phase == 'src_1':

                            if args.method == 'DANN':
                                domain_label_src1 = torch.zeros(interval).float()
                                domain_label_src2 = torch.ones(interval).float()
                                domain_label_src3 = 2 * torch.ones(interval).float()
                                domain_labels = torch.cat((domain_label_src1, domain_label_src2, domain_label_src3),
                                                          dim=0)
                                domain_labels = domain_labels.to(self.device)

                                ad_outputs = self.Gd(features)
                                adv_loss = self.criterion(ad_outputs, domain_labels.long())
                                loss = loss + args.trade_off * adv_loss

                            elif args.method == 'CORAL':
                                distance_loss_1 = CORAL(features.narrow(0, 0, interval),
                                                        features.narrow(0, interval, interval))
                                distance_loss_2 = CORAL(features.narrow(0, 0, interval),
                                                        features.narrow(0, 2 * interval, interval))
                                distance_loss_3 = CORAL(features.narrow(0, interval, interval),
                                                        features.narrow(0, 2 * interval, interval))
                                distance_loss = args.trade_off * 1 / 3 * (distance_loss_1 +
                                                                          distance_loss_2 + distance_loss_3)

                                loss = loss + distance_loss

                            elif args.method == 'MMD':
                                distance_loss_1 = MMD(features.narrow(0, 0, interval),
                                                        features.narrow(0, interval, interval))
                                distance_loss_2 = MMD(features.narrow(0, 0, interval),
                                                        features.narrow(0, 2 * interval, interval))
                                distance_loss_3 = MMD(features.narrow(0, interval, interval),
                                                        features.narrow(0, 2 * interval, interval))
                                distance_loss = args.trade_off * 1 / 3 * (distance_loss_1 +
                                                                          distance_loss_2 + distance_loss_3)

                                loss = loss + distance_loss

                            elif args.method == 'Triplet_loss':
                                loss = loss + args.trade_off * self.Tripletloss(features, labels)

                            elif args.method == 'Mixup':
                                aug_outputs = self.Gc(self.Gf(aug_inputs))
                                aug_loss = lamda[0, 0] * self.criterion(aug_outputs, src_1_labels) \
                                           + lamda[0, 1] * self.criterion(aug_outputs, src_2_labels) \
                                           + lamda[0, 2] * self.criterion(aug_outputs, src_3_labels)
                                loss = loss + aug_loss

                            else:
                                loss = loss

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
                            correct = torch.eq(pred, labels).float().sum().item()
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








