#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
import logging
import warnings

from utils.logger import setlogger
from utils.train import Train_utils
from utils.MLDG_train import MLDG_train_utils
from utils.DAEL_train import DAEL_train_utils

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # model parameters
    parser.add_argument('--model_name', type=str, default='DG_for_RMFD', help='the name of the model')

    # data parameters
    parser.add_argument('--data_name', type=str, default='PU', choices=['PHM2009', 'PU'], help='dataset')
    parser.add_argument('--transfer_task', type=list, default=[[0, 1, 2], 3], help='transfer learning tasks')
    parser.add_argument('--normalizetype', type=str, default='mean-std', choices=['-1-1', 'mean-std'])
    parser.add_argument('--in_channel', type=int, default=1, help='number of channels of input')
    parser.add_argument('--num_train_samples', type=int, default=200, help='number of training samples')
    parser.add_argument('--num_test_samples', type=int, default=50, help='number of test samples')

    # training parameters
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument('--batch_size', type=int, default=32, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--method', type=str, default='AGG', choices=['AGG', 'DANN', 'MMD', 'CORAL',
                                                                               'Triplet_loss', 'Mixup', 'DAEL', 'MLDG'])

    # optimization information
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--inner_lr', type=float, default=1e-3, help='inner learning rate for meta-learning')
    parser.add_argument('--trade_off', type=float, default=0.01, help='trade-off parameter')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='20, 40', help='the learning rate decay for step and stepLR')
    parser.add_argument('--epoch', type=int, default=60, help='max number of epoch')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    if args.method == 'AGG' or args.method == 'CORAL' or args.method == 'MMD' \
            or args.method == 'DANN' or args.method == 'Triplet_loss' or args.method == 'Mixup':
        trainer = Train_utils(args, save_dir)

    elif args.method == 'MLDG':
        trainer = MLDG_train_utils(args, save_dir)

    elif args.method == 'DAEL':
        trainer = DAEL_train_utils(args, save_dir)

    trainer.setup()
    trainer.train()

