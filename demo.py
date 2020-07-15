#!/usr/bin/python3
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import json
import argparse

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from mec.data_manip.metrics import Accuracy
from mec.training.sync_trainer import startWorkers, trainAndValLocal

# 测试数据集
from torchvision.datasets import CIFAR10
pre_image_size = (34, 34)
image_size     = (32, 32)
data_transform = transforms.Compose([
    transforms.Resize(pre_image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=25, translate=(.2, .2) , 
        scale=(0.8, 1.2), shear=8, 
        resample=Image.BILINEAR, fillcolor=0),
    transforms.RandomCrop(image_size, padding=2, fill=(0,0,0) ),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
test_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_set = CIFAR10('downloaded_models', train=True,  transform=data_transform, download=True)
valid_set = CIFAR10('downloaded_models', train=False, transform=test_transform, download=True)

# 预训练公开模型
from torchvision.models.resnet import resnet50, resnet18


train             = False
test              = False
score             = False
prod              = False
mix               = False
deploy            = False
train_from_init   = True
batch_size        = 512
#val_split         = 0.2
learning_rate     = 1e-3
static_lr         = False
epochs            = 1
data_worker_num  = 8                       # 每个DataLoader启用的进程数
path              = 'results/temp'
history_file_name = 'history.json'
model_file_name   = 'current_model.pth'
best_model_fname  = 'best_model.pth'
excel_filename    = 'scores.xls'
control_ip        = "192.168.1.99" # manager的IP
publish_port      = '8700'
report_port       = '8701'
dist_port         = '12500'
worker_gpu_ids    = [0,1,2] # worker所使用的gpu编号 [0,1,2,3]
worker_ranks      = [0,1,2] # worker本身编号 [0,1,2,3]
sync_worker_num   = len(worker_ranks)         # 总worker数，单机的情况等于上两者的长度
norm = None

parser = argparse.ArgumentParser()
parser.add_argument('-train', '--train', action='store_true',
    help='train model')
parser.add_argument('-test', '--test', action='store_true',
    help='evaluate model on test set')
parser.add_argument('-c', '--continue_training', action='store_true',
    help='continue training from last point')
parser.add_argument('-score', '--score', action='store_true',
    help='calc precision, recall and F1, then write to an excel file')
parser.add_argument('-prod', '--prod', action='store_true',
    help='test production per image')
parser.add_argument('-mix', '--mix', action='store_true',
    help='output image mix matrix as xlsx file')
parser.add_argument('-d', '--deploy', action='store_true',
    help='generate index to wiki_idx json file')
parser.add_argument('-lr', '--learning_rate', type=float,
    help='designating statis training rate')
parser.add_argument('-e', '--epochs', type=int,
    help='how many epochs to train in this run')
parser.add_argument('-p', '--path', type=str,
    help='path to store results')

args = parser.parse_args()

if args.train:
    train=True
if args.test:
    test=True
if args.score:
    score=True
if args.prod:
    prod=True
if args.mix:
    mix=True
if args.deploy:
    deploy=True
if args.continue_training:
    train_from_init=False
if args.learning_rate:
    learning_rate = args.learning_rate
    static_lr=True
if args.epochs:
    epochs = args.epochs
if args.path:
    path = args.path
history_file_name    = path + '/' + history_file_name
model_file_name      = path + '/' + model_file_name
best_model_fname     = path + '/' + best_model_fname

# -------------------------------------------------------------------------


def main():
    # model
    class_to_idx = train_set.class_to_idx
    idx_to_class = {class_to_idx[x]: x for x in class_to_idx}
    num_classes  = len(class_to_idx)
    print("classes: ", num_classes)
    print(idx_to_class)

    model = resnet18(pretrained=True)
    model.fc = nn.Linear(512, num_classes)
    

    opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.01, nesterov=True)
    #model.getConfigParams()
    criterion = torch.nn.CrossEntropyLoss()
    metrics = Accuracy()
    lr_scheduler=lambda epoch: learning_rate

    init_epoch=0
    if train:
        startWorkers(
            model, opt, criterion, metrics, 
            train_set, valid_set, 
            batch_size, sync_worker_num, data_worker_num,
            worker_ranks, worker_gpu_ids, 
            control_ip, publish_port, report_port, dist_port
        )
        trainAndValLocal(
            train_set, valid_set, metrics, #评价函数
            batch_size, lr_scheduler, 
            control_ip, publish_port, report_port, sync_worker_num,
            data_worker_num,
            init_epoch, epochs, 
            model_file_name, best_model_fname,
            history_file_name)    
  


if __name__ == "__main__":
    main()