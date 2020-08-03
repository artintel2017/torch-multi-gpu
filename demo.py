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
from mec.training.sync_trainer import startWorkers, trainAndVal, trainAndValLocal

# 测试数据集
from torchvision.datasets import CIFAR10
pre_image_size = (256, 256)
image_size     = (224, 224)
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

from mec.configs.arguments import *

#print( [(k,eval(k)) for k in dir()] )  


# 多机运行时需指定本地使用哪个网卡，否则可能因网络连接速度太慢拖累训练速度
# 单机训练时不需要此参数，默认指定本地地址127.0.0.1
os.environ['NCCL_SOCKET_IFNAME'] = 'eno2' 
#os.environ['NCCL_SOCKET_IFNAME'] = 'eno1np0'

process_num_per_loader = 8                    # 每个DataLoader启用的进程数
# worker_gpu_ids         = [0,1,2,3]            # worker所使用的gpu编号
# worker_ranks           = [0,1,2,3]            # worker编号
# sync_worker_num        = 4                    # 总worker数，单机的情况等于上两者的长度

worker_gpu_ids         = [0]            # worker所使用的gpu编号
worker_ranks           = [0]            # worker编号
sync_worker_num        = 1                    # 总worker数，单机的情况等于上两者的长度
batch_size             = 256*sync_worker_num
control_ip             = "192.168.1.99"       # manager的IP，如果不设置，则默认127.0.0.1



# -------------------------------------------------------------------------


def main():
    # model
    class_to_idx = train_set.class_to_idx
    idx_to_class = {class_to_idx[x]: x for x in class_to_idx}
    num_classes  = len(class_to_idx)
    print("classes: ", num_classes)
    print(idx_to_class)

    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, num_classes)
    
    opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.01, nesterov=True)
    criterion = torch.nn.CrossEntropyLoss()
    print(Accuracy)
    metrics = Accuracy()
    print(metrics)
    lr_scheduler=lambda epoch: learning_rate

    if train:
        # trainAndValLocal(
        #     model, opt, criterion, metrics, 
        #     train_set, valid_set, 
        #     batch_size=batch_size, 
        #     lr_scheduler = lr_scheduler,
        #     process_num_per_loader=process_num_per_loader,
        #     rank_list=worker_ranks, 
        #     gpu_id_list=worker_gpu_ids
        # )
        startWorkers(
            model, opt, criterion, metrics, 
            train_set, valid_set, 
            batch_size, sync_worker_num, process_num_per_loader,
            worker_ranks, worker_gpu_ids,
            control_ip=control_ip
        )
        trainAndVal(
            train_set, valid_set, metrics, 
            batch_size, lr_scheduler,
            sync_worker_num=sync_worker_num,
            control_ip=control_ip
        )    
  


if __name__ == "__main__":
    main()