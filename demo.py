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
from mec.data_manip.metrics import Accuracy
from mec.training.sync_trainer import startWorkers, trainAndVal, trainAndValLocal

# 演示数据
from demo_dataset import train_set, valid_set

# 预训练公开模型
from torchvision.models.resnet import resnet50, resnet18

# 运行参数
from configs import parse_configs
parse_configs()
from configs import *

#print( [(k,eval(k)) for k in dir()] )  


# 多机运行时需指定本地使用哪个网卡，否则可能因网络连接速度太慢拖累训练速度
# 单机训练时不需要此参数，默认指定本地地址127.0.0.1
# os.environ['NCCL_SOCKET_IFNAME'] = 'eno2' 
# os.environ['NCCL_SOCKET_IFNAME'] = 'eno1np0'



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
    metrics = Accuracy()
    print(metrics)
    lr_scheduler=lambda epoch: learning_rate

    if train:
        trainAndValLocal(
            model, opt, criterion, metrics, 
            train_set, valid_set, 
            batch_size, lr_scheduler, epochs,
            process_num_per_loader = process_num_per_loader,
            rank_list              = worker_ranks, 
            gpu_id_list            = worker_gpu_ids,
            control_ip             = control_ip,
            port                   = basic_port
        )
        # startWorkers(
        #     model, opt, criterion, metrics, 
        #     train_set, valid_set, 
        #     batch_size, sync_worker_num, process_num_per_loader,
        #     worker_ranks, worker_gpu_ids,
        #     control_ip=control_ip
        # )
        # trainAndVal(
        #     train_set, valid_set, metrics, 
        #     batch_size, lr_scheduler,
        #     sync_worker_num=sync_worker_num,
        #     control_ip=control_ip
        # )    
  


if __name__ == "__main__":
    main()