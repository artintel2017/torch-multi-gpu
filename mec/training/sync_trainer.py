# sync_trainer.py
# created: CS
# 多进程多卡同步训练模块封装



import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from .basic_trainer import BasicTrainer
from ..comms.sync_rpc import SyncRpcWorker, SyncRpcController
from ..comms.transmit import DistEnv, TensorTransmittor
from ..utils.logs import Logger
from ..utils.monitor import Monitor

class WorkerSync():
    """
    同步训练worker类
    每次接受manager广播来分配batch
    forward和backward结束后
    用dist.reduce上传grad
    用dist.broadcast回传weights
    grad在manager上延迟更新

        参数:
        model           : 数据模型
        dataset_dict    : 数据集表，key为name，value为单个dataset
        device          : 使用的设备
        rank            : 进程编号，在组内应唯一
        sync_worker_num : worker数量
        controll_ip     : 仅IP地址，如"127.0.0.1"
        control_port    : 发布使用的端口
        report_port     : 返回分数使用的端口
        dist_port       : torch.dist使用的端口
    """    
    def __init__(self, 
                 model, optimizer, criterion, metrics, 
                 dataset_dict, batch_transform_dict,
                 batch_size, process_num_per_loader,
                 rank, gpu_id, sync_worker_num, control_ip,
                 publish_port, report_port, dist_port):
        self.printToLog = Logger(
            filepath ='logs/worker_{}.log'.format(rank),
            prefix   ='worker_{}; gpu_{}'.format(rank, gpu_id)
        )
        self.sync_worker_num = sync_worker_num
        self.world_size      = sync_worker_num
        self.rpc_proxy       = SyncRpcWorker(control_ip, publish_port, report_port, self.world_size, self.printToLog)
        self.model           = model
        self.trainer         = BasicTrainer(model, optimizer, criterion, metrics)
        self.rank            = rank
        self.device          = torch.device('cuda:{}'.format(gpu_id))
        self.env             = DistEnv(rank, self.world_size, control_ip, dist_port, self.printToLog)
        self.default_group   = self.env.newGroup(range(0, self.world_size))
        self.transmittor     = TensorTransmittor(self.default_group , logger=self.printToLog)

        self.dataset_dict = dataset_dict
        self.batch_transform_dict = batch_transform_dict
        batch_size_per_worker = int(batch_size/self.world_size)
        self.dataloader_dict = {
            dataset_name: DataLoader(
                dataset_dict[dataset_name],
                batch_size  = batch_size_per_worker,
                sampler     = DistributedSampler(dataset_dict[dataset_name], batch_size_per_worker, rank),
                num_workers = process_num_per_loader,
                pin_memory  = True
            )
            for dataset_name in dataset_dict
        }
        self.rpc_proxy.registerMethod(self.trainEpoch)
        self.rpc_proxy.registerMethod(self.validEpoch) 
        self.rpc_proxy.registerMethod(self.)
    
    def startLoop(self):
        self.rpc_proxy.startLoop()

    def _returnScore(self, flag, batch, sample_num, loss, met):
        respond = {
            'flag'   : flag,
            'batch'  : batch,
            'samples': sample_num,
            'loss'   : loss,
            'met'    : met
        }
        self.printToLog(repr(respond)[1:-2])
        self.rpc_proxy.reportMessage(respond)
            
    def trainEpoch(self, dataset_name, epoch, lr, reduce='full'):
        self.printToLog("initizating training epoch {}".format(epoch))
        train_loader = self.dataloader_dict[dataset_name]
        train_loader.sampler.set_epoch(epoch)
        #train_iter = iter(train_loader)
        self.printToLog("learning rate: {}".format(lr))
        self.trainer.setLearningRate(lr)
        self.trainer.model.train()
        if dataset_name in self.batch_transform_dict:
            batch_transform = self.batch_transform_dict[dataset_name]
            self.printToLog("setting up batch transforms")
        else:
            batch_transform = {lambda x: x}
            self.printToLog("no batch transforms")
        self.printToLog("epoch {}, begin training".format(epoch))
        for batch, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            data, target = batch_transform(data, target)
            self.trainer.zero_grad()
            self.trainer.forwardData(data)
            self.trainer.backwardGrad(target)
            self._returnScore('train_batch', batch, len(data), *self.trainer.getScores())
    
    def validEpoch(self, dataset_name, epoch):
        self.printToLog("initizating validation epoch {}".format(epoch))
        valid_loader = self.dataloader_dict[dataset_name]
        valid_loader.sampler.set_epoch(epoch)
        self.trainer.model.eval()
        if dataset_name in self.batch_transform_dict:
            batch_transform = self.batch_transform_dict[dataset_name]
            self.printToLog("setting up batch transforms")
        else:
            batch_transform = {lambda x: x}
            self.printToLog("no batch transforms")
        self.printToLog("epoch {}, begin validation".format(epoch))
        for batch, (data, target) in enumerate(valid_loader):
            data, target = data.to(self.device), target.to(self.device)
            data, target = batch_transform(data, target)
            self.trainer.forwardNoGrad(data)
            self.trainer.calcScores(target)
            self._returnScore('valid_batch', batch, len(data), *self.trainer.getScores())
    
    def saveModelWeights(self, filename, rank=1, rank_list=[-1]):
        """
            指定位置保存当前模型权重
        """
        if self.rank == rank or self.rank in rank_list:
            self.trainer.saveModel(filename)
    
    def loadModelWeights(self, filename, rank=1, rank_list=[-1]):
        """
            从指定位置读取模型权重
        """
        if self.rank == rank or self.rank in rank_list:
            self.trainer.loadModel(filename, map_location=self.device)
    
    def crossModelWeights(self, style='full'):
        """
            各个进程的模型梯度取平均
            用于同步，最多每代进行一次
        """
        self.transmittor.crossTensors(self.trainer.model, style=style)
    
    def broadCastModelWeights(self, rank, group):
        """
            从一个进程向其他进程广播模型权重
            用于初始化时同步
            一般是第一个进程进行广播
        """
        self.transmittor.broadcastTensors(self.trainer.model, rank, )
        pass
    
class ControllerSync():
    """

    """
    def __init__(self, 
            train_set_len, valid_set_len, batch_size, sync_worker_num,
            init_epoch, total_epochs, metric_name,
            lr_scheduler, control_ip, publish_port, report_port, logger
        ):
        # 日志
        self.printToLog = Logger(
            filepath ='logs/controller.log'.format(rank),
            prefix   ='controller'
        )
        # 控制接口
        self.rpcWorkers   = SyncRpcController(control_ip, publish_port, report_port, sync_worker_num, self.printToLog)
        # 数据
        self.train_set_len    = train_set_len
        self.valid_set_len    = valid_set_len
        self.train_loader_len = int(np.ceil(train_set_len/batch_size) )
        self.valid_loader_len = int(np.ceil(valid_set_len/batch_size) )
        # 训练
        self.lr_scheduler = lr_scheduler
        self.monitor      = Monitor(init_epoch, total_epochs, train_set_len, valid_set_len, metric_name)
    
    def trainEpoch(self, epoch):
        lr = self.lr_scheduler(epoch)
        self.rpcWorkers.trainEpoch('train_set', epoch, lr, reduce='full')
        total_sample_num = 0
        total_loss       = 0
        total_met        = 0
        for i in range(self.self.train_set_len):
            single_result = self.rpcWorkers.recieveSingleMessage():
            self.printToLog('single_result: {}'.format(repr(single_result)[1:-2]))
            sample_num = single_result['sample_num']
            loss       = single_result['loss']
            met        = single_result['met']
            total_sample_num += sample_num
            total_loss       += loss
            total_met        += met
            avg_loss = total_loss/total_sample_num
            avg_met  = total_met /total_sample_num
            self.monitor.updateTraining(loss, avg_loss, met, avg_met)
    
    def validEpoch(self, epoch):
        self.rpcWorkers.trainEpoch('valid_set', epoch)
        total_val_sample_num = 0
        total_val_loss       = 0
        total_val_met        = 0
        for i in range(self.self.train_set_len):
            single_result = self.rpcWorkers.recieveSingleMessage():
            self.printToLog('single_result: {}'.format(repr(single_result)[1:-2]))
            val_sample_num = single_result['sample_num']
            val_loss       = single_result['loss']
            val_met        = single_result['met']
            total_val_sample_num += val_sample_num
            total_val_loss       += val_loss
            total_val_met        += val_met
            avg_val_loss = total_val_loss/total_val_sample_num
            avg_val_met  = total_val_met /total_val_sample_num
            self.monitor.updateValidation(val_loss, avg_val_loss, val_met, avg_val_met)
    
    def endEpoch(self):
        self.monitor.updateEpoch()
    
    def saveModel(self, filename, rank_list):
        self.rpcWorkers.saveModelWeights(filename=filename, rank_list=rank_list)
    
    def loadModel(self, filename, rank):
        self.rpcWorkers.loadModelWeights(filename=filename, rank=rank)
    
# ========================== 
def trainAndVal(
            model, optimizer, criterion, metrics, #模型、优化器、损失函数、评价函数
            train_set, valid_set, #训练集
            batch_size, lr_scheduler, 
            rank_list, gpu_list,
            train_batch_transform, 
            data_workers_per_process=num_data_workers,
            init_epoch=init_epoch, total_epochs=epochs, history=history):
    pass


def startWorker(
        model, optimizer, criterion, metrics, #模型、优化器、损失函数、评价函数
        train_set, valid_set, #训练集，验证集
        batch_size, data_workers_per_process,
        rank_list, gpu_list, 
        train_batch_transform=None, data_batch_transform=None,
        init_epoch=0, total_epochs=1):
    pass