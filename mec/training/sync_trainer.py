# sync_trainer.py
# created: CS
# 多进程多卡同步训练模块封装



import json
import os
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from .basic_trainer import BasicTrainer
from ..comms.sync_rpc import SyncRpcWorker, SyncRpcController
from ..comms.transmit import DistEnv, TensorTransmittor
from ..utils.logs import Logger
from ..utils.monitor import Monitor
from ..utils.history import History
from ..configs.arguments import *

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
                 rank, gpu_id, sync_worker_num, control_ip='127.0.0.1',
                 port=12500):
        self.printToLog = Logger(
            filepath ='logs/worker_{}.log'.format(rank),
            prefix   ='worker_{}|gpu_{}'.format(rank, gpu_id)
        )
        rpc_port  = port
        dist_port = str(port+10)
        self.printToLog("initiating worker {} ...".format(rank))
        self.dataset_dict = dataset_dict
        self.batch_transform_dict = batch_transform_dict
        self.sync_worker_num = sync_worker_num
        self.printToLog("worker num :", sync_worker_num)
        self.world_size      = sync_worker_num
        self.printToLog('initiating data loader ...')
        # for dataset_name in dataset_dict:
        #     print(
        #         "\ndataset_name:", dataset_name, dataset_dict[dataset_name],
        #         "\nbatch_size:",  int(batch_size/self.world_size),
        #         "\nsampler:",     DistributedSampler(dataset_dict[dataset_name], sync_worker_num, rank),
        #         "\nnum_workers:", process_num_per_loader,
        #     )
        self.dataloader_dict = {
            dataset_name: DataLoader(
                dataset_dict[dataset_name],
                batch_size  = int(batch_size/self.world_size),
                sampler     = DistributedSampler(dataset_dict[dataset_name], sync_worker_num, rank),
                num_workers = process_num_per_loader,
                pin_memory  = True
            )
            for dataset_name in dataset_dict
        }
        #if self.pid != os.getpid(): return # sub processes forked by Dataloader
        self.printToLog("initiating device")
        self.device          = torch.device('cuda:{}'.format(gpu_id))
        self.printToLog("device:", self.device)
        self.rank            = rank
        self.printToLog("rank:", rank)
        self.printToLog("initiating model")
        self.model           = model #.to(self.device)
        self.model.to(self.device)
        self.printToLog("initiating trainer")
        self.trainer         = BasicTrainer(model, optimizer, criterion, metrics)
        self.printToLog("initiating torch.distributed environment")        
        self.env             = DistEnv(rank, self.world_size, control_ip, dist_port, self.printToLog)
        self.default_group   = self.env.newGroup(range(self.world_size))
        self.printToLog("initiating transmittor")
        self.transmittor     = TensorTransmittor(list(range(self.world_size)) , logger=self.printToLog)
        self.printToLog("initialing rpc proxy")
        self.rpc_server = SyncRpcWorker(control_ip, rpc_port, rank, self.printToLog)
        self.rpc_server.registerMethod(self.averagingGrads)
        self.rpc_server.registerMethod(self.averagingWeights)
        self.rpc_server.registerMethod(self.broadCastModelWeights)
        self.rpc_server.registerMethod(self.gatherAveragedModelWeights)
        # ------------ training methods ------------
        self.rpc_server.registerMethod(self.initTrainEpoch)
        self.rpc_server.registerMethod(self.batchTrainNoUpdate)
        self.rpc_server.registerMethod(self.updateWeights) 
        # ------------ validation methods ------------
        self.rpc_server.registerMethod(self.initTestEpoch)
        self.rpc_server.registerMethod(self.batchTest)
        # ------------ saving methods ------------
        self.rpc_server.registerMethod(self.saveModelWeights)
        self.rpc_server.registerMethod(self.loadModelWeights)
        # init rpc proxy last, after all preparations are ready
        self.printToLog('workers ready') 
    
    def mainLoop(self):
        self.rpc_server.mainLoop()

    # def _returnScore(self, flag, batch, sample_num, loss, met):
    #     respond = {
    #         'flag'   : flag,
    #         'batch'  : batch,
    #         'samples': sample_num,
    #         'loss'   : loss,
    #         'met'    : met
    #     }
    #     self.printToLog(repr(respond)[1:-2])
    #     self.rpc_server.reportMessage(respond)
        
    # data communicating --------------------------
    def averagingWeights(self, style='full'):
        """
            各个进程的模型参数取平均
        """
        self.printToLog("averaging weights")
        self.transmittor.crossTensors(self.trainer.model, style=style)
    
    def averagingGrads(self, style='full'):
        """
            各个进程的模型梯度取平均
        """
        self.printToLog("averaging grads")
        self.transmittor.crossGrads(self.trainer.model, style=style)
    
    def gatherAveragedModelWeights(self, rank, group=None):
        """
            将所有进程的权重集中求平均
            结果保存至一个进程
            用于初始化时同步
        """
        self.transmittor.meanGatherTensors(self.trainer.model, rank, group)
    
    def broadCastModelWeights(self, rank, group=None):
        """
            从一个进程向其他进程广播模型权重
        """
        self.transmittor.broadcastTensors(self.trainer.model, rank, group)
        
    # training methods ----------------------------
    def initTrainEpoch(self, dataset_name, epoch, lr):
        self.printToLog("initializing training epoch {}".format(epoch))
        self.printToLog("learning rate: {}".format(lr))
        self.trainer.initEpoch()
        self.trainer.setLearningRate(lr)
        self.trainer.model.train()
        self.train_batch_index = 0
        self.printToLog("initializing train loader iter")
        self.printToLog('dataloader length: ', len(self.dataloader_dict[dataset_name]) )
        self.printToLog('dataloader dict: ', self.dataloader_dict)
        train_loader = self.dataloader_dict[dataset_name]
        self.printToLog('dataloader: ', train_loader)
        try:
            self.train_iter = iter(train_loader)
        except Exception as e:
            print(e)
        self.printToLog('dataloader iter', self.train_iter)
        if dataset_name in self.batch_transform_dict:
            self.train_batch_transform = self.batch_transform_dict[dataset_name]
            self.printToLog("setting up batch transforms")
        else:
            self.train_batch_transform = {lambda x: x}
            self.printToLog("no batch transforms")
        self.printToLog("epoch {}, begin training".format(epoch))
        
    def batchTrainNoUpdate(self):
        self.printToLog("train batch {}".format(self.train_batch_index) )
        data, target = next(self.train_iter)
        batch_sample_num = len(target)
        self.printToLog("getting data")
        data, target = data.to(self.device), target.to(self.device)
        if self.train_batch_transform is not None:
            print(self.train_batch_transform)
            data, target = self.train_batch_transform(data, target)
        self.printToLog("forwarding")
        self.trainer.forwardData(data)
        self.printToLog("backwarding")
        self.trainer.backwardGrad(target)
        self.train_batch_index += 1
        loss, met = self.trainer.getScores()
        return batch_sample_num, loss, met
        
    def updateWeights(self):
        self.printToLog("updating weights")
        self.trainer.updateWeights()
        
    # validation methods ----------------------------------
    def initTestEpoch(self, dataset_name, epoch):
        self.printToLog("initizating validation epoch {}".format(epoch))
        valid_loader = self.dataloader_dict[dataset_name]
        valid_loader.sampler.set_epoch(epoch)
        self.valid_iter = iter(valid_loader)
        self.trainer.model.eval()
        self.valid_batch_index = 0
        if dataset_name in self.batch_transform_dict:
            batch_transform = self.batch_transform_dict[dataset_name]
            self.printToLog("setting up batch transforms")
        else:
            batch_transform = {lambda x: x}
            self.printToLog("no batch transforms")
        self.printToLog("epoch {}, begin validation".format(epoch))
        
        # for batch, (data, target) in enumerate(valid_loader):
        #     data, target = data.to(self.device), target.to(self.device)
        #     data, target = batch_transform(data, target)
        #     self.trainer.forwardNoGrad(data)
        #     self.trainer.calcScores(target)
        #     self._returnScore('valid_batch', batch, len(data), *self.trainer.getScores())
            
    def batchTest(self):
        self.printToLog("validation batch {}".format(self.valid_batch_index))
        data, target = next(self.valid_iter)
        batch_sample_num = len(target)
        self.printToLog("getting data")
        data, target = data.to(self.device), target.to(self.device)
        if self.train_batch_transform is not None:
            data, target = self.train_batch_transform(data, target)
        self.printToLog("forwarding")
        self.trainer.forwardData(data)
        self.printToLog("backwarding")
        self.trainer.backwardGrad(target)
        self.valid_batch_index += 1
        loss, met = self.trainer.getScores()
        return batch_sample_num, loss, met
     
    # ------------ saving methods ------------
    def saveModelWeights(self, filename, rank=0, rank_list=[-1]):
        """
            指定位置保存当前模型权重
        """
        if self.rank == rank or self.rank in rank_list:
            self.trainer.saveModel(filename)
    
    def loadModelWeights(self, filename, rank=0, rank_list=[-1]):
        """
            从指定位置读取模型权重
        """
        self.printToLog("filename={}, rank={}".format(filename, rank) )
        if self.rank==rank or self.rank in rank_list:
            if not os.path.exists(filename): 
                self.printToLog('warning: no model weight file found')
                return 
            try:
                self.trainer.loadModel(filename, map_location=self.device)
            except Exception as e:
                self.printToLog(e)

    
class ControllerSync():
    """

    """
    def __init__(self, 
            train_set_len, valid_set_len, batch_size, 
            #init_epoch, total_epochs, metric_name,
            #lr_scheduler, 
            control_ip, port, sync_worker_num#,
            #current_model_filename, best_model_filename, history_filename
        ):
        rpc_port  = port
        dist_port = str(port+10)
        # 日志
        self.printToLog = Logger(
            filepath ='logs/controller.log',
            prefix   ='controller'
        )
        # 控制接口
        self.rpcWorkers   = SyncRpcController(control_ip, port, sync_worker_num, self.printToLog)
        #self.rpcWorkers.startWorking()
        # 数据
        self.train_set_len    = train_set_len
        self.valid_set_len    = valid_set_len
        self.train_loader_len = int(np.ceil(train_set_len/batch_size) )
        self.valid_loader_len = int(np.ceil(valid_set_len/batch_size) )
        # 训练
        #self.lr_scheduler = lr_scheduler
        #self.metric_name  = metric_name
        #self.monitor      = Monitor(init_epoch, total_epochs, self.train_loader_len, self.valid_loader_len, metric_name)
        # 评价
        self.best_epoch_loss = 1e6
        self.best_epoch_met  = 0.
    
    def __del__(self):
        #self.rpcWorkers.stopLooping()
        pass
    
    # def averagingGrads(self, style='full'):
    #     self.rpcWorkers.averagingGrads(style='full')

    # def averagingWeights(self, style='full'):
    #     self.rpcWorkers.averagingWeights(style='full')

    # def broadCastModelWeights(self, rank, group=None):
    #     self.rpcWorkers.broadCastModelWeights(rank, group)

    # def gatherAveragedModelWeights(self, rank, group=None):
    #     self.rpcWorkers.gatherAveragedModelWeights(rank, group=None)

    # def initTrainEpoch(self, dataset_name, epoch, lr):
    #     #self.printToLog("initiating training epoch {}, lr={}".format(epoch, lr))
    #     self.rpcWorkers.initTrainEpoch(dataset_name, epoch, lr)

    # def batchTrainNoUpdate(self, batch_index):
    #     #self.printToLog("training batch {}".format(batch_index))
    #     return self.rpcWorkers.batchTrainNoUpdate()

    # def updateWeights(self):
    #     self.rpcWorkers.updateWeights()

    # def initTestEpoch(self, dataset_name, epoch):
    #     #self.printToLog("initiating validation epoch {}".format(epoch))
    #     self.rpcWorkers.initTestEpoch(dataset_name, epoch)

    # def batchTest(self):
    #     #self.printToLog("validation batch {}".format(batch_index))
    #     return self.rpcWorkers.batchTest()

    # def saveModelWeights(self, filename, rank=1, rank_list=[-1]):
    #     self.rpcWorkers.saveModelWeights(filename, rank=1, rank_list=[-1])

    # def loadModelWeights(self, filename, rank=1, rank_list=[-1]):
    #     self.rpcWorkers.loadModelWeights(filename, rank=1, rank_list=[-1])
        
    def startWorking(self):
        self.rpcWorkers.startWorking()
    
    def stopWorking(self):
        self.rpcWorkers.stopWorking()
        
    def stopLooping(self):
        self.rpcWorkers.stopLooping()

    def trainEpoch(self, epoch, lr, dataset_name='train', style='full', monitor=None):
        #lr = self.lr_scheduler(epoch)
        self.printToLog("initiating training epoch {}; lr={}".format(epoch, lr) )
        self.rpcWorkers.initTrainEpoch(dataset_name, epoch, lr)
        total_sample_num = 0
        total_loss       = 0
        total_met        = 0
        total_avg_loss   = 0
        total_avg_met    = 0
        for batch_index in range(self.train_loader_len):
            self.printToLog("training batch {}".format(batch_index))
            result_list = self.rpcWorkers.batchTrainNoUpdate()
            #self.averagingGrads()
            self.rpcWorkers.updateWeights()
            self.rpcWorkers.averagingWeights(style=style)
            batch_sample_num = 0
            batch_total_loss = 0
            batch_total_met  = 0
            for sample_num, loss, met in result_list:
                self.printToLog('single_result: loss={}, score={}'.format(loss, met) )
                batch_sample_num += sample_num
                batch_total_loss += loss * sample_num
                batch_total_met  += met  * sample_num
            total_sample_num += batch_sample_num
            total_loss       += batch_total_loss
            total_met        += batch_total_met
            batch_loss = batch_total_loss / batch_sample_num
            batch_met  = batch_total_met  / batch_sample_num
            total_avg_loss = total_loss/total_sample_num
            total_avg_met  = total_met /total_sample_num
            if monitor!=None:
                monitor.updateTraining(batch_loss, total_avg_loss, batch_met, total_avg_met)
        return total_avg_loss, total_avg_met
    
    def testEpoch(self, epoch, dataset_name='valid', monitor=None):
        self.printToLog("initiating test epoch {}".format(epoch))
        self.rpcWorkers.initTestEpoch(dataset_name, epoch)
        total_val_sample_num = 0
        total_val_loss       = 0
        total_val_met        = 0
        total_avg_val_loss       = 0
        total_avg_val_met        = 0
        for val_batch_index in range(self.valid_loader_len):
            self.printToLog("test batch {}".format(val_batch_index))
            result_list = self.rpcWorkers.batchTest()
            batch_val_sample_num = 0
            batch_val_total_loss = 0
            batch_val_total_met  = 0
            for val_sample_num, val_loss, val_met in result_list:
                self.printToLog('single_result: loss={}, score={}'.format(val_loss, val_met) )
                batch_val_sample_num += val_sample_num
                batch_val_total_loss += val_loss * val_sample_num
                batch_val_total_met  += val_met  * val_sample_num
            total_val_sample_num += batch_val_sample_num
            total_val_loss       += batch_val_total_loss
            total_val_met        += batch_val_total_met
            batch_val_loss = batch_val_total_loss / batch_val_sample_num
            batch_val_met  = batch_val_total_met  / batch_val_sample_num
            total_avg_val_loss = total_val_loss/total_val_sample_num
            total_avg_val_met  = total_val_met /total_val_sample_num
            if monitor!=None:
                monitor.updateValidation(batch_val_loss, total_avg_val_loss, batch_val_met, total_avg_val_met)
        if total_val_met>self.best_epoch_met or self.best_epoch_met==None:
            self.is_best_epoch   = True
            self.best_epoch_loss = total_val_loss
            self.best_epoch_met  = total_val_met
        return total_avg_val_loss, total_avg_val_met
    
    # def endEpoch(self):
    #     self.monitor.updateEpoch()
    
    def saveModelWeights(self, filename, rank=0):
        self.printToLog('saving model')
        self.rpcWorkers.saveModelWeights(filename, rank)
    
    def loadModelWeight(self, filename, rank=0):
        self.rpcWorkers.loadModelWeights(filename, rank)
        self.rpcWorkers.broadCastModelWeights(rank)
    
# ========================== # ==========================
def startWorkerProcess(
        model, optimizer, criterion, metrics,
        dataset_dict, batch_transform_dict,
        batch_size, process_num_per_loader,
        rank, gpu_id, 
        sync_worker_num, 
        control_ip, 
        port
        ):
    torch.cuda.set_device( gpu_id )
    os.environ['CUDA_VISIBLE_DEVICE']='{}'.format(gpu_id)
    #time.sleep(3)
    worker = WorkerSync(
        model, optimizer, criterion, metrics, 
        dataset_dict, batch_transform_dict,
        batch_size, process_num_per_loader, rank, gpu_id,
        sync_worker_num, control_ip,
        port)
    worker.mainLoop()

def startWorkers(
        model, optimizer, criterion, metrics, 
        train_set, valid_set, 
        batch_size=batch_size, 
        sync_worker_num=1, 
        process_num_per_loader=process_num_per_loader,
        rank_list=worker_ranks, 
        gpu_id_list=worker_gpu_ids, 
        control_ip=control_ip, 
        port=basic_port, 
        train_batch_transform=None, 
        valid_batch_transform=None
    ):
    assert len(rank_list)==len(gpu_id_list), 'rank_list has different length from gpu_id_list'
    assert min(rank_list)>=0,                'rank must be greater than 0'
    assert max(rank_list)<sync_worker_num,   'rank exceed limits'
    dataset_dict         = {'train':train_set, 'valid':valid_set}
    batch_transform_dict = {'train':train_batch_transform, 'valid':valid_batch_transform}
    worker_pool = []
    for rank, gpu_id in zip(rank_list, gpu_id_list):
        args = (
            model, optimizer, criterion, metrics, 
            dataset_dict, batch_transform_dict,
            batch_size, process_num_per_loader,
            rank, gpu_id, sync_worker_num, control_ip, port
        ) 
        worker_process = mp.Process(target=startWorkerProcess, args=args)
        worker_pool.append(worker_process)
        worker_process.start()

# ==========================  ========================== 
def trainAndVal(
            train_set, valid_set, metrics, 
            batch_size, lr_scheduler, 
            control_ip=control_ip, 
            port=basic_port, 
            sync_worker_num=sync_worker_num,
            total_epochs=epochs, 
            current_model_filename=model_filename, 
            best_model_filename=best_model_filename,
            history_filename=history_filename, 
            continue_training=continue_training):
    controller = ControllerSync(
        len(train_set), len(valid_set), batch_size,
        control_ip, port, sync_worker_num,
    )
    history = History(history_filename)
    if continue_training: history.loadHistory()
    init_epoch = history.data['epochs']
    monitor = Monitor(
        init_epoch, 
        total_epochs, 
        int(np.ceil(len(train_set)/batch_size) ), 
        int(np.ceil(len(valid_set)/batch_size) ), 
        metrics.name()
    )
    controller.printToLog('waking workers')
    controller.startWorking()
    if continue_training:
        controller.loadModelWeight(current_model_filename)
    controller.printToLog('prepare to train')
    for epoch in range(init_epoch, init_epoch+total_epochs):
        lr = lr_scheduler(epoch)
        monitor.beginEpoch()
        loss, met     = controller.trainEpoch(epoch, lr, dataset_name='train', style='full', monitor=monitor)
        v_loss, v_met = controller.testEpoch(epoch, 'valid', monitor)
        controller.saveModelWeights(current_model_filename)
        if controller.is_best_epoch:
            controller.saveModelWeights(best_model_filename)
        monitor.endEpoch()
        history.updateHistory(loss, met, v_loss, v_met, lr)
        history.saveHistory()
    controller.stopLooping()
    
def trainAndValLocal(
        model, optimizer, criterion, metrics, 
        train_set, valid_set, 
        batch_size=batch_size, 
        lr_scheduler = lambda: learning_rate,
        process_num_per_loader=process_num_per_loader,
        rank_list=worker_ranks, 
        gpu_id_list=worker_gpu_ids, 
        control_ip=control_ip, 
        port=basic_port, 
        train_batch_transform=None, 
        valid_batch_transform=None
    ):
    sync_worker_num=len(gpu_id_list)
    startWorkers(
        model, optimizer, criterion, metrics, train_set, valid_set, 
        batch_size, sync_worker_num, process_num_per_loader,
        rank_list, gpu_id_list, control_ip, port, 
        train_batch_transform, valid_batch_transform
    )    
    trainAndVal(
            train_set, valid_set, metrics, 
            batch_size, lr_scheduler, 
            control_ip=control_ip, 
            port=port, 
            sync_worker_num=sync_worker_num
        )
    pass