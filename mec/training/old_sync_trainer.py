import os
import json
import time
import zmq
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .monitor import Monitor


from torchvision.models.resnet import resnet50
import torch.nn as nn
import torch.optim as optim


#from dataset_single import get_data_loader

# --------------------------------- traning utils -------------------------------
# -------------------------------------------------------------------------


# Trainer类
# 封装训练中的各个步骤
# 不负责流程组织
# 不负责数据传递
class Trainer():
    def __init__(self, model, optimizer, criterion, metrics):
        # ----- basic elements -----
        self.model     = model     # 模型
        self.optimizer = optimizer # 优化器
        self.criterion = criterion # 损失函数
        self.metrics   = metrics   # 评分函数
        # ----- temporary figures -----
        self.loss   = 0
        self.met    = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def forwardData(self, data):
        self.optimizer.zero_grad()
        self.result = self.model(data)

    # 增量式前向：不归零梯度
    def forwardDataInc(self, data):
        self.result=self.model(data)

    # 前向但不记录梯度
    @torch.no_grad()
    def forwardNoGrad(self, data):
        self.result=self.model(data)

    def backwardGrad(self, target):
        self.loss = self.criterion(self.result, target)
        self.loss.backward()
        self.met, _ = self.metrics(self.result, target)

    # 只计算loss，不回传梯度
    @torch.no_grad()
    def calcScores(self, target):
        self.loss = self.criterion(self.result, target)
        self.met, _ = self.metrics(self.result, target)

    def setLearningRate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def updateWeights(self):
        self.optimizer.step()

    def getScores(self):
        return self.loss.item(), self.met

# Workers 类
# 负责数据传递
# 负责基本任务承接
# class WorkerBase():
#     pass

#
class WorkerSync():
    r"""
    同步训练worker类
    每次接受manager广播来分配batch
    forward和backward结束后
    用dist.reduce上传grad
    用dist.broadcast回传weights
    grad在manager上延迟更新

        参数:
        model           : 数据模型
        dataset         : 数据集，应可索引
        device          : 使用的设备
        rank            : 进程编号，在组内应唯一
        sync_worker_num : worker数量
        manager_ip      : 仅IP地址，如"127.0.0.1"
        task_port       : 发布使用的端口
        score_port      : 返回分数使用的端口
        dist_port       : torch.dist使用的端口
    """
    def __init__(self, trainer, train_loader, valid_loader,
                 device, rank, sync_worker_num, num_workers_per_loader, manager_ip,
                 train_batch_transform=None, valid_batch_transform=None,
                 dist_port="8100", task_port="8101", score_port="8102",
                 manager_with_gpu=True):
        # 训练相关
        self.trainer              = trainer
        self.device               = device
        self.train_loader         = train_loader
        self.valid_loader         = valid_loader
        self.train_batch_transform = train_batch_transform
        self.valid_batch_transform = valid_batch_transform
        self.train_iter           = None
        self.valid_iter           = None
        # 分布式相关
        os.environ['MASTER_ADDR'] = manager_ip
        os.environ['MASTER_PORT'] = dist_port
        self.dist_addr            = "tcp://" + manager_ip + ":" + dist_port
        self.rank                 = rank
        self.sync_worker_num      = sync_worker_num
        self.manager_with_gpu     = manager_with_gpu
        self.world_size           = sync_worker_num + 1
        # 通讯相关
        self.context              = None
        self.task_addr            = "tcp://" + manager_ip + ":" + task_port
        self.task_socket          = None
        self.score_addr           = "tcp://" + manager_ip + ":" + score_port
        self.score_socket         = None
        # 日志文件
        self.logfile = open("logs/worker_{:d}.log".format(rank), 'w')
        # try:
            # self.logfile = open("logs/worker_{:d}.log".format(rank), 'a')
        # except FileNotFoundError:
            # self.logfile = open("logs/worker_{:d}.log".format(rank), 'w')

    def __del__(self):
        self.closeSocket()
        self.logfile.close()

    def printToLog(self, *content):
        print("[worker_{:d}|{}]".format(self.rank, time.strftime("%y-%m-%d_%H:%M:%S")),
              *content, file=self.logfile, flush=True)
        #print("[manager]", *content)

    def initSocket(self):
        self.printToLog("initilzating socket:")
        self.printToLog("task  addr = '{}'".format(self.task_addr) )
        self.printToLog("score addr = '{}'".format(self.score_addr) )
        self.context       = zmq.Context()
        self.task_socket   = self.context.socket(zmq.SUB)
        self.task_socket.connect(self.task_addr)
        self.task_socket.setsockopt(zmq.SUBSCRIBE, b'')
        self.score_socket  = self.context.socket(zmq.PUSH)
        self.score_socket.connect(self.score_addr)

    def closeSocket(self):
        if self.task_socket != None:
            self.task_socket.disconnect(self.task_addr)
            self.task_socket = None
        if self.score_socket != None:
            self.score_socket.disconnect(self.score_addr)
            self.score_socket = None

    def recvMessage(self):
        return eval(self.task_socket.recv().decode() )

    def sendMessage(self, msg):
        return self.score_socket.send(repr(msg).encode() )

    def initTorchDist(self):
        self.printToLog("dist args:", 'nccl', self.dist_addr,
            self.rank, self.world_size)
        dist.init_process_group('nccl',
            rank=self.rank, world_size=self.world_size)
        # dist.init_process_group('nccl', init_method=self.dist_addr,
        #     rank=self.rank, world_size=self.world_size)
        self.worker_group = dist.new_group(list(range(1,self.world_size)) )

    # def setBatch(self, batch_index_list):
        # self.dataloader.batch_sampler.setBatch(batch_index_list)

    # 开始一个epoch
    def initTrainEpoch(self, epoch, lr):
        self.printToLog("setting training epoch")
        self.train_loader.sampler.set_epoch(epoch)
        self.printToLog("setting training iter")
        # if self.train_iter != None:
        #     while True:
        #         try :
        #             next(self.train_iter)
        #         except StopIteration:
        #             break
        self.train_iter = iter(self.train_loader)
        self.printToLog("setting train iter complete")
        self.trainer.setLearningRate(lr)
        self.trainer.model.train()

    def initValidEpoch(self, epoch):
        self.printToLog("setting validation epoch")
        self.valid_loader.sampler.set_epoch(epoch)
        self.printToLog("setting validation iter")
        if self.valid_iter != None:
            while True:
                try :
                    next(self.valid_iter)
                except StopIteration:
                    break
        self.valid_iter = iter(self.valid_loader)
        self.trainer.model.eval()

    # 只计算grad，不包括更新weights
    def batchTrainNoUpdate(self):
        self.printToLog("batch train")
        data, target = next(self.train_iter)
        self.batch_sample_num = len(target)
        self.printToLog("getting data")
        data, target = data.to(self.device), target.to(self.device)
        if self.train_batch_transform is not None:
            data, target = self.train_batch_transform(data, target)
        self.printToLog("forwarding")
        self.trainer.forwardData(data)
        self.printToLog("backwarding")
        self.trainer.backwardGrad(target)
        self.printToLog("batch train complete")

    def updateWeights(self):
        self.trainer.updateWeights()

    # no grad
    def batchValidate(self):
        data, target = next(self.valid_iter)
        self.batch_sample_num = len(target)
        data, target = data.to(self.device), target.to(self.device)
        self.trainer.forwardNoGrad(data)
        self.trainer.calcScores(target)

    def getScores(self):
        return self.trainer.getScores()

    def saveState(self, filename):
        return torch.save(self.trainer.model.state_dict(), filename)

    # 交换梯度
    # 视情况需传入group决定交换梯度的对象
    def crossGrads(self, async_op=False):
        for p_group in self.trainer.optimizer.param_groups:
            for param in p_group['params']:
                #print(param.size())
                dist.all_reduce(param.grad, group=self.worker_group, op=dist.ReduceOp.SUM, async_op=async_op)
        if async_op: dist.barrier(group=self.worker_group)
        for p_group in self.trainer.optimizer.param_groups:
            for param in p_group['params']:
                param.grad /= self.sync_worker_num


    # 上传梯度
    # 默认rank0为管理进程
    def uploadGrads(self, async_op=False):
        for p_group in self.trainer.optimizer.param_groups:
            for param in p_group['params']:
                dist.reduce(param.grad, 0, op=dist.ReduceOp.SUM, async_op=async_op)
        if async_op: dist.barrier()

    # 下载权值
    # 默认rank0为管理进程
    def downloadWeights(self, async_op=False):
        for p_group in self.trainer.optimizer.param_groups:
            for param in p_group['params']:
                dist.broadcast(param, 0, async_op=async_op)
        if async_op: dist.barrier()

    # 上传和交换权值
    # 默认rank0为管理进程
    def exchangeGradsAndWeights(self, async_op=False):
        for p_group in self.trainer.optimizer.param_groups:
            for param in p_group['params']:
                dist.reduce(param.grad, 0, op=dist.ReduceOp.SUM, async_op=async_op)
                dist.broadcast(param.grad, 0, async_op=async_op)
        if async_op: dist.barrier()

#def gen_optimizer(model):
#    return ...

def sync_worker_process(model, optimizer, criterion, metrics, train_set, valid_set, 
                        batch_size, device, rank, sync_worker_num, num_workers_per_loader,
                        train_batch_transform, valid_batch_transform,
                        manager_ip, sync_flag='cross'):
    #print("worker process ", rank)
    world_size = sync_worker_num+1
    ###
    torch.cuda.set_device(device) 
    # 如果不设置这一条，在DataLoader里设置pin_memory=True以后，
    # 每个worker进程里的DataLoader都会占用0号GPU

    model = model.to(device)
    #opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.01, nesterov=True)
    #trainer = Trainer(model, opt, criterion, metrics)

    trainer = Trainer(model, optimizer, criterion, metrics)
    batch_size_per_worker = int(batch_size/sync_worker_num)
    train_loader = DataLoader(
        train_set,
        batch_size = batch_size_per_worker,
        sampler = DistributedSampler(
            train_set,
            num_replicas=sync_worker_num,
            rank=rank%sync_worker_num,
            #shuffle=True
            ),
        num_workers = num_workers_per_loader,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size = batch_size_per_worker,
        sampler = DistributedSampler(
            valid_set,
            num_replicas=sync_worker_num,
            rank=rank%sync_worker_num
            #shuffle=True
            ),
        num_workers = num_workers_per_loader,
        pin_memory=True
    )
    #train_loader, valid_loader, _ = get_data_loader(batch_size=64, val_split=0.2, num_workers=num_workers_per_loader)
    worker  = WorkerSync(trainer, train_loader, valid_loader,
                         device, rank, sync_worker_num,
                         num_workers_per_loader, manager_ip,
                         train_batch_transform=train_batch_transform)
    worker.printToLog("train loader len {}".format(len(train_loader)) )
    worker.initSocket()
    worker.initTorchDist()
    #worker.sendMessage({'flag':'ready'})
    worker.printToLog("beginning loop")
    while True:
        msg = worker.recvMessage()
        worker.printToLog("message recieved:")
        worker.printToLog(msg)
        if msg['flag'] == 'init':
            sync_flag = msg['sync_flag']
        if msg['flag'] == 'quit':
            break
        elif msg['flag'] == 'train_epoch':
            epoch = msg['epoch']
            lr = msg['lr']
            worker.printToLog(("===== train epoch {}; "
                               "train loader len : {} ===== "
                              ).format(epoch, len(train_loader))
                             )
            worker.initTrainEpoch(epoch, lr)
            for train_batch_index in range(len(worker.train_loader) ):
                worker.batchTrainNoUpdate()
                if sync_flag=='cross':
                    worker.printToLog("cross grads")
                    worker.crossGrads(async_op=True)
                    worker.printToLog("updating weights")
                    worker.updateWeights()
                else :
                    worker.printToLog("exchanging")
                    #worker.exchangeGradsAndWeights(async_op=True)
                    worker.printToLog("uploading grads")
                    worker.uploadGrads()
                    worker.printToLog("downloading weights")
                    worker.downloadWeights()
                #print("getting scores")
                sample_num = worker.batch_sample_num
                loss, met  = worker.getScores()
                respond = {
                    'flag'   : 'train_score',
                    'samples': sample_num,
                    'loss'   : loss,
                    'met'    : met
                }
                worker.sendMessage(respond)
                worker.printToLog(
                    ("batch {:d}; "
                     "sample num : {}; "
                     "loss : {:.4f}; "
                     "{} : {:.4f}").format(
                        train_batch_index, sample_num, loss, metrics.getMetricName(), met
                     )
                )
        elif msg['flag'] == 'valid_epoch':
            #worker.printToLog("valid batch")
            worker.initValidEpoch(epoch)
            epoch = msg['epoch']
            for valid_batch_index in range(len(worker.valid_loader) ):
                worker.batchValidate()
                loss, met = worker.getScores()
                sample_num = worker.batch_sample_num
                respond = {
                    'flag'   : 'valid_score',
                    'valid_samples': sample_num,
                    'valid_loss'   : loss,
                    'valid_met'    : met
                }
                worker.sendMessage(respond)
                worker.printToLog(
                    ("v batch {:d}; "
                     "v samples {}; "
                     "v loss: {:.3f}; "
                     "v {}: {:.3f}").format(
                        valid_batch_index, sample_num, loss, metrics.getMetricName(), met
                     )
                )
        elif msg['flag'] == 'save_model':
            if rank in msg['ranks_to_save']: #
                worker.printToLog('---- saving model ----')
                model_filename = msg['model_filename']
                worker.saveState(model_filename)
                if msg['is_best'] == True:
                    worker.printToLog('---- saving best model ----')
                    best_model_fname = msg['best_model_fname']
                    worker.saveState(best_model_fname)
        else:
            worker.printToLog("--- unknown message types ---")
            worker.printToLog(msg)
            continue
    worker.closeSocket()


# Manager 类
# 负责流程组织
# 用消息通信控制worker进程
# torch.distributed无法同时使用多个后端，为效率选择了nccl后，无法传递cpu数据
class ManagerSync():

    def __init__(self, trainer, train_set_len, valid_set_len, batch_size, rank, sync_worker_num,
                 sync_flag='cross', history_filename="history.json",
                 result_path="results/temp", model_filename ="current_model.pth",
                 best_model_fname="best_model.pth",  manager_ip="127.0.0.1",
                 dist_port="8100", task_port="8101", score_port="8102"):
        # 基本数据
        self.trainer          = trainer
        self.sync_worker_num  = sync_worker_num # 训练进程的个数，区别于dataloader中的num_workers
        # 数据相关
        self.train_loader_len = int(np.ceil(train_set_len/batch_size/sync_worker_num) )
        self.valid_loader_len = int(np.ceil(valid_set_len/batch_size/sync_worker_num) )
        # 结果保存
        self.model_filename   = result_path + '/' + model_filename
        self.best_model_fname = result_path + '/' + best_model_fname
        self.history_filename = result_path + '/' + history_filename
        # 分布式相关
        os.environ['MASTER_ADDR'] = manager_ip
        os.environ['MASTER_PORT'] = dist_port
        self.dist_addr        = "tcp://" + manager_ip + ":" + dist_port
        self.rank             = rank
        self.world_size       = sync_worker_num + 1
        # 通讯相关
        self.context          = None
        self.task_addr        = "tcp://" + manager_ip + ":" + task_port
        self.task_socket      = None
        self.score_addr       = "tcp://" + manager_ip + ":" + score_port
        self.score_socket     = None
        # 日志文件
        self.logfile = open("logs/manager.log", 'w')
        # try:
        #     self.logfile = open("logs/manager.log", 'a')
        # except FileNotFoundError:
        #     self.logfile = open("logs/manager.log", 'w')

    def __del__(self):
        self.closeSocket()
        self.logfile.close()

    def printToLog(self, *content):
        print("[manager|{}]".format(time.strftime("%y-%m-%d_%H:%M:%S") ),
              *content, file=self.logfile, flush=True)
        #print("[manager]", *content)

    def initSocket(self):
        self.printToLog("initizating socket:")
        self.printToLog("task  addr = '{}'".format(self.task_addr) )
        self.printToLog("score addr = '{}'".format(self.score_addr) )
        self.context       = zmq.Context()
        self.task_socket   = self.context.socket(zmq.PUB)
        self.task_socket.bind(self.task_addr)
        self.score_socket  = self.context.socket(zmq.PULL)
        self.score_socket.bind(self.score_addr)

    def closeSocket(self):
        self.printToLog("closing socket")
        if self.task_socket != None:
            self.task_socket.unbind(self.task_addr)
            self.task_socket = None
        if self.score_socket != None:
            self.score_socket.unbind(self.score_addr)
            self.score_socket = None

    def sendMessage(self, msg):
        return self.task_socket.send(repr(msg).encode() )

    def recvMessage(self):
        return eval(self.score_socket.recv().decode() )

    def initTorchDist(self):
        self.printToLog("dist args:", 'nccl', self.dist_addr,
            self.rank, self.world_size)
        dist.init_process_group('nccl',
            rank=self.rank, world_size=self.world_size)
        # dist.init_process_group('nccl', init_method=self.dist_addr,
        #     rank=self.rank, world_size=self.world_size)

    # 上传梯度
    # 默认rank0为管理进程
    def gatherGrads(self, async_op=False):
        for p_group in self.trainer.optimizer.param_groups:
            for param in p_group['params']:
                dist.reduce(param.grad, 0, op=dist.ReduceOp.SUM, async_op=async_op)
        if async_op: dist.barrier()

    # 下载权值
    # 默认rank0为管理进程
    def broadcastWeights(self, async_op=False):
        for p_group in self.trainer.optimizer.param_groups:
            for param in p_group['params']:
                dist.broadcast(param, 0, async_op=async_op)
        if async_op: dist.barrier()

    def updateWeights(self):
        self.trainer.updateWeights()

    # 上传和交换权值
    # 默认rank0为管理进程
    def exchangeGradsAndWeights(self, async_op=False):
        for p_group in self.trainer.optimizer.param_groups:
            for param in p_group['params']:
                dist.reduce(param.grad, 0, op=dist.ReduceOp.SUM, async_op=async_op)
                #dist.broadcast(param.data, 0, async_op=async_op)
                dist.broadcast(param.grad, 0, async_op=async_op)
        if async_op: dist.barrier()

# manager 进程
def sync_manager_process(
        model, optimizer, criterion, metrics,
        train_set, valid_set, batch_size,
        init_epoch, total_epochs,
        device, rank, sync_worker_num,
        manager_ip="127.0.0.1", sync_flag='cross',
        result_path="results/temp"
    ):
    train_set_len = len(train_set)
    valid_set_len = len(valid_set)
    world_size = sync_worker_num+1
    trainer = Trainer(model.to(device), optimizer, None, metrics)
    manager = ManagerSync(trainer, train_set_len, valid_set_len,
                          batch_size, rank, sync_worker_num,
                          manager_ip=manager_ip)
    manager.initSocket()
    manager.initTorchDist()
    manager.printToLog("getting data")
    tempdata = torch.randn(3,224,224)
    manager.printToLog("moving data")
    tempdata = tempdata.to(device).unsqueeze(0)
    manager.printToLog("calc result")
    tempresult = trainer.model(tempdata)
    manager.printToLog("calc loss")
    loss = torch.mean(tempresult)
    manager.printToLog("backward")
    loss.backward()
    manager.printToLog("zero_grad")
    trainer.zero_grad()
    manager.printToLog("init monitor")
    monitor = Monitor(init_epoch, total_epochs,
                      manager.train_loader_len, manager.valid_loader_len,
                      metrics.getMetricName())
    message = {
        'flag'      : 'init',
        'sync_flag' : sync_flag
    }
    manager.sendMessage(message)
    for epoch in range(init_epoch+1, init_epoch+total_epochs+1):
        manager.printToLog(
            (
                "===== train epoch {}; "
                "train loader len : {} ====="
            ).format(epoch, manager.train_loader_len)
        )
        message = {
            'flag' : 'train_epoch',
            'epoch': epoch
        }
        manager.sendMessage(message)
        # training
        total_samples = 0
        total_loss    = 0
        total_met     = 0
        for batch in range(manager.train_loader_len):
            if sync_flag == "cross":
                manager.printToLog("workers corssing grads ...")
                pass
            else :
                manager.printToLog("-- exchanging --" )
                # manager.exchangeGradsAndWeights(async_op=True)
                manager.printToLog("gathering grads")
                manager.gatherGrads()
                manager.printToLog("broadcasting weights")
                manager.broadcastWeights()
                manager.printToLog("updating weights")
                manager.updateWeights()
                trainer.zero_grad()
            sample_num = 0
            loss = 0
            met = 0
            for worker_index in range(sync_worker_num):
                respond = manager.recvMessage()
                sample_num    += respond['samples']
                loss          += respond['loss']/sync_worker_num
                total_loss    += respond['loss'] * respond['samples']
                met           += respond['met']/sync_worker_num
                total_met     += respond['met'] * respond['samples']
            total_samples += sample_num
            avg_loss       = total_loss / total_samples
            avg_met        = total_met  / total_samples
            manager.printToLog(
                ("batch {:d}; "
                 "sample num : {}; "
                 "loss : {:.3f}; "
                 "{} : {:.3f}").format(
                    batch,
                    sample_num,
                    loss,
                    metrics.getMetricName(),
                    met
                )
            )
            monitor.updateTraining(loss, avg_loss, met, avg_met)
        # validation
        manager.printToLog(
            (
                "===== valid epoch {}; "
                "train loader len : {} ====="
            ).format(epoch, manager.train_loader_len)
        )
        total_val_samples = 0
        total_val_loss    = 0
        total_val_met     = 0
        message = {
            'flag' : 'valid_epoch',
            'epoch': epoch
        }
        manager.sendMessage(message)
        for val_batch in range(manager.valid_loader_len):
            val_sample_num = 0
            val_loss = 0
            val_met = 0
            for i in range(sync_worker_num):
                respond = manager.recvMessage()
                val_sample_num    += respond['valid_samples']
                val_loss          += respond['valid_loss'] / sync_worker_num
                total_val_loss    += respond['valid_loss'] * respond['valid_samples']
                val_met           += respond['valid_met']/sync_worker_num
                total_val_met     += respond['valid_met'] * respond['valid_samples']
            total_val_samples += val_sample_num
            avg_val_loss       = total_val_loss / total_val_samples
            avg_val_met        = total_val_met  / total_val_samples
            monitor.updateValidation(val_loss, avg_val_loss, val_met, avg_val_met)
            manager.printToLog(
                ("val batch {:d}; "
                 "val sample num : {}; "
                 "val loss : {:.3f}; "
                 "val {} : {:.3f}").format(
                    val_batch,
                    val_sample_num,
                    val_loss,
                    metrics.getMetricName(),
                    val_met
                )
            )
        monitor.updateEpoch(avg_loss, avg_val_loss, avg_met, avg_val_met)
        # epoch
    # epoch for loop end
    manager.printToLog("sending quit")
    message = {
        'flag': 'quit'
    }
    manager.sendMessage(message)
    monitor.close()

# ------------------------------ no gpu manager--------------------------------

# 无GPU的Manager类
# 负责流程组织
# 负责模型存取
class ManagerSyncNoGpu():

    def __init__(self, train_set_len, valid_set_len, batch_size, sync_worker_num,
                 history=None, result_path="results/temp", model_filename ="current_model.pth",
                 best_model_fname="best_model.pth", history_filename="history.json",
                 manager_ip="127.0.0.1", dist_port="8100", task_port="8101", score_port="8102"):
        # 基本数据
        self.sync_worker_num  = sync_worker_num # 训练进程的个数，区别于dataloader中的num_workers
        # 数据相关
        self.train_loader_len = int(np.ceil(train_set_len/batch_size) )
        self.valid_loader_len = int(np.ceil(valid_set_len/batch_size) )
        # 结果保存
        self.model_filename   = result_path + '/' + model_filename
        self.best_model_fname = result_path + '/' + best_model_fname
        self.history_filename = result_path + '/' + history_filename
        if history == None:
            self.history = {'epochs':0, 'train_loss':[], 'train_met':[], 'val_loss':[], 'val_met':[], 'lr':[]}
            self.best_loss     = 0
            self.best_met      = 0
            self.best_val_loss = 1e6
            self.best_val_met  = 0
        else:
            self.history = history
        if self.history['epochs']>0:
            self.best_loss     = min(self.history['train_loss'])
            self.best_met      = max(self.history['train_met'])
            self.best_val_loss = min(self.history['val_loss'])
            self.best_val_met  = max(self.history['val_met'])
        # 通讯相关
        self.context          = None
        self.task_addr        = "tcp://" + manager_ip + ":" + task_port
        self.task_socket      = None
        self.score_addr       = "tcp://" + manager_ip + ":" + score_port
        self.score_socket     = None
        # 分布式相关
        os.environ['MASTER_ADDR'] = manager_ip
        os.environ['MASTER_PORT'] = dist_port
        self.dist_addr        = "tcp://" + manager_ip + ":" + dist_port
        self.rank             = 0
        self.world_size       = sync_worker_num + 1
        # 日志文件
        self.logfile = open("logs/manager.log", 'w')

    def __del__(self):
        self.closeSocket()
        self.logfile.close()

    def printToLog(self, *content):
        print("[manager|{}]".format(time.strftime("%y-%m-%d_%H:%M:%S") ),
              *content, file=self.logfile, flush=True)
        #print("[manager]", *content)

    def initSocket(self):
        self.printToLog("initizating socket:")
        self.printToLog("task  addr = '{}'".format(self.task_addr) )
        self.printToLog("score addr = '{}'".format(self.score_addr) )
        self.context       = zmq.Context()
        self.task_socket   = self.context.socket(zmq.PUB)
        self.task_socket.bind(self.task_addr)
        self.score_socket  = self.context.socket(zmq.PULL)
        self.score_socket.bind(self.score_addr)

    def closeSocket(self):
        self.printToLog("closing socket")
        if self.task_socket != None:
            self.task_socket.unbind(self.task_addr)
            self.task_socket = None
        if self.score_socket != None:
            self.score_socket.unbind(self.score_addr)
            self.score_socket = None

    def initTorchDist(self):
        self.printToLog("dist args:", 'nccl', self.dist_addr,
            self.rank, self.world_size)
        dist.init_process_group('nccl',
            rank=self.rank, world_size=self.world_size)

    def sendMessage(self, msg):
        return self.task_socket.send(repr(msg).encode() )

    def recvMessage(self):
        return eval(self.score_socket.recv().decode() )

    # 返回值：是否是最好结果
    def updateHistory(self, epoch, lr, loss, met, val_loss, val_met):
        self.history['epochs'] += 1
        self.history['lr']        .append(lr)
        self.history['train_loss'].append(loss)
        self.history['train_met'] .append(met)
        self.history['val_loss']  .append(val_loss)
        self.history['val_met']   .append(val_met)
        self.printToLog("saving history:")
        self.printToLog("filename = {}".format(self.history_filename))
        self.printToLog("content:")
        self.printToLog("epoch {}, lr {}".format(epoch, lr) )
        self.printToLog("lr {}, t_los {:3f}, t_met {:3f}, v_los {:3f}, v_met {:3f}".format(
            lr, loss, met, val_loss, val_met)
        )
        with open(self.history_filename, 'w', encoding='utf8') as file:
            json.dump(self.history, file, ensure_ascii=False, indent=2)
        if val_met > self.best_val_met:
            self.best_loss     = loss
            self.best_met      = met
            self.best_val_loss = val_loss
            self.best_val_met  = val_met
            self.printToLog("new best score at epoch {:d}:".format(epoch))
            self.printToLog("loss {:.3f}, met {:.3f}, v_los {:.3f}, v_met {:.3f}".format(
                loss, met, val_loss, val_met)
            )
            return True
        return False


# manager 进程
def sync_manager_process_no_gpu(
        metrics_name, train_set_len, valid_set_len, batch_size,
        init_epoch, total_epochs, lr_scheduler, sync_worker_num,
        manager_ip="127.0.0.1", result_path="results/temp",
        history=None, sync_flag='cross'
    ):
    world_size = sync_worker_num+1
    manager_no_gpu = ManagerSyncNoGpu(
        train_set_len, valid_set_len,
        batch_size, sync_worker_num,
        history=history, result_path=result_path,
        manager_ip=manager_ip)
    manager_no_gpu.initSocket()
    manager_no_gpu.printToLog("init monitor")
    monitor = Monitor(init_epoch, total_epochs,
                      manager_no_gpu.train_loader_len,
                      manager_no_gpu.valid_loader_len,
                      metrics_name)
    manager_no_gpu.initTorchDist()
    message = {
        'flag'      : 'init',
        'sync_flag' : 'cross'
    }
    manager_no_gpu.sendMessage(message)
    manager_no_gpu.printToLog("message sent:")
    manager_no_gpu.printToLog(message)
    for epoch in range(init_epoch, init_epoch+total_epochs):
        manager_no_gpu.printToLog(
            (
                "===== train epoch {}; "
                "train loader len : {} ====="
            ).format(epoch, manager_no_gpu.train_loader_len)
        )
        message = {
            'flag' : 'train_epoch',
            'epoch': epoch+1,
            'lr'   : lr_scheduler(epoch+1)
        }
        manager_no_gpu.sendMessage(message)
        manager_no_gpu.printToLog("message sent:")
        manager_no_gpu.printToLog(message)
        # training
        total_samples = 0
        total_loss    = 0
        total_met     = 0
        for batch in range(manager_no_gpu.train_loader_len):
            manager_no_gpu.printToLog("workers corssing grads ...")
            sample_num = 0
            loss = 0
            met = 0
            for worker_index in range(sync_worker_num):
                respond = manager_no_gpu.recvMessage()
                sample_num    += respond['samples']
                loss          += respond['loss']/sync_worker_num
                total_loss    += respond['loss'] * respond['samples']
                met           += respond['met']/sync_worker_num
                total_met     += respond['met'] * respond['samples']
            total_samples += sample_num
            avg_loss       = total_loss / total_samples
            avg_met        = total_met  / total_samples
            manager_no_gpu.printToLog(
                ("batch {:d}; "
                 "sample num : {}; "
                 "loss : {:.3f}; "
                 "{} : {:.3f}").format(
                    batch,
                    sample_num,
                    loss,
                    metrics_name,
                    met
                )
            )
            monitor.updateTraining(loss, avg_loss, met, avg_met)
        # validation
        manager_no_gpu.printToLog(
            (
                "===== valid epoch {}; "
                "train loader len : {} ====="
            ).format(epoch, manager_no_gpu.train_loader_len)
        )
        total_val_samples = 0
        total_val_loss    = 0
        total_val_met     = 0
        message = {
            'flag' : 'valid_epoch',
            'epoch': epoch
        }
        manager_no_gpu.sendMessage(message)
        for val_batch in range(manager_no_gpu.valid_loader_len):
            val_sample_num = 0
            val_loss = 0
            val_met = 0
            for i in range(sync_worker_num):
                respond = manager_no_gpu.recvMessage()
                val_sample_num    += respond['valid_samples']
                val_loss          += respond['valid_loss'] / sync_worker_num
                total_val_loss    += respond['valid_loss'] * respond['valid_samples']
                val_met           += respond['valid_met']/sync_worker_num
                total_val_met     += respond['valid_met'] * respond['valid_samples']
            total_val_samples += val_sample_num
            avg_val_loss       = total_val_loss / total_val_samples
            avg_val_met        = total_val_met  / total_val_samples
            monitor.updateValidation(val_loss, avg_val_loss, val_met, avg_val_met)
            manager_no_gpu.printToLog(
                ("v batch {:d}; "
                 "v sample num : {}; "
                 "v loss : {:.3f}; "
                 "v {} : {:.3f}").format(
                    val_batch,
                    val_sample_num,
                    val_loss,
                    metrics_name,
                    val_met
                )
            )
        monitor.updateEpoch(avg_loss, avg_val_loss, avg_met, avg_val_met)
        is_best_epoch = manager_no_gpu.updateHistory(
            epoch, lr_scheduler(epoch+1), avg_loss, avg_met, avg_val_loss, avg_val_met
        )
        # 发送信号保存训练结果
        # 只有在本地无GPU manager模型中才需要让worker保存模型
        # 因此模型文件名只保存在manager中，需要一并发送给worker
        message = {
            'flag' : 'save_model',
            'ranks_to_save' : [1], # 指定哪几个进程来保存模型
            'model_filename' : manager_no_gpu.model_filename,
            'is_best' : False
        }
        if is_best_epoch:
            message['best_model_fname'] = manager_no_gpu.best_model_fname
            message['is_best'] = True
        manager_no_gpu.sendMessage(message)
        # epoch
    # epoch for loop end
    manager_no_gpu.printToLog("sending quit")
    message = {
        'flag': 'quit'
    }
    manager_no_gpu.sendMessage(message)
    monitor.close()

