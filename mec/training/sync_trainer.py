# sync_trainer.py
# created: CS
# 多进程多卡同步训练模块封装


from .basic_trainer import Trainer
from ..comms.sync_rpc import SyncRpcWorker, SyncRpcController
from ..utils.logs import Logger

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
        control_port    : 发布使用的端口
        report_port     : 返回分数使用的端口
        dist_port       : torch.dist使用的端口
    """    
    def __init__(self, trainer):
        self.rpc_controller = SyncRpcWorker
        self.printToLog = Logger
        pass
    
    def 