# data_trasmitting.py
# 创建：陈硕
# 创建日期：2020.06.20
# 文件描述：封装跨卡、跨机模型传递功能

import os
import torch
import torch.distributed as dist

from torch.distributed.distributed_c10d import _get_default_group


class DistEnv:
    """
        torch.distributed所使用的分布式环境
    """
    def __init__(self, rank, world_size, control_ip, dist_port, logger=print):
        self.printToLog = logger
        self.rank       = rank
        self.world_size = world_size
        self.control_ip = control_ip
        self.dist_port  = dist_port 
        self._initTorchDist()
    
    def _initTorchDist(self):
        self.printToLog("dist args:", 'nccl', self.control_ip,
            self.rank, self.world_size)
        os.environ['MASTER_ADDR'] = self.control_ip
        os.environ['MASTER_PORT'] = self.dist_port
        dist.init_process_group(
            backend='nccl',
            rank=self.rank, 
            world_size=self.world_size)
        #self.worker_group = dist.new_group(list(range(1,self.world_size)) )

    def newGroup(self, rank_list):
        """
            按照rank_list建立一个新的组
        """
        return dist.new_group(rank_list)

class TensorTransmittor:
    def __init__(self, logger=print):
        self.printToLog = logger
        
    def _optiParams(self, optimizer):
        """
            封装对应优化器的参数生成器
        """
        for p_group in optimizer.param_groups:
            for param in p_group['params']:
                yield param
    
    def _getParamGen(self, var):
        if isinstance(var, (torch.Tensor, torch.nn.Parameter) ):
            return [var]
        elif isinstance(var, torch.nn.modules.Module):
            return var.parameters()
        elif isinstance(var, torch.optim.Optimizer):
            return self._optiParams(var)
        else: 
            self.printToLog("warning: unknown param type, not one of [Tensor, Parameter, Module, Optimizer]")
            return var
    
    def crossGrads(self, params, group=None, style='full', async_op=False):
        """
            参数:
            params: Tensor、Parametor、Module、Optimizer
            style选择：
                full:    普通 all-reduce, n*(n-1)
                ring:    ring all-reduce, n-1, spread
                cube:    cube all-reduce, n-1
                partial: cube partial-reduce, log2(n)
            async_op: 是否异步操作
        """
        param_generator = self._getParamGen(params)
        if group is None:
            group = _get_default_group()
        if style=='full':
            for param in param_generator:
                dist.all_reduce(
                    param.grad, 
                    op=dist.ReduceOp.SUM, 
                    group=group, 
                    async_op=async_op)
            for param in param_generator:
                param.grad /= group.size()
        else:
            self.printToLog("warining: style {} not supported")
            return
        if async_op: dist.barrier()
    
    def crossTensors(self, params, group=None, style='full', async_op=False):
        """
            参数:
            params: Tensor、Parametor、Module、Optimizer
            style选择：
                full:    普通 all-reduce, n*(n-1)
                ring:    ring all-reduce, n-1, spread
                cube:    cube all-reduce, n-1
                partial: cube partial-reduce, log2(n)
        """
        param_generator = self._getParamGen(params)
        if group is None:
            group = _get_default_group()
        if style=='full':
            for param in param_generator:
                dist.all_reduce(
                    param, 
                    group=group, 
                    op=dist.ReduceOp.SUM, 
                    async_op=async_op)
            for param in param_generator:
                param.data /= group.size()
        else:
            self.printToLog("warining: style {} not supported")
            return
        if async_op: dist.barrier()        
        pass
        