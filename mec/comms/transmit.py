# data_trasmitting.py
# 创建：陈硕
# 创建日期：2020.06.20
# 文件描述：封装跨卡、跨机模型传递功能

import os
import numpy as np
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
        self.printToLog("torch distributed environment initiated successfully")
        #self.worker_group = dist.new_group(list(range(1,self.world_size)) )

    def newGroup(self, rank_list):
        """
            按照rank_list建立一个新的组
        """
        return dist.new_group(rank_list)

# 用于cube all-reduce的分配函数
def cube_correspond(n, turn):
    return (1<<turn)^n

class TensorTransmittor:
    def __init__(self, rank_list, logger=print):
        self.printToLog     = logger
        self.rank           = dist.get_rank()
        self.rank_list      = rank_list
        self.default_group  = dist.new_group(rank_list)
        self.cube_phase     = 0
        self.def_group_size = len(rank_list)
        self.cube_dim       = int(np.log2(self.def_group_size))
        self.partial_groups = [
            dist.new_group([self.rank, cube_correspond(self.rank, turn)]) if cube_correspond(self.rank, turn)<self.def_group_size
            else dist.new_group([self.rank])
            for turn in range(self.cube_dim)
        ]
        
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
        elif isinstance(var, (list, tuple)):
            return var
        else: 
            self.printToLog("Error: unknown param type, not one of [Tensor, Parameter, Module, Optimizer]")
            raise(Exception("Error: unknown param type in TensorTransmittor._getParamGen()"))
    
    def crossGrads(self, params, group=None, style='full', async_op=False):
        """
            参数:
            params: Tensor、Parametor、Module、Optimizer
            style选择：(目前仅实现了full)
                full:    普通 all-reduce, n*(n-1)
                ring:    ring all-reduce, n-1, spread
                cube:    cube all-reduce, n-1
                partial: cube partial-reduce, log2(n)
            async_op: 是否异步操作
        """
        param_generator = self._getParamGen(params)
        if group is None:
            group = self.default_group
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
            group = self.default_group
        if style=='partial':
            group = self.partial_groups[self.cube_phase]            
            self.cube_phase = (self.cube_phase+1)%self.cube_dim
        else:
            self.printToLog("warining: style {} not supported")
        for param in param_generator:
            dist.all_reduce(
                param.data, 
                group=group, 
                op=dist.ReduceOp.SUM, 
                async_op=async_op)
            param.data /= group.size()
        #for param in param_generator:
        #    param.data /= group.size()
        if async_op: dist.barrier()        

    def broadcastTensors(self, params, src_rank, group=None, async_op=False):
        """
        从src_rank进程将params复制至其他进程
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
            group = self.default_group
        for param in param_generator:
            dist.broadcast(
                param.data, 
                src=src_rank, 
                group=group, 
                async_op=async_op)
        if async_op: dist.barrier()   

    def meanGatherTensors(self, params, dst_rank, group=None, async_op=False):
        """
        求取group中所有进程的params的平均值，结果存至dst_rank进程
            参数:
            params: Tensor、Parametor、Module、Optimizer
            group: 参与的组
        """
        param_generator = self._getParamGen(params)
        if group is None:
            group = self.default_group
        for param in param_generator:
            dist.reduce(
                param.data, 
                dst=dst_rank,
                group=group, 
                op=dist.ReduceOp.SUM,
                async_op=async_op)
        if dist.get_rank() == dst_rank:
            for param in param_generator:
                param.data /= group.size()
        if async_op: dist.barrier() 
        
        
