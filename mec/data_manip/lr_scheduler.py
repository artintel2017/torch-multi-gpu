# lr_scheduler.py
# 创建： CS
# 修改： TM
# 保存各种学习率策略
# 所有函数保持函数格式：
# function_name(epoch) -> lr

import math
import numpy as np

class CosineLR():
    def __init__(self, warm_epoch=0, lr=0.001, period=30, only_decrease=True):
        """学习率余弦更新策略，通过参数可设置warmup的代数、最大学习率、余弦更新周期、是否去除上升段。
        Args:
            warm_epoch (int): warmup的代数，epoch在此区间内线性上升至最大学习率。Default: 0.
            lr （float): 最大学习率，余弦函数最高点的纵坐标。Default: 0.001.
            period (int): 余弦更新的周期，指余弦函数从最高点到最低点需要的代数。Default: 30.
            only_decrease (bool): 若为``True``则仅保留余弦下降段，若为``False``则使用完整的余弦函数。Default: ``True``.
        
        Example:
            >>> scheduler = CosineLR(warm_epoch=5, lr=0.001, period=30, only_decrease=True)
            >>> for epoch in range(180):
            >>>    lr = scheduler(epoch)
            >>>    train(...)
        """
        self.warm_epoch = warm_epoch
        self.lr = lr
        self.period = period
        self.only_decrease = only_decrease

    def __call__(self, epoch):
        """根据输入的epoch返回当前代的学习率"""
        if epoch < self.warm_epoch:
            return (epoch % self.warm_epoch) / self.warm_epoch * self.lr
        elif self.only_decrease is True:
            return (1 + math.cos(math.pi * ((epoch - self.warm_epoch) % self.period) / self.period)) * self.lr / 2
        else:
            return (1 + math.cos(math.pi * (epoch - self.warm_epoch) / self.period)) * self.lr / 2

class ExponentialLR():
    def __init__(self, warm_epoch=0, lr=0.001, rate=0.9):
        """学习率指数下降更新策略，通过参数可设置warmup的代数、最大学习率、指数下降速率。指数下降公式为：lr*(rate^epoch)。
        Args:
            warm_epoch (int): warmup的代数，epoch在此区间内线性上升至最大学习率。Default: 0.
            lr （float): 最大学习率，指数函数下降起点的纵坐标。Default: 0.001.
            rate (float): 指数下降速率，即指数函数的底数。Default: 0.9
        
        Example:
            >>> scheduler = ExponentialLR(warm_epoch=5, lr=0.001, rate=0.9)
            >>> for epoch in range(180):
            >>>    lr = scheduler(epoch)
            >>>    train(...)
        """
        self.warm_epoch = warm_epoch
        self.lr = lr
        self.rate = rate

    def __call__(self, epoch):
        """根据输入的epoch返回当前代的学习率"""
        if epoch < self.warm_epoch:
            return (epoch % self.warm_epoch) / self.warm_epoch * self.lr
        else:
            return self.lr * np.power(self.rate , (epoch - self.warm_epoch))
