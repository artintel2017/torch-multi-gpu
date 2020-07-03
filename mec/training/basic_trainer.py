# basic_trainer.py
# created: CS
# 基本的训练类



class Trainer():
    """
        Trainer类
        封装训练中的各个步骤
        不负责流程组织
        不负责数据传递
        参数：
            model:     模型
            optimizer: 优化器
            criterion: 损失函数
            metrics:   评分器
    """
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