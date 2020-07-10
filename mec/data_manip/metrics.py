import torch



# metrics base “评价标准” 类
# 定义了基本的评价标准接口
# 所有的评价标准必须继承此类

class MetricBase():
    def __init__(self):
        pass
        
    def __call__(self, batch_output, batch_target):
        self.addData(batch_output, batch_target)
        return self.getBatchScore(), self.getEpochScore()
        
    def __str__(self):
        return self.getMetricName()
        
    # 可重载
    # 定义评价标准的名字
    def getMetricName(self):
        return 'met'
    
    # 应重载
    # 每个batch输入数据
    def addData(self, batch_output, batch_target):
        pass
    
    # 应重载
    # 每epoch初始重置数据
    def init(self):
        pass
        
    # 应重载
    # 计算每个batch的分数
    def getBatchScore(self):
        pass
        
    # 应重载
    # 计算每个epoch的分数
    def getEpochScore(self):
        pass
    
# 评价标准：accuracy 准确率
# 用于单标签分类
class Accuracy(MetricBase):
    def __init__(self):
        MetricBase.__init__(self)
        self.totalSamples = 0
        self.totalHits    = 0
        self.batchScore   = 0.
        self.epochScore   = 0.
        
    def getMetricName(self):
        return 'acc'    
        
    def init(self):
        self.totalSamples = 0
        self.totalHits = 0
        pass
        
    # 应重载
    # 每个batch输入数据
    def addData(self, batch_output, batch_target):
        currentCount = len(batch_output)
        if batch_target.ndimension()>1:
            batch_target = batch_target.max(dim=-1)[1]
        hits = torch.sum(batch_output.max(dim=-1)[1] == batch_target).item()
        self.batchScore = hits / currentCount
        self.totalHits += hits
        self.totalSamples += currentCount
        self.epochScore = self.totalHits / self.totalSamples
        
    # 计算每个batch的分数
    def getBatchScore(self):
        return self.batchScore
        
    # 应重载
    # 计算每个epoch的分数
    def getEpochScore(self):
        return self.epochScore

# 评价标准：accuracy 准确率
# 用于单标签分类、且标签为one hot表示时的情况
# 可适用平滑和混淆
class AccuracyOneHot(MetricBase):
    pass
    
# 评价标准：mean average precision 
# 适用于多标签分类
# 每个batch单独输入数据
# 每个epoch先清除缓存
class MeanAveragePrecision(MetricBase):
    def __init__(self, tag_dimension=1, total_sample_num=1, device=torch.device('cpu') ):
        MetricBase.__init__(self)
        self.tagDimension = tag_dimension
        self.totalSampleNum = total_sample_num
        self.currentCount = 0
        self.batchSize = 1
        self.outputs = torch.zeros( (total_sample_num, tag_dimension) ).to(device)
        self.targets = torch.zeros( (total_sample_num, tag_dimension) ).to(device)
        self.tempPrecisionArray = torch.zeros(total_sample_num).to(device)
        self.tempRecallArray    = torch.zeros(total_sample_num).to(device)
        
    def getMetricName(self):
        return 'map'
        
    def init(self):
        self.batchScore = 0.0
        self.epochScore = 0.0
        self.currentCount = 0
        pass
        
    def addData(self, batch_output, batch_target):
        assert batch_output.ndimension()==2, 'output must be 2 dimension: batch/feature'
        assert batch_target.ndimension()==2, 'target must be 2 dimension: batch/feature'
        #print(self.outputs.size())
        #print(batch_output.size())
        self.batchSize = len(batch_output)
        self.outputs[self.currentCount: self.currentCount+self.batchSize] = torch.sigmoid(batch_output)
        self.targets[self.currentCount: self.currentCount+self.batchSize] = batch_target
        self.currentCount += self.batchSize
        
    def getBatchScore(self):
        return 0.0
        #return self.batchScore
        
    def getAveragePrecision(self, index):
        outputs, indices = self.outputs[0:self.currentCount, index].sort(descending=True)
        targets = self.targets[0:self.currentCount, index][indices]
        none_zero_indices = outputs!=0
        outputs = outputs[none_zero_indices]
        targets = targets[none_zero_indices]
        total_positive = torch.sum(targets>0)
        # 按threshold从到到低的顺序分别计算precision和recall
        for i in range(self.currentCount):
            threshold = outputs[i]
            true_positive = torch.sum(targets[0:i]>0)
            predicted_positive = torch.sum(outputs>threshold)
            precision = true_positive / (predicted_positive.type(torch.double) + 1e-6)
            recall    = true_positive / (total_positive.type(torch.double) + 1e-6)
            # print("\n------ ")
            # print("true pos: {}, pred pos: {}, precision: {}".format(true_positive, predicted_positive, precision) )
            # print("true pos: {}, total pos: {}, recall: {}".format(true_positive, total_positive, recall) )
            self.tempPrecisionArray[i] = precision
            self.tempRecallArray[i]    = recall
        # print(self.tempRecallArray[0:self.currentCount])
        # print(self.tempPrecisionArray[0:self.currentCount])
        # 按recall从高到低排序，方便后面的运算
        recallArray, recallIndices = self.tempRecallArray[0:self.currentCount].sort(descending=True)
        precisionArray = self.tempPrecisionArray[recallIndices]
        # 11-点-AP
        total_precision = 0
        current_precision = 0
        #for index in range(11):
        recall_index = 10
        required_recall = index*0.1
        next_recall = (index-1)*0.1
        maxPrecision = recallArray[0]
        for index, recall in enumerate(recallArray) : # recall 从高到低
            # print("\nindex: ", index)
            while recall < next_recall: # 下一个recall点
                # print("\nrecall; ", recall)
                # print(maxPrecision)
                total_precision += maxPrecision
                recall_index -= 1
                required_recall = recall_index*0.1
                next_recall = (recall_index-1)*0.1
            precision = precisionArray[index]
            if precision > maxPrecision: 
                maxPrecision = precision 
        total_precision += maxPrecision # recall == 0 时的precision
        # print(total_precision, total_precision/11.0, index)
        return total_precision / 11.0
                
    def getEpochScore(self):
        totalAP = 0.0
        for i in range(self.tagDimension): # 对所有输出维度
            totalAP += self.getAveragePrecision(i)
        self.epochScore = totalAP / self.tagDimension
        return self.epochScore.item()
    
# mean F1 score
# 评价标准：平均F1分数
# 适用于多标签分类
class meanF1Score(MetricBase):
    def __init__(self, tag_dimension=1, threshold=0.0, device=torch.device('cuda')):
        MetricBase.__init__(self)
        self.device = device
        self.batchScore = 0.0
        self.epochScore = 0.0
        self.threshold = threshold
        self.tp = torch.zeros(tag_dimension).to(device)
        self.tn = torch.zeros(tag_dimension).to(device)
        self.fp = torch.zeros(tag_dimension).to(device)
        self.fn = torch.zeros(tag_dimension).to(device)
        
    # 可重载
    # 定义评价标准的名字
    def getMetricName(self):
        return 'mF1'
    
    # 重载
    # 每个batch输入数据
    def addData(self, batch_output, batch_target):
        guess_pos  = batch_output > self.threshold # 判断为真
        target_pos = batch_target > 0 # 实际为真
        guess_neg  = batch_output < self.threshold # 判断为假
        target_neg = batch_target < 0 # 实际为假
        # 以下为向量，维度为tag dimension
        tp = torch.sum(guess_pos*target_pos, dim=0).type(torch.float) # 猜真实真，为tp
        fp = torch.sum(guess_pos*target_neg, dim=0).type(torch.float) + 1e-6 # 猜真实假，为fp
        fn = torch.sum(guess_neg*target_pos, dim=0).type(torch.float) + 1e-6 # 猜假实真，为fp
        #tn = torch.sum(guess_neg*target_neg, dim=0) # 猜假实假，为tn
        # 计算每个batch的分数
        self.batchScore = tp *2 / (tp*2 + fp + fn)
        # 计算每个epoch的分数
        self.tp += tp
        self.fp += fp
        self.fn += fn
        # print("---- shape: ----\n", self.tp.size(), tp.size() )
        # print("---- shape: ----\n", self.fp.size(), fp.size() )
        # print("---- shape: ----\n", self.fn.size(), fn.size() )
        self.epochScore = self.tp * 2 / (self.tp*2 + self.fp + self.fn)
    
    # 重载
    # 每epoch初始重置数据
    def init(self):
        self.tp.zero_()
        self.tn.zero_()
        self.fp.zero_()
        self.fn.zero_()
        self.tp += 1e-6
        self.tn += 1e-6
        self.fp += 1e-6
        self.fn += 1e-6
        
    # 重载
    # 计算每个batch的分数
    def getBatchScore(self):
        return torch.mean(self.batchScore).item()
        
    # 重载
    # 计算每个epoch的分数
    def getEpochScore(self):
        return torch.mean(self.epochScore).item()
        
    def getEpochScore_(self):
        return self.epochScore