# 数据集相关操作
import zmq
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

class IndicedDataset(Dataset):
    def __init__(self, dataset, indices):
        assert np.max(indices)<len(dataset), 'indices value must not beyond dataset length'
        assert not np.min(indices)<0, 'indices value must not below 0'
        self.indices = indices
        self.dataset = dataset
        self.class_to_idx = dataset.class_to_idx

    def __getitem__(self, index):
        return self.dataset[self.indices[index] ]
        
    def __len__(self):
        return len(self.indices)


class DynamicBatchSampler(Sampler):
    def __init__(self):
        self.batch     = []
        #self.nextBatch = None
        
    def __len__(self):
        # This has to be a TypeError, otherwise, since this is used in
        # `len(dataloader)`, `list(dataloader)` will fail.
        # see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        raise TypeError('Cannot determine the length of a DynamicSampler')
        
    def __iter__(self):
        #yield self.getBatch()
        return self
      
    def __next__(self):
        print(" // \\\\ getting batch ...")
        if self.batch == None:
            raise StopIteration
        print(self.batch)
        return self.getBatch()
        
    def getBatch(self):
        #batch = self.batch
        #self.batch = self.nextBatch
        print("[==d sampler==] getting batch")
        return self.batch
        
    def setBatch(self, batch_index_list):
        self.batch = batch_index_list
        # self.nextBatch = batch_index_list
        # if self.batch == None :
        #     self.batch = batch_index_list

class RemoteBatchSampler(Sampler):
    
    def __init__(self, addr):
        self.addr    = addr
        self.context = zmq.Context()
        self.socket  = self.context.socket(zmq.SUB)
        self.socket.connect(addr)
        
    def __next__(self):
        pass
    
class RemoteBatchSamplerServer:
    
    def __init__(self, addr):
        pass
        
    def setEpochIndices(self, indices):
        self.indices = indices


class ResultWrapper:
    
    def __init__(self):
        pass
        
    def __run__(self, input):
        pass