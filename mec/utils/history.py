import json


class History:
    def __init__(self, filename, logger=print):
        self.filename = filename
        self.printToLog = logger
        self.data = {
            'epochs':0, 
            'train_loss':[], 
            'train_met':[], 
            'val_loss':[], 
            'val_met':[], 
            'lr':[]
        }
        
    def updateHistory(self, train_loss, train_met, val_loss, val_met, lr):
        self.data['epochs'] += 1
        self.data['train_loss'].append( train_loss )        
        self.data['train_met'].append(  train_met  )       
        self.data['val_loss'].append(   val_loss   )      
        self.data['val_met'].append(    val_met    )     
        self.data['lr'].append(         lr         )

    def loadHistory(self):
        try:
            with open(self.filename) as file:
                self.data = json.load(file)
                print('loading successfule')
        except FileNotFoundError as e:
            self.printToLog('warining: training history record file not found')
        
    def saveHistory(self):
        with open(self.filename, 'w') as file:
            json.dump(self.data, file, indent=True, ensure_ascii=False)