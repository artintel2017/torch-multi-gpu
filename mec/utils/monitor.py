# 进度监视条

import time
from tqdm import tqdm

def reset_tqdm(pbar):
    pbar.n = 0
    pbar.last_print_n = 0
    pbar.start_t = time.time()
    pbar.last_print_t = time.time()

# 打印类，用于打印训练信息
class Monitor():
    def __init__(self, init_epoch, total_epochs, 
                 train_batch_num, val_batch_num, 
                 metric_name, bar_cols=100, show=True):
        # --------------- tast attributes ---------------
        self.init_epoch      = init_epoch
        self.total_epochs    = total_epochs
        self.train_batch_num = train_batch_num
        self.val_batch_num   = val_batch_num
        self.metric_name     = metric_name
        # --------------- status records ---------------
        self.is_closed = False
        self.current_epoch       = init_epoch
        self.current_train_batch = 0
        self.current_val_batch   = 0
        # --------------- initiation actions ---------------
        self.bar_cols = bar_cols
        self.bar0 = None
        self.bar1 = None
        self.bar2 = None
        if show: self._initBars()
        # --------------- data ---------------
        self.avg_loss     = 0
        self.avg_met      = 0
        self.avg_val_loss = 0
        self.avg_val_met  = 9

    
    def __del__(self):
        if not self.is_closed:
            self.close()
    
    def _initBars(self):
        # 监控初始化
        self.bar0 = tqdm(range(self.init_epoch, self.init_epoch+self.total_epochs), desc = '', position=0,
            bar_format='{desc}│{bar}│{elapsed}s{postfix}', ncols=self.bar_cols)
        # for i in range(self.init_epoch): 
            # self.bar0.update(self.init_epoch)
        self.bar0.set_description_str('epoch:{:4d}/{:4d}'.format(self.init_epoch, self.init_epoch+self.total_epochs))
        self.bar0.set_postfix_str(" t_los={1:.3f}, t_{0}={2:.3f}| v_los={3:.3f}, v_{0}={4:.3f}".format(
            self.metric_name, 0, 0, 0, 0))
        self.bar1 = tqdm(range(self.train_batch_num), desc="", position=1,
                bar_format='{desc}│{bar}│{elapsed}s{postfix}', ncols=self.bar_cols)
        self.bar2 = tqdm(range(self.val_batch_num), desc = '', position=2,
                bar_format='{desc}│{bar}│{elapsed}s{postfix}', ncols=self.bar_cols, leave=True)
        self.bar2.set_description_str('batch:{:4d}/{:4d}'.format(0, self.val_batch_num)) 
        self.bar2.set_postfix_str("validate─➤ los={1:.3f} avg={2:.3f}| {0}={3:.3f} avg={4:.3f}".format(
            self.metric_name, 0, 0, 0, 0)) 

    def close(self):
        self.bar0.close()
        self.bar1.close()
        self.bar2.close()
        print("\n\n")
        self.is_closed = True
        
    def updateTraining(self, loss, avg_loss, met, avg_met):
        self.bar1.update()
        self.current_train_batch += 1
        self.bar1.set_description_str('batch:{:4d}/{:4d}'.format(self.current_train_batch, self.train_batch_num))
        self.bar1.set_postfix_str(
            "training─➤ los={1:.3f} avg={2:.3f}| {0}={3:.3f} avg={4:.3f}".format(
                self.metric_name, loss, avg_loss, met, avg_met
            ) # format
        )
        self.avg_loss = avg_loss
        self.avg_met  = avg_met
        
    def updateValidation(self, val_loss, avg_val_loss, val_met, avg_val_met):
        self.bar2.update()   
        self.current_val_batch += 1
        self.bar2.set_description_str('v_bth:{:4d}/{:4d}'.format(self.current_val_batch, self.val_batch_num))  
        self.bar2.set_postfix_str(
            "validate─➤ los={1:.3f} avg={2:.3f}| {0}={3:.3f} avg={4:.3f}".format(
                self.metric_name, val_loss, avg_val_loss, val_met, avg_val_met
            ) 
        ) 
        self.avg_val_loss = avg_val_loss
        self.avg_val_met  = avg_val_met

    def beginEpoch(self):
        self.current_train_batch  = 0
        self.current_val_batch    = 0
        reset_tqdm(self.bar1)
        reset_tqdm(self.bar2)        
        pass
    
    def endEpoch(self, avg_loss=None, avg_val_loss=None, avg_met=None, avg_val_met=None):
        avg_loss     = self.avg_loss     if avg_loss     is None else avg_loss     
        avg_met      = self.avg_met      if avg_met      is None else avg_met      
        avg_val_loss = self.avg_val_loss if avg_val_loss is None else avg_val_loss 
        avg_val_met  = self.avg_val_met  if avg_val_met  is None else avg_val_met   
        self.current_epoch       += 1
        self.bar0.update()
        self.bar0.set_description_str('epoch:{:4d}/{:4d}'.format(self.current_epoch, self.init_epoch+self.total_epochs))
        self.bar0.set_postfix_str(
            " t_los={1:.3f}, t_acc={2:.3f}| v_los={3:.3f}, v_{0}={4:.3f}".format(
                self.metric_name, avg_loss, avg_met, avg_val_loss, avg_val_met
            )
        )

    def updateEpoch(self, avg_loss=None, avg_val_loss=None, avg_met=None, avg_val_met=None):
        avg_loss     = self.avg_loss     if avg_loss     is None else avg_loss     
        avg_met      = self.avg_met      if avg_met      is None else avg_met      
        avg_val_loss = self.avg_val_loss if avg_val_loss is None else avg_val_loss 
        avg_val_met  = self.avg_val_met  if avg_val_met  is None else avg_val_met  
        self.current_epoch       += 1
        self.current_train_batch  = 0
        self.current_val_batch    = 0
        self.bar0.update()
        self.bar0.set_description_str('epoch:{:4d}/{:4d}'.format(self.current_epoch, self.init_epoch+self.total_epochs))
        self.bar0.set_postfix_str(
            " t_los={1:.3f}, t_acc={2:.3f}| v_los={3:.3f}, v_{0}={4:.3f}".format(
                self.metric_name, avg_loss, avg_met, avg_val_loss, avg_val_met
            )
        )
        reset_tqdm(self.bar1)
        reset_tqdm(self.bar2)
                   