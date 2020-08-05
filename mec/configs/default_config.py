# default configs

class Options:
    def __init__(self):
        self.batch_size               = 1
        self.learning_rate            = 1e-3
        self.epochs                   = 1
        self.process_num_per_loader   = 0                    # 每个DataLoader启用的进程数
        self.path                     = 'results/temp'
        self.history_filename         = 'results/temp/history.json'
        self.model_filename           = 'results/temp/current_model.pth'
        self.best_model_filename      = 'results/temp/best_model.pth'
        self.excel_filename           = 'results/temp/scores.xlsx'
        self.control_ip               = "127.0.0.1"       # manager的IP
        self.basic_port               = 12500
        self.worker_gpu_ids           = [0]              # worker所使用的gpu编号 [0,1,2,3]
        self.worker_ranks             = [0]              # worker本身编号 [0,1,2,3]
        self.sync_worker_num          = 1    # 总worker数，单机的情况等于上两者的长度

conf_g = Options()