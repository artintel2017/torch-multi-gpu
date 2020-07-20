

train                    = False
test                     = False
score                    = False
prod                     = False
mix                      = False
deploy                   = False
continue_training        = False
batch_size               = 1
learning_rate            = 1e-3
epochs                   = 1
process_num_per_loader   = 0                    # 每个DataLoader启用的进程数
path                     = 'results/temp'
history_filename         = 'history.json'
model_filename           = 'current_model.pth'
best_model_filename      = 'best_model.pth'
excel_filename           = 'scores.xls'
control_ip               = "127.0.0.1"       # manager的IP
basic_port               = 12500
worker_gpu_ids           = [0]              # worker所使用的gpu编号 [0,1,2,3]
worker_ranks             = [0]              # worker本身编号 [0,1,2,3]
sync_worker_num          = 1    # 总worker数，单机的情况等于上两者的长度
norm = None

