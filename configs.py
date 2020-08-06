from mec.configs.default_config import conf_g
import os
import argparse


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
process_num_per_loader   = 8                     # 每个DataLoader启用的进程数
path                     = 'results/temp'
history_filename         = 'history.json'
model_filename           = 'current_model.pth'
best_model_filename      = 'best_model.pth'
excel_filename           = 'scores.xls'
control_ip               = "127.0.0.1"            # control IP
basic_port               = 12500                  # 基本端口，会占用其后几个连续端口
worker_gpu_ids           = [0,1,2,3]              # worker所使用的gpu编号 [0,1,2,3]
worker_ranks             = [0,1,2,3]              # worker本身编号 [0,1,2,3]
sync_worker_num          = 4    # 总worker数，单机的情况等于上两者的长度
batch_size               = 256*sync_worker_num

# 多机运行时需指定本地使用哪个网卡，否则可能因网络连接速度太慢拖累训练速度
# 单机训练时不需要此参数，默认指定本地地址127.0.0.1
# os.environ['NCCL_SOCKET_IFNAME'] = 'eno2' 
# os.environ['NCCL_SOCKET_IFNAME'] = 'eno1np0'

def parse_configs():
    global train         
    global test                    
    global score                   
    global prod                    
    global mix                     
    global deploy                  
    global continue_training       
    global batch_size              
    global learning_rate           
    global epochs                  
    global process_num_per_loader  
    global path                    
    global history_filename        
    global model_filename          
    global best_model_filename     
    global excel_filename               
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', '--train', action='store_true',
        help='train model')
    parser.add_argument('-test', '--test', action='store_true',
        help='evaluate model on test set')
    parser.add_argument('-c', '--continue_training', action='store_true',
        help='continue training from last point')
    parser.add_argument('-score', '--score', action='store_true',
        help='calc precision, recall and F1, then write to an excel file')
    parser.add_argument('-prod', '--prod', action='store_true',
        help='test production per image')
    parser.add_argument('-mix', '--mix', action='store_true',
        help='output image mix matrix as xlsx file')
    parser.add_argument('-d', '--deploy', action='store_true',
        help='generate index to wiki_idx json file')
    parser.add_argument('-lr', '--learning_rate', type=float,
        help='designating statis training rate')
    parser.add_argument('-e', '--epochs', type=int,
        help='how many epochs to train in this run')
    parser.add_argument('-p', '--path', type=str,
        help='path to store results')
    parser.add_argument('-manager', '--start_manager', action='store_true',
        help='train model')
    parser.add_argument('-workers', '--start_workers', action='store_true',
        help='train model')

    args = parser.parse_args()

    if args.train:
        print("training")
        global train
        train=True
    if args.test:
        test=True
    if args.score:
        score=True
    if args.prod:
        prod=True
    if args.mix:
        mix=True
    if args.deploy:
        deploy=True
    if args.continue_training:
        continue_training=True
    if args.learning_rate:
        learning_rate = args.learning_rate
    if args.epochs:
        epochs = args.epochs
    if args.path:
        path = args.path
    history_filename        = os.path.join(path, history_filename)
    model_filename          = os.path.join(path, model_filename)
    best_model_filename     = os.path.join(path, best_model_filename)
        