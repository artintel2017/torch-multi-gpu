# dist_test.py
# created: CS
# 测试显卡通信组件

import time
import torch
import torch.multiprocessing as mp
import mec.comms.sync_rpc as rpc
import mec.comms.transmit as trans
import mec.utils.logs as logs

ip   = '192.168.1.99'
port = '9999'



def start_process(rank, gpu_list):
    world_size = len(gpu_list)
    log = logs.Logger(prefix='process {}|'.format(rank))
    env = trans.DistEnv(rank, world_size, control_ip=ip, dist_port=port, logger=log)
    test_group_list = [0,1]
    group = env.newGroup(test_group_list)
    transmittor = trans.TensorTransmittor([0,1,2,3],logger=log)
    
    p = torch.nn.Parameter(
        torch.randn(
            (1,4), 
            requires_grad=True, 
            device=torch.device(
                'cuda:{}'.format(gpu_list[rank]) 
            )
        )
    )
    
    l = torch.sum(p*p)
    l.backward()
    
    
    log("tensor: ", p)
    log("grad: " , p.grad)
    time.sleep(1)
    
    if rank in test_group_list:
        transmittor.crossTensors(p, group)
        transmittor.crossGrads(p, group)
        log("tensor: ", p)
        #log("grad: ", p.grad)
        
        

def main():
    gpu_list = [0,1,2,3]
    rank_list = list(range(len(gpu_list)))
    process_pool = []
    for rank in gpu_list:
        p = mp.Process(target=start_process, args=(rank, gpu_list))
        process_pool.append( p )
    for p in process_pool:
        p.start()
    
if __name__ == '__main__':
    main()