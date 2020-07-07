# rpc_test.py
# created: CS
# 测试rpc组件

import time
import torch.multiprocessing as mp

import mec.comms.sync_rpc as rpc

ip = '192.168.1.99'

def start_worker():
    worker = rpc.SyncRpcWorker(ip, '9001', '9002', 1)
    worker.registerMethod(lambda x,y,z: len(x)+len(y)+len(z), 'a.b.c')
    worker.registerMethod(lambda x,y,z: x+y+z, 'all.add')
    worker.startLoop()

def start_controller():
    controller = rpc.SyncRpcController(ip, '9001', '9002', 1)
    time.sleep(1)
    print( controller.a('a') )
    print( controller.a.b("123") )
    print( controller.a.b.c("123", [4, 'abc', 3.875], {1: 5, 666:(254, 'aba')}) )
    print( controller.all.add( 8, 9, 10) )
    print( controller.stopLoop() )
    
    
if __name__ == '__main__':
    
    wp = mp.Process(target=start_worker)
    cp = mp.Process(target=start_controller)
    
    wp.start()
    cp.start()
    
    cp.join()
    wp.join()
    
    