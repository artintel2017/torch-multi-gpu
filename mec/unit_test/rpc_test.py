# rpc_test.py
# created: CS
# 测试rpc组件

import time
import torch.multiprocessing as mp

import mec.comms.sync_rpc as rpc

#ip = '192.168.1.99'
ip = '127.0.0.1'
port = 9900
worker_num = 4

count = 0
def test():
    global count
    count+=1
    print("recieved ", count, " times")
    return 'got'

def start_worker(rank):
    worker = rpc.SyncRpcWorker(ip, port, rank)
    worker.registerMethod(lambda x,y,z: len(x)+len(y)+len(z), 'a.b.c')
    worker.registerMethod(lambda x,y,z: x+y+z, 'all.add')
    worker.registerMethod(test)
    worker.mainLoop()

def start_controller(worker_num):
    controller = rpc.SyncRpcController(ip, port, worker_num)
    #controller.startWorking()
    # print( controller.a('a') )
    # print( controller.a.b("123") )
    #print( controller.a.b.c("123", [4, 'abc', 3.875], {1: 5, 666:(254, 'aba')}) )
    #print( controller.all.add( 8, 9, 10) )
    #controller.stopWorking()
    controller.stopLooping()
    
    #recieve_count = 0
    #for i in range(100000):
    #    result = controller.test()
    #    if result == ['got'] * controller.worker_num:
    #        recieve_count += 1
    #print("{} messages recieved".format(recieve_count) )
    
    controller.closeSocket()


if __name__ == '__main__':
    
    process_pool = []
    for i in range(worker_num):
        wp = mp.Process(target=start_worker, args=(i,))
        process_pool.append(wp)
    
    
    for wp in process_pool:
        wp.start()
        time.sleep(0.5)
 
    cp = mp.Process(target=start_controller, args=(worker_num,) )
    cp.start()
        
    for wp in process_pool:
        wp.join()
        
    
    