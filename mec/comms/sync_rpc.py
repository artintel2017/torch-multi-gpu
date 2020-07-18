# process_signalling.py
# 创建：陈硕
# 创建日期：2020.06.20
# 文件描述：进程通信封装，只用于通知，不用于大量传输数据



import zmq
import time


# ----------------------------- 同步RPC -----------------------------
# 多个server对应一个client

test_signal   = 'test'
good_signal   = 'good'
check_signal  = 'check'
quit_signal   = 'quit'
start_signal  = 'start'



class SyncRpcBase:
    def __init__(self, server_ip, port, logger):
        self.printToLog = logger
        #
        self.context        = None
        self.publish_addr   = "tcp://{}:{}".format(server_ip, str(port)   )
        self.publish_socket = None
        self.report_addr    = "tcp://{}:{}".format(server_ip, str(port+1) )
        self.report_socket  = None
        self.check_addr     = "tcp://{}:{}".format(server_ip, str(port+2) )
        self.check_socket   = None
        self.logger         = logger
        self.initSocket()

    def __del__(self):
        self.closeSocket()


class _Method: 
    # some magic to bind an RPC method to an RPC server. 
    # supports "nested" methods (e.g. examples.getStateName) 
    def __init__(self, name, send, logger=print): 
        self.__name = name
        self.__send = send
        self.printToLog = logger
    def __getattr__(self, name): 
        self.printToLog("attribute {:s}".format(name) ) 
        method = _Method("{:s}.{:s}".format(self.__name, name), self.__send) 
        self.__setattr__(name, method) 
        return method 
    def __call__(self, *args, **kwargs): 
        return self.__send(self.__name, *args, **kwargs)  

class SyncRpcController(SyncRpcBase):
    """
        同步RPC服务端
    """
    def __init__(self, server_ip, port, worker_num, logger=print):
        #
        self.printToLog = logger
        self.printToLog('initiating synchronized rpc controller')
        SyncRpcBase.__init__(self, server_ip, port, logger)
        self.worker_num = worker_num
        self.is_working = False
        self.is_looping = False
        # check workers
        # self check
        self.printToLog('waiting for publishing socket')
        self._WaitPubSockReady()
        self.printToLog('publishing socket ready')
        self.printToLog('synchronizing all workers')
        self._waitAllWorkersReady()
        self.printToLog('confirmed {} workers ready'.format(self.worker_num) )

    def __del__(self):
        self.closeSocket()

    def __getattr__(self, name):
        method = _Method(name, self._callMethod, self.printToLog)
        self.__setattr__(name, method)
        return method

    def initSocket(self):
        self.printToLog("initizating socket:")
        self.printToLog("publish  addr = '{}'".format(self.publish_addr) )
        self.printToLog("report   addr = '{}'".format(self.report_addr) )
        self.context       = zmq.Context()
        # publish socket
        self.publish_socket = self.context.socket(zmq.PUB)
        self.publish_socket.bind(self.publish_addr)
        # report socket
        self.report_socket = self.context.socket(zmq.PULL)
        self.report_socket.bind(self.report_addr)
        # self check socket
        self.self_check_sub_socket = self.context.socket(zmq.SUB)
        self.self_check_sub_socket.connect(self.publish_addr)
        self.self_check_sub_socket.setsockopt(zmq.SUBSCRIBE, b'')
        # workers check socket
        self.check_socket = self.context.socket(zmq.REQ)
        self.check_socket.bind(self.check_addr)
        # 
        self.printToLog("socket initizating complete")

    def closeSocket(self):
        self.printToLog("closing socket ...")
        if self.publish_socket != None:
            self.publish_socket.unbind(self.publish_addr)
            self.publish_socket = None
        if self.report_socket != None:
            self.report_socket.unbind(self.report_addr)
            self.report_socket = None
        if self.self_check_sub_socket != None:
            self.self_check_sub_socket.disconnect(self.publish_addr)
            self.self_check_sub_socket = None
        if self.check_socket !=None:
            self.check_socket.unbind(self.check_addr)
            self.check_socket = None
        self.printToLog("socket closed")

    def _callMethod(self, name, *args, **kwargs):
        """
            调用工作者的方法，并获取返回值
        """
        self.printToLog("calling: ", (name, args, kwargs) )
        self._broadcastMessage( (name, args, kwargs) )
        result = self._gatherMessages()
        self.printToLog("result: ", result)
        return result

    def _broadcastMessage(self, msg):
        """
            将消息广播至所有的工作者
        """
        self.printToLog("message sent:", msg)
        self.publish_socket.send( repr(msg).encode() )
        
    def _recieveSingleMessage(self):
        """
            从工作者收集信息
            一次只收集一个
        """
        return eval(self.report_socket.recv().decode())
    
    def _gatherMessages(self):
        """
            从所有的工作者汇集消息
            等候直到所有的工作者消息汇总完毕
            返回一个list
        """
        result_list = []
        for i in range(self.worker_num):
            result_list.append( eval(self.report_socket.recv(0).decode()) )
            self.printToLog(i+1, "results recieved")
        return result_list

    def _WaitPubSockReady(self):
        while True:
            self.publish_socket.send(repr(test_signal).encode())
            try:
                result = self.self_check_sub_socket.recv(zmq.NOBLOCK)
                if result is not None:
                    result = eval(result.decode())
                self.printToLog('message: ',result)
                if result == test_signal: break
            except zmq.ZMQError:
                self.printToLog('not ready')
            time.sleep(1)
    
    def _waitAllWorkersReady(self):
        self._sendControlSignal(check_signal, sync_check=True)
        # if self.is_working:
        #     self.printToLog('warning! checking workers in working status')
        #     return
        # workers_set = set()
        # while True:
        #     self.printToLog('sending control signal, confirmed worker num: ',len(workers_set))
        #     # count workers
        #     self.check_socket.send(repr(check_signal).encode() )
        #     rank = eval(self.check_socket.recv().decode())
        #     self.printToLog('worker respond got, rank {}'.format(rank))
        #     assert type(rank) is int, 'check respond signal error, should be int'
        #     assert rank<self.worker_num and rank>=0, \
        #         'worker respond rank exceeds limit, worker num {}, get {}'.format(
        #         self.worker_num, rank)
        #     if rank not in workers_set: 
        #         workers_set.add(rank)
        #         self.printToLog('counted workers: ', workers_set)
        #         if len(workers_set)==self.worker_num: # counted all
        #             break
        #     else: #rank in workers_set: indicate delay joined worker, count again
        #         self.printToLog(' [warning] delay joined worker, rank {}, count again'.format(rank) )
        #         time.sleep(1)
        #         workers_set = { rank }
                    
    def _sendControlSignal(self, signal, sync_check=False):
        if self.is_working:
            self.printToLog('warning! sending control signal in working status')
            return
        workers_set = set()
        while len(workers_set)<self.worker_num:
            self.printToLog('sending control signal, confirmed worker num: ',len(workers_set))
            self.check_socket.send(repr(signal).encode() )
            rank = eval(self.check_socket.recv().decode())
            #self.printToLog('worker respond got, rank {}'.format(rank))
            assert type(rank) is int, 'check respond signal error, should be int'
            assert rank<self.worker_num and rank>=0, \
                'worker respond rank exceeds limit, worker num {}, get {}'.format(
                self.worker_num, rank)
            if rank not in workers_set: 
                workers_set.add(rank)
                self.printToLog('counted workers: ', workers_set)
                if len(workers_set)==self.worker_num: # counted all
                    break
            elif sync_check: # not synchronized: count again
                self.printToLog('==== delay joined worker, rank {}, count again'.format(rank) )
                time.sleep(1)
                workers_set = { rank }
            else: 
                raise Exception('unhandled asynchronized signal, probably indicates disorder of workers, \
                                try reboot all workers before start controllers')

    
    def startWorking(self):
        if not self.is_working:
            self.printToLog('calling all workers to start')
            self._sendControlSignal(start_signal)
            self.is_working = True
            self.is_looping = True
    
    def stopWorking(self):
        self._callMethod('stopWorking')
        self.is_working = False
    
    def stopLooping(self):
        if self.is_working:
            self._broadcastMessage(quit_signal)
            #self._callMethod('stopLooping')
        else:
            self._sendControlSignal(quit_signal)
        self.is_working = False
        self.is_looping = False 

class SyncRpcWorker(SyncRpcBase):
    """
        同步RPC客户端
    """
    def __init__(self, server_ip, port, rank, logger=
                  print):
        SyncRpcBase.__init__(self, server_ip, port, logger)
        self.printToLog = logger
        self.rank = rank
        self.function_dict = {}
        self.is_working = False
        self.is_looping = False
        self.registerMethod(self.stopWorking)
        self.registerMethod(self.stopLooping)
        #self.registerMethod(self.detectRespond)

    def __del__(self):
        self.closeSocket()

    def initSocket(self):
        self.printToLog("initilzating socket:")
        self.printToLog("publish addr = '{}'".format(self.publish_addr) )
        self.printToLog("report  addr = '{}'".format(self.report_addr) )
        self.printToLog("check   addr = '{}'".format(self.check_addr) )
        self.context       = zmq.Context()
        # publish socket
        self.publish_socket   = self.context.socket(zmq.SUB)
        self.publish_socket.connect(self.publish_addr)
        self.publish_socket.setsockopt(zmq.SUBSCRIBE, b'')
        # report socket
        self.report_socket  = self.context.socket(zmq.PUSH)
        self.report_socket.connect(self.report_addr)
        # workers check socket
        self.check_socket = self.context.socket(zmq.REP)
        self.check_socket.connect(self.check_addr)

    def closeSocket(self):
        self.printToLog('closing socket, rank {}'.format(self.rank) )
        if self.publish_socket != None:
            self.publish_socket.disconnect(self.publish_addr)
            self.publish_socket = None
        if self.report_socket != None:
            self.report_socket.disconnect(self.report_addr)
            self.report_socket = None
        if self.check_socket !=None:
            self.check_socket.disconnect(self.check_addr)
            self.check_socket = None

    def recieveBroadcast(self):
        return eval(self.publish_socket.recv().decode())

    def reportMessage(self, msg):
        """
            将消息发送至控制者
        """
        return self.report_socket.send( repr(msg).encode() )

    def registerMethod(self, function, name=None):
        if name is None:
            name = function.__name__
        self.function_dict[name] = function

    def excecuteMethod(self, func_name, args, kwargs):
        if func_name in self.function_dict:
            self.printToLog("excecuting function: [func name] {}; [args] {}; [kwargs] {}".format(
                func_name, args, kwargs )
            )
            result = self.function_dict[func_name](*args, **kwargs)
        else:
            self.printToLog("warning: wrong function name. [func name] {}; [args] {}; [kwargs] {}".format(
                func_name, args, kwargs
            ))
            result = None
        self.printToLog('result: ', result)
        return result
                
    # def detectRespond(self):
    #     return good_signal

    def waitControlSignal(self):
        self.printToLog('waiting for control signal, rank {}'.format(self.rank) )
        signal = eval(self.check_socket.recv().decode())
        self.printToLog('controller signal recieved: \"{}\"'.format(signal))
        self.printToLog('respond control signal, rank {}'.format(self.rank) )
        #time.sleep(0.2)
        self.check_socket.send( repr(self.rank).encode() )
        self.printToLog('respond sent: {}'.format(self.rank) )
        return signal

    def mainLoop(self):
        #self.is_working = False
        self.is_looping = True
        while self.is_looping:
            signal = self.waitControlSignal()
            if signal == quit_signal:
                self.is_looping = False
                time.sleep(3)
                break
            if signal == check_signal:
                continue
                time.sleep(1)
            elif signal==start_signal:
                self.printToLog('start working loop')
                self.is_working = True
                while self.is_working:
                    self.printToLog("waiting for task message")
                    msg = self.recieveBroadcast()
                    self.printToLog("message recieved: \"{}\"".format(msg) )
                    if msg==test_signal: continue
                    elif msg==quit_signal:
                        self.is_looping = False
                        break
                    func_name, func_args, func_kwargs = msg
                    result = self.excecuteMethod(func_name, func_args, func_kwargs)
                    self.reportMessage(result)
        #time.sleep(3)
        self.printToLog('exiting')
        #self.closeSocket()
    
    def stopWorking(self):
        self.is_working = False
        return good_signal
    
    def stopLooping(self):
        self.is_looping = False
        self.is_working = False