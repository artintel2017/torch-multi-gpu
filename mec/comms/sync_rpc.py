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



class SyncRpcBase:
    def __init__(self, server_ip, publish_port, report_port, logger):
        self.printToLog = logger
        #
        self.context        = None
        self.publish_addr   = "tcp://" + server_ip + ":" + publish_port
        self.publish_socket = None
        self.report_addr    = "tcp://" + server_ip + ":" + report_port
        self.report_socket  = None
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
    def __init__(self, server_ip, publish_port, report_port, worker_num, logger=print):
        #
        self.printToLog = logger
        self.printToLog('initiating synchronized rpc controller')        
        SyncRpcBase.__init__(self, server_ip, publish_port, report_port, logger)
        self.worker_num = worker_num
        # self check
        self._WaitPubSockReady()
        self.printToLog('publishing socket ready')
        while True:
            detected_worker_num = self._detectWorkers()
            if detected_worker_num==worker_num: break
        self.printToLog('workers ready')
        # if detected_worker_num<=0:
        #     self.printToLog('warning: no workers detected')
        # if worker_num!=detected_worker_num:
        #     self.printToLog('warning: designated worker number mismatch')

    def __del__(self):
        self.closeSocket()

    def __getattr__(self, name):
        method = _Method(name, self.callMethod, self.printToLog)
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
        self.check_sub_socket = self.context.socket(zmq.SUB)
        self.check_sub_socket.connect(self.publish_addr)
        self.check_sub_socket.setsockopt(zmq.SUBSCRIBE, b'')
        self.printToLog("socket initizating complete")

    def closeSocket(self):
        self.printToLog("closing socket")
        if self.publish_socket != None:
            self.publish_socket.unbind(self.publish_addr)
            self.publish_socket = None
        if self.report_socket != None:
            self.report_socket.unbind(self.report_addr)
            self.report_socket = None
        if self.check_sub_socket != None:
            self.check_sub_socket.disconnect(self.publish_addr)
            self.check_sub_socket = None

    def callMethod(self, name, *args, **kwargs):
        """
            调用工作者的方法，并获取返回值
        """
        self.printToLog("calling: ", (name, args, kwargs) )
        self.broadcastMessage( (name, args, kwargs) )
        result = self.gatherMessages()
        self.printToLog("result: ", result)
        return result

    def broadcastMessage(self, msg):
        """
            将消息广播至所有的工作者
        """
        self.printToLog("message sent:", msg)
        self.publish_socket.send( repr(msg).encode() )
        
    def recieveSingleMessage(self):
        """
            从工作者收集信息
            一次只收集一个
        """
        return eval(self.report_socket.recv().decode())
    
    def gatherMessages(self):
        """
            从所有的工作者汇集消息
            等候直到所有的工作者消息汇总完毕
            返回一个list
        """
        result_list = []
        for i in range(self.worker_num):
            result_list.append( eval(self.report_socket.recv(0).decode()) )
        return result_list

    def _WaitPubSockReady(self):
        while True:
            self.publish_socket.send(repr(test_signal).encode())
            try:
                result = self.check_sub_socket.recv(zmq.NOBLOCK)
                if result is not None:
                    result = eval(result.decode())
                self.printToLog('message: ',result)
                if result == test_signal: break
            except zmq.ZMQError:
                self.printToLog('not ready')
            time.sleep(0.5)
    
    def _detectWorkers(self):
        self.broadcastMessage( ('detectRespond', (), {}) )
        self.printToLog('waiting for workers to respond')
        time.sleep(1)
        result_list = []
        while True:
            try:
                result_list.append(self.report_socket.recv(zmq.NOBLOCK))
            except zmq.ZMQError as e:
                break
        self.printToLog('respond:', result_list)
        correct_respond_count = 0
        for respond in result_list:
            if eval(respond)==good_signal: correct_respond_count+=1
        self.worker_num = correct_respond_count
        self.printToLog('{} workers detected'.format(self.worker_num))
        return correct_respond_count

class SyncRpcWorker(SyncRpcBase):
    """
        同步RPC客户端
    """

    def __init__(self, server_ip, publish_port, report_port, logger=print):
        SyncRpcBase.__init__(self, server_ip, publish_port, report_port, logger)
        self.printToLog = logger
        self.function_dict = {}
        self.is_working = False
        self.registerMethod(self.stopLoop)
        self.registerMethod(self.detectRespond)

    def __del__(self):
        self.closeSocket()

    def initSocket(self):
        self.printToLog("initilzating socket:")
        self.printToLog("publish addr = '{}'".format(self.publish_addr) )
        self.printToLog("report  addr = '{}'".format(self.report_addr) )
        self.context       = zmq.Context()
        self.publish_socket   = self.context.socket(zmq.SUB)
        self.publish_socket.connect(self.publish_addr)
        self.publish_socket.setsockopt(zmq.SUBSCRIBE, b'')
        self.report_socket  = self.context.socket(zmq.PUSH)
        self.report_socket.connect(self.report_addr)

    def closeSocket(self):
        if self.publish_socket != None:
            self.publish_socket.disconnect(self.publish_addr)
            self.publish_socket = None
        if self.report_socket != None:
            self.report_socket.disconnect(self.report_addr)
            self.report_socket = None

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
                
    def detectRespond(self):
        return good_signal

    def startLoop(self):
        self.is_working = True
        while self.is_working:
            msg = self.recieveBroadcast()
            if msg==test_signal: continue
            self.printToLog("message recieved: ", msg)
            func_name, func_args, func_kwargs = msg
            result = self.excecuteMethod(func_name, func_args, func_kwargs)
            self.reportMessage(result)
            
    def stopLoop(self):
        self.is_working = False
        return good_signal