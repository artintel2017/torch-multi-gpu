# process_signalling.py
# 创建：陈硕
# 创建日期：2020.06.20
# 文件描述：进程通信封装，只用于通知，不用于大量传输数据



import zmq
import time


# ----------------------------- 同步RPC -----------------------------
# 多个server对应一个client



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
        SyncRpcBase.__init__(self, server_ip, publish_port, report_port, logger)
        self.printToLog = logger

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
        self.publish_socket = self.context.socket(zmq.PUB)
        self.publish_socket.bind(self.publish_addr)
        self.report_socket = self.context.socket(zmq.PULL)
        self.report_socket.bind(self.report_addr)

    def closeSocket(self):
        self.printToLog("closing socket")
        if self.publish_socket != None:
            self.publish_socket.unbind(self.publish_addr)
            self.publish_socket = None
        if self.report_socket != None:
            self.report_socket.unbind(self.report_addr)
            self.report_socket = None

    def callMethod(self, name, *args, **kwargs):
        """
            调用工作者的方法，并获取返回值
        """
        self.broadcastMessage( (name, args, kwargs) )
        return self.gatherMessages()

    def broadcastMessage(self, msg):
        """
            将消息广播至所有的工作者
        """
        print("message:", msg)
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
            result_list.append( eval(self.report_socket.recv().decode()) )
        return result_list
    
    def detectWorkers(self):
        self.broadcastMessage( ('detectRespond', [], {}) )
        self.printToLog('waiting for workers to respond')
        time.sleep(1)
        result_list = []
        while True:
            try:
                result_list.append(self.report_socket.recv(zmq.NOBLOCK))
            except zmq.ZMQError as e:
                break
        self.worker_num = len(result_list)
        return self.worker_num

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
            return self.function_dict[func_name](*args, **kwargs)
        else:
            self.printToLog("warning: wrong function name. [func name] {}; [args] {}; [kwargs] {}".format(
                func_name, args, kwargs
            ))
            return None
        
    def detectRespond(self):
        return '1'

    def startLoop(self):
        self.is_working = True
        while self.is_working:
            msg = self.recieveBroadcast()
            func_name, func_args, func_kwargs = msg
            result = self.excecuteMethod(func_name, func_args, func_kwargs)
            self.reportMessage(result)
            
    def stopLoop(self):
        self.is_working = False
        return 'OK'