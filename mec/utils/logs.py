# logs.py
# 创建：cs
# 创建日期：2020.06.30
# 日志功能

import sys
import time

class Logger():
    def __init__(self, filepath=None, prefix=''):
        try:
            self.log_file = open(filepath, 'w')
        except FileNotFoundError as e:
            print(e)
            self.log_file = sys.stdout
        self.prefix = '[' + prefix + '{}]'
    
    def __del__(self):
        self.log_file.close()

    def print(self, *args, **kwargs):
        print(
            self.prefix.format( time.strftime("%Y-%m-%d_%H:%M:%S") ),
            *content,
            **kwargs,
            file=self.log_file, 
            flush=True
        )

    self.__call__ = self.print