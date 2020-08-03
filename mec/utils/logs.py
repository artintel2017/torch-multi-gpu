# logs.py
# 创建：cs
# 创建日期：2020.06.30
# 日志功能

import sys
import time

class Logger():
    def __init__(self, filepath=None, logfile=sys.stdout, prefix=''):
        self.log_file = logfile
        if filepath is not None:
            try:
                self.log_file = open(filepath, 'w')
            except FileNotFoundError as e:
                print(e)
        self.prefix = '[' + prefix + '|{}]'
    
    # def __del__(self):
    #     self.log_file.close()

    def __call__(self, *args, **kwargs):
        print(
            self.prefix.format( time.strftime("%Y-%m-%d_%H:%M:%S") ),
            *args,
            **kwargs,
            file=self.log_file,
            flush=True
        )
        
    
        
    def __getattr__(self, name):
        return self.log_file.__getattribute__(name)
