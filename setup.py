from distutils.core import setup

setup(name='mec',
      version='1.0',
      description='Multi Gpu training Library',
      author='CS',
      author_email='artintel@163.com',
      url='https://gitee.com/shinong/cv_dl_multi_gpu_basic_components',
      packages=[
          'mec', 
          'mec.comms', 
          'mec.data_manip', 
          'mec.training',
          'mec.utils'
        ],
      install_requires=['zmq', 'torch']
     )