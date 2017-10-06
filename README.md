#模式识别大作业一

##依赖

Python3.5
anaconda套装

##复现

把数据集放到目录下
运行main.py
注释不同文件名可以运行不同的数据集

##结果

'''
sonar
  预测      第1类   第2类   全部
实际第1类    63     33      96
实际第2类    36     72      108
mixed_accuracy：0.661765

usps_3&8
  预测      第1类   第2类   全部
实际第1类    789     31      820
实际第2类    20     680      700
mixed_accuracy：0.966447
'''

##数据集

都处理成一样的格式，取消usps的one-hot
每一行一个样本，前面d个数据，-1是label
链接：
[声呐 ](http://pan.baidu.com/s/1bpIs1lX )密码：43jd
[3和8 ](http://pan.baidu.com/s/1qXU2YpY )密码：ja0c