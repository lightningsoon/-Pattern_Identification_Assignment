import json
import numpy as np


#To->Eval
w_filename='w.json'

def save_par(w):
    with open(w_filename,'w') as fp:
        json.dump(list(w.ravel()),fp)

def load_par():
    with open(w_filename,'r') as fp:
        return np.array(json.load(fp))[:,np.newaxis]

def Comp_as1(y,label):
    # y=(y>0.0 if label>0 else y<=0.0)
    y= y>0.0
    return np.sum(y)

def simple_Fisher_Friterion(w,S_b):
    '''
    Fisher简化版的判别函数
    :param w: 投影向量d*1
    :param S_b: 类间离散度矩阵
    :return:
    '''
    return np.dot(np.dot(np.transpose(w),S_b),w)[0][0]