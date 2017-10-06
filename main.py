# %%
import numpy as np
from Fisher import Fisher
import process_data as pd
import Eval
import time
import os





# To->main

### t0=time.time()
# file_name = './Sonar.csv'
file_name = './usps_3&8.csv'

cross_validation_number = 10
class_number = 2


# dataset = dict()

def main():
    data = dict(x=[[], []], y=[[], []])
    val_data = dict(x=[[], []], y=[[], []])
    Pred_True = [0] * class_number
    pred_num=[0,0]
    L = [0, 0]
    last_j=float('-inf')

    datas, L[0], L[1], Lall = pd.read_datas(file_name)
    data_block = pd.split_datas(datas, cross_validation_number)
    # i = 0
    for cla, val in data_block:
        # print(i)
        # i += 1
        for j in range(len(cla)):
            (data['x'][j], data['y'][j], val_data['x'][j], val_data['y'][j]) = (
            cla[j][:, :-1], cla[j][:, -1:], val[j][:, :-1], val[j][:, -1:])
        myfisher = Fisher(L[0], L[1], Lall, data)

        W = myfisher.W_Direction()  # 投影参数
        J=Eval.simple_Fisher_Friterion(W,myfisher.between_class_scatter_matrix())#评估一下投影效果，并更新投影向量
        if J>last_j:
            last_j=J
            W_great=W
        W0 = myfisher.OneKey_W0(W)  # 阈值
        # %%
        for j in range(class_number):
            predict_y = myfisher.Pred_result(W, val_data['x'][j], W0)
            # print(predict_y)
            pred_num[j]+=len(val_data['x'][j])
            Pred_True[j] += Eval.Comp_as1(predict_y, val_data['y'][j][0][0])
            # if j==1:
            #     print(Eval.Comp_as1(predict_y, val_data['y'][j][0][0]))

            # print(W,W0)
            # print(S_W)
            # print(time.time()-t0)
            # break
    print('  预测      第1类   第2类   全部')
    for j in range(class_number):
        print('实际第{0}类    {1}     {2}      {3}'.format(j+1,Pred_True[j],pred_num[j]-Pred_True[j],pred_num[j]))
    print('mixed_accuracy：%f' % ((Pred_True[0]+pred_num[1]-Pred_True[1])/(pred_num[0]+pred_num[1])))

    # print(os.path.isfile(Eval.w_filename),W_great)
    # if os.path.isfile(Eval.w_filename)==False and W_great:
    print('评价函数最大的W保存在',os.path.abspath(Eval.w_filename))
    Eval.save_par(W_great)
    pass


if __name__ == '__main__':
    main()

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