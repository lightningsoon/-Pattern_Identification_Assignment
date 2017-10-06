import logging
import numpy as np
import random

#%%
# To->process_data
## ---sonar
# label=('R','M')
# file_size=(208,61)
## ---usps
label=('3','8')
file_size=(9298,257)
label_dict={label[0]:3,label[1]:8}

def save_38(data):
    import csv
    with open('usps_3&8.csv','w',newline='') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerows(data)
        print('写完了')
        exit(0)

def read_datas(file_name):
    '''
    读取并处理文件
    :param file_name:
    :return:
    '''
    datas1,datas2=[],[]
    with open(file_name,'r') as f:
        datas=f.readlines()
    #%%
    for i in range(len(datas)):
        datas[i]=datas[i][:-1]
        datas[i]=datas[i].split(',')
        # print(i)
        datas[i][:-1]=list(map(float,datas[i][:-1]))
        # print(datas[i][-1])
        # try:
        datas[i][-1]=label_dict[datas[i][-1]]#替换成数字
        # except KeyError:
        #     continue
    # print('数据大小',np.array(datas).shape)
    # assert np.array(datas).shape==file_size
    # 可以用转成数组处理
    for data in datas:
        # print(data[-1])
        #分别添加到两个数据
        if data[-1]==label_dict[label[0]]:
            datas1.append(data)
        elif data[-1]==label_dict[label[1]]:
            datas2.append(data)
        else:
            # continue
            print('这个标签 %s 未知，请看一下，在第%d行' % (data[-1],datas.index(data)))
    # 可以用转成数组处理
    #随机一下顺序
    # save_38(datas1+datas2)
    map(random.shuffle,[datas1,datas2])
    L1,L2,Lall=len(datas1),len(datas2),file_size[0]
    # assert (L1+L2)==Lall
    print('——————数据读取完成！——————')
    return ((datas1,datas2),L1,L2,Lall)
#%%
def split_datas(datas,n):
    '''
    验证集和训练集分开
    :param n: n折交叉验证
    :param datas: 打乱后的数据，
    :return: 若n=10，第一类，第二类，交叉验证集20*61
    '''
    data_block_len=[len(data)//n for data in datas]
    for i in range(n):
        val,cla=[],[]
        for j,data in enumerate(datas):
            val_left,val_right=data_block_len[j]*(i),data_block_len[j]*(i+1)
            val.append(np.array(data[val_left:val_right]))
            cla.append(np.array(data[:val_left]+data[val_right:]))
        yield cla,val