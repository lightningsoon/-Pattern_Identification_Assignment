import numpy as np

x = []  # n*d
w = []  # d*1
y = []  # 0*n

x = np.array(x)
w = np.array(w)
y = np.array(y)


def within_class_scatter_matrix(x, m):
    '''

    :param x: n*d->n*d*1
    :param m: d*1->1*d*1
    :return:类內离散度矩阵 d*d
    '''
    x = x[:, :, np.newaxis]
    m_d = len(m)
    m = m[np.newaxis, :, :]  # 1*d*1
    n, d = np.shape(x)[0], np.shape(x)[1]
    assert d == m_d
    temp = x - m
    temp_tran = np.transpose(temp, (0, 2, 1))  # n*1*d
    # print(list(map(np.shape,[x-m,x_transpose-m_transpose])))
    # 三维点乘变循环
    s = []
    for a, b in zip(temp, temp_tran):
        s.append(np.dot(a, b))
    s = np.array(s)
    s = np.sum(s, 0)
    assert s.shape == (d, d)
    return s


def pooled_within_class_scatter_matrix(wcsm1, wcsm2, L1, L2, Lall):
    '''

    :param wcsm1: d*d
    :param wcsm2:
    :return: 总类內离散度矩阵 d*d
    '''
    return (L1 * wcsm1 + L2 * wcsm2) / Lall





def projective_mean(w, m):
    '''

    :param w: 投影方向 d*1
    :param m: 类均值向量 d*1
    :return:投影后的均值 0*0 scalar
    '''

    def pred_y(x, w):
        '''
        :param x: n*d
        :param w: d*1
        :return: 预测值 0*n
        '''
        return x.dot(w).ravel()

    assert np.shape(w) == np.shape(m)
    # print('w:',np.transpose(w).shape,'m',m.shape)
    # print(pred_y(np.transpose(w),m)[0])
    return pred_y(np.transpose(w), m)[0]


def projective_within_class_scatter_matrix(y, M):
    '''

    :param y: 投影后的样本0*n
    :param M: 均值向量投影后 0*0
    :return:投影后类內离散度0*0 scalar
    '''
    M = np.array([M])  # 0*1
    return pow(np.linalg.norm(y - M), 2)


def projective_between_class_scatter_matrix(m1, m2):
    '''

    :param m1: 投影后均值
    :param m2:
    :return: 投影后类间离散度0*0 scalar
    '''
    return pow((m1 - m2), 2)


def w_direction(S_W, m1, m2):
    '''

    :param S_W:总类內离散度矩阵 d*d
    :param m1: 类均值向量 d*1
    :param m2:
    :return: d*1 投影参数（大小无所谓）
    '''
    # print(S_W**-1)
    # print()
    return np.dot(np.linalg.inv(S_W), (m1 - m2))


def W0(m1, m2):
    '''
    不考虑先验概率
    :param m1: scalar 投影后均值
    :param m2:
    :return: 阈值
    '''
    return -(m1 + m2) / 2

class Fisher(object):
    def __init__(self, L1, L2, Lall, data):
        self.L1 = L1
        self.L2 = L2
        self.Lall = Lall
        self.M = [[], []]  # 均值向量
        self.M[0], self.M[1] = map(self.__class_mean_vector, [data['x'][0], data['x'][1]])
        self.data_x=data['x']

    def __class_mean_vector(self, x):
        '''

        :param x:n*d
        :return:类均值向量 d*1
        '''
        return np.transpose(np.mean(x, 0, keepdims=True))

    def Pred_result(self, w, x, w0):
        '''

        :param w:投影参数d*1
        :param x: n*d
        :param w0: 0*0
        :return: 决策结果 (n,)
        '''
        w0 = np.array([w0])
        temp = np.dot(x, w)
        return temp.ravel() + w0

    def W_Direction(self):
        Sw1, Sw2 = within_class_scatter_matrix(self.data_x[0], self.M[0]), within_class_scatter_matrix(self.data_x[1], self.M[1])
        SW = pooled_within_class_scatter_matrix(Sw1, Sw2, self.L1, self.L2, self.Lall)
        W = w_direction(SW, self.M[0], self.M[1])
        return W

    def OneKey_W0(self, W):
        m1, m2 = projective_mean(W, self.M[0]), projective_mean(W, self.M[1])
        return W0(m1, m2)

    def between_class_scatter_matrix(self):
        '''

        :param m1: 第一类均值向量 d*1
        :param m2: 第二类均值向量
        :return: 类间离散度矩阵 d*d
        '''
        m_ = self.M[0] - self.M[1]

        return np.dot(m_, np.transpose(m_))
