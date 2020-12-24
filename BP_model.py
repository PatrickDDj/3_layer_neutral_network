import random

import numpy as np


def tanh(z):
    return np.tanh(z)


def sigmod(z):
    return 1 / (1 + np.exp(-z))


# 初始化矩阵：w1, b1, w2, b2
def initialize_parameters(n_x, n_h, n_y):

    # 用正态分布随机初始化w
    # b初始化为全0
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters


# 前向传播
def forward_propagation(X, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    # 第一层input
    z1 = np.dot(w1, X) + b1
    # 第一层输出， 选取tanh作为激活函数
    a1 = tanh(z1)

    # 第二层输入
    z2 = np.dot(w2, a1) + b2
    # 第二层输出， 选取sigmod作为激活函数
    a2 = sigmod(z2)

    # 缓存当前各个结点数值
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

    # a2单独返回，可用于后续直接计算损失函数
    return a2, cache


# 计算代价函数
def cal_cost(a, Y):
    # 样本数
    m = Y.shape[1]

    # 采用交叉熵（cross-entropy）作为代价函数
    entropy = np.multiply(np.log(a), Y) + np.multiply((1 - Y), np.log(1 - a))
    cost = - np.sum(entropy) / m

    return cost


# 反向传播
def backward_propagation(parameters, cache, X, Y):
    m = Y.shape[1]
    w2 = parameters['w2']

    a1 = cache['a1']
    a2 = cache['a2']

    # 计算dw1、db1、dw2、db2
    # 连续函数求导
    dz2 = a2 - Y
    dw2 = np.dot(dz2, a1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m

    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = np.dot(dz1, X.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

    return grads


# 更新参数
def update_parameters(parameters, grads, learning_rate=0.005):
    # 获取原来矩阵以及优化梯度
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    # 按照相应学习率更新
    w1 = w1 - dw1 * learning_rate
    b1 = b1 - db1 * learning_rate
    w2 = w2 - dw2 * learning_rate
    b2 = b2 - db2 * learning_rate

    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters


# 建立神经网络
def fit(X_train, Y_train, n_x, n_h, n_y, iterations=10000, learning_rate=0.005):
    # 初始化矩阵参数
    parameters = initialize_parameters(n_x, n_h, n_y)

    # 训练
    for i in range(0, iterations):
        # 前向传播
        a2, cache = forward_propagation(X_train, parameters)
        # 计算损失函数
        cost = cal_cost(a2, Y_train)

        # 反向传播
        grads = backward_propagation(parameters, cache, X_train, Y_train)

        # 更新矩阵参数
        parameters = update_parameters(parameters, grads, learning_rate)

        # 每500次打印损失函数信息
        if i % 2000 == 0:
            print('第%i次cost：%f' % (i, cost))

    return parameters


# 评估
def evaluate(parameters, X_test, Y_test):
    a2, _ = forward_propagation(X_test, parameters)

    # 标签数目
    m = Y_test.shape[0]
    # 测试集数目
    n = Y_test.shape[1]

    correct = 0
    for i in range(n):
        # 获取当前各个标签概率的数组
        possibility_array = a2[:, i]

        # 选取最大的概率
        max_p = max(possibility_array)

        # 若概率最大的标签和测试集一致，那么说明预测成功
        for k in range(m):
            if possibility_array[k] == max_p:
                if Y_test[k, i] == 1:
                    correct += 1

    accuracy = correct / n * 100
    print('准确率：%.2f%%' % accuracy)




def load_Iris(p=0.8):
    f = open('data/Iris.csv', 'r')

    content = f.read().split('\n')

    feature_names = content[0].split(',')[1:-1]

    content = content[1:-1]
    random.shuffle(content)

    n = len(content)
    m = len(feature_names)

    X = np.zeros((n, m))
    Y = np.zeros((n, 3))

    f = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }

    for i in range(n):
        line = content[i].split(',')
        line.pop(0)
        Y[i][f[line[-1]]] = 1

        line.pop(-1)

        X[i] = line
    s = int(p * n)

    return X[0:s].T, Y[0:s].T, X[s:-1].T, Y[s:-1].T


def load_zoo(p=0.8):
    f = open('data/zoo.csv', 'r')

    content = f.read().split('\n')

    feature_names = content[0].split(',')[1:-1]

    content = content[1:-1]
    random.shuffle(content)

    n = len(content)
    m = len(feature_names)

    X = np.zeros((n, m))
    Y = np.zeros((n, 7))


    for i in range(n):
        line = content[i].split(',')
        line.pop(0)
        Y[i][int(line[-1])-1] = 1
        line.pop(-1)
        X[i] = line
    s = int(p * n)

    return X[0:s].T, Y[0:s].T, X[s:-1].T, Y[s:-1].T

