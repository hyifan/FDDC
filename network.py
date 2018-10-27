# -- coding: utf-8 --
import random
import numpy as np
import math
import json

def main(training_data, test_data, epochs, mini_batch_size, eta, lmbda, sizes):
    # net = Network([912, num, 1])
    net = Network(sizes)
    net.SGD(training_data, epochs, mini_batch_size, eta, lmbda, test_data)
    return net.weights, net.biases


class Network(object):

    def __init__(self, sizes):
        # numpy.random.randn()是从标准正态分布中返回一个或多个样本值
        # 使用独⽴⾼斯随机变量来选择权重和偏置，其被归⼀化为均值为0，标准差1
        self.num_layers = len(sizes)
        self.sizes = sizes                                       # 接收参数[特征长度,第二层神经元数量,1]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # 初始化第1层到第2层，第2层到第3层网络的偏置
        # 初始化第1层到第2层，第2层到第3层网络的权重
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        # 对于每一个输入a，使用训练好的b和w，计算其输出
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda, test_data):
        '''
        主函数，接受6个参数
        training_data 训练数据，格式为二维数组，[0][0]长度为特征长度，[0][1]为label值
        epochs 迭代期数量
        mini_batch_size 采样时的⼩批量数据的 ⼤⼩
        eta 速率
        lmbda 规范化参数
        '''
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)                         # 打乱训练数据的顺序
            mini_batches = [                                      # 生成一个二维数组，包含training_data[k:k+mini_batch_size]
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                # 对生成的每一个小批量数据进行梯度计算
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print(j, self.evaluate(test_data))

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        '''
        update_mini_batch 的⼯作是对 mini_batch 中的每⼀个训练样本计算梯度，然后适当地更新 self.weights 和 self.biases。
        在每个迭代期，它⾸先随机地将训练数据打乱，然后将它分成多个适当⼤⼩的⼩批量数据。
        这是⼀个简单的从训练数据的随机采样⽅法。
        然后对于每⼀个 mini_batch 我们应⽤⼀次梯度下降。
        它仅仅使⽤ mini_batch 中的训练数据，根据单次梯度下降的迭代更新⽹络的权重和偏置。
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # 循环小批量数据中的mini_batch_size条数据，根据每条数据计算 delta_nabla_b, delta_nabla_w
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # zip()将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表，元素个数与最短的列表一致
            # 将每行数据执行后得到的梯度b相加
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            # 将每行数据执行后得到的梯度w相加
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # 新w＝旧w－(相加后的梯度w/len＋lmbda\n*旧w)*eta
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.weights = [w-(eta * lmbda /len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        # 新b＝旧b－(相加后的梯度b/len)*eta
        self.biases = [b-(eta / len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        '''
        反向传播算法，用于快速计算代价函数梯度
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in list(zip(self.biases, self.weights)):
            # 前向传播，计算Zl
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            # 前向传播，计算al=sigmoid(Zl)
            activations.append(activation)

        # 根据反向传播第1个方程计算输出层误差：二次代价函数C对于a的偏导数 * sigmoid(z)的导数
        delta = self.cost_derivative(activations[-1], y[0], y[1]) * sigmoid_prime(zs[-1])
        # 根据反向传播第3个方程计算b的梯度为误差值
        nabla_b[-1] = delta
        # 根据反向传播第4个方程计算w的梯度为al-1*误差值
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            # 反向传播，根据反向传播第2个方程计算各层误差
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(self.feedforward(x), y) for (x, y) in test_data]
        res = sum([cost_E(y[0], y[1], x[0]) for (x, y) in test_results]) / len(test_data)
        return res

    def cost_derivative(self, output_activations, label, market):
        # 计算二次代价函数C对于a的偏导数
        res = math.log(max(market, 2), 2) * (output_activations - label)
        return res

def cost_E(label, market, answer):
    if (label == 0):
        a = 0.8
    else:
        a = min(abs(answer/label - 1), 0.8)
    return  a * math.log(max(market, 2), 2)


def sigmoid(z):
    # 激活函数
    return np.log(1 + np.exp(z))


def sigmoid_prime(z):
    # 计算sigmoid(z)的导数
    return 1.0/(1.0 + np.exp(-z))


def feedforward2(a, ws, bs):
    # 对于每一个输入a，使用训练好的b和w，计算其输出
    for b, w in zip(bs, ws):
        a = sigmoid(np.dot(w, a)+b)
    return a


def get_result(answer_data, sys, ws, bs, fileName):
    length = len(answer_data[0])
    data = {}
    for i in range(0, len(sys)):
        data[sys[i]] = round(feedforward2(np.array(answer_data[i]).reshape((length, 1)), ws, bs)[0][0] / 1000000, 2)
    fw = open(fileName + '.json', 'w', encoding='utf-8')
    json.dump(data, fw, ensure_ascii = False, indent = 4)
