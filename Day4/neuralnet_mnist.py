"""
实现神经网络的推理处理
评价它的识别精度（accuracy），即能在多大程度上正确分类。

神经网络的输入层有784个神经元，输出层有 10 个神经元。
输入层的 784 这个数字来源于图像大小的 28 × 28 = 784
输出层的 10 这个数字来源于 10 类别分类（数字 0 到 9，共 10 类别）
这个神经网络有 2 个隐藏层，第 1 个隐藏层有 50 个神经元，第 2 个隐藏层有 100 个神经元
这个 50 和 100 可以设置为任何值
"""

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import pickle
from Day4.softmax import softmax
from PIL import Image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=False)
    return x_test, t_test

# 读入保存在pickle文件sample_weight.pkl中的学习到的权重参数
def init_network():
    # 以字典变量的形式保存了权重和偏置参数
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

# 以 NumPy 数组的形式输出各个标签对应的概率
# 比如输出 [0.1, 0.3, 0.2, ..., 0.04] 的数组，该数组表示“0”的概率为 0.1，“1”的概率为 0.3，等等
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

if __name__ == "__main__":
    # 获得 MNIST 数据集，生成网络
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y) # 获取概率最高的元素的索引，作为这一组数据的预测结果
        # 比较神经网络所预测的答案和正确解标签，将回答正确的概率作为识别精度。
        if p == t[i]:
            accuracy_cnt += 1

    print("Accuracy：" + str(float(accuracy_cnt) / len(x)))
    # Accuracy:0.9352  表示有 93.52 % 的数据被正确分类

    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # 输出神经网络各层的权重的形状
    print(x.shape) # (10000, 784)
    print(x[0].shape) # (784,)
    print(W1.shape) # (784, 50)
    print(W2.shape) # (50, 100)
    print(W3.shape) # (100, 10)

    # 基于批处理的代码实现
    batch_size = 100

    for i in range(0, len(x), batch_size): #以100为步长
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        # axis=1 这指定了在100 × 10的数组中，沿着第1维方向（以第1维为轴）找到值最大的元素的索引（第0维对应第1个维度）
        # 矩阵的第0维是列方向，第1维是行方向
        p = np.argmax(y_batch, axis=1)
        # 较一下以批为单位进行分类的结果和实际的答案
        # 生成由 True/False 构成的布尔型数组，并计算 True 的个数
        accuracy_cnt += np.sum(p == t[i:i+batch_size])