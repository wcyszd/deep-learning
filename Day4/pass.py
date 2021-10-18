'''
各层间信号传递的实现
'''
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 用 NumPy 多维数组来实现式
# A = XW + B
# 从输入层到隐藏层1
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1 # [0.3 0.7 1.1]


# ------------------------------------------

Z1 = sigmoid(A1)
print(Z1) # [0.57444252, 0.66818777, 0.75026011]

# 实现从第1层到第2层的信号传递
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
print(Z1.shape)
print(W2.shape)
print(B2.shape)
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

# 从第2层到输出层的信号传递

# 定义一个恒等函数，将其作为输出层的激活函数
def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)