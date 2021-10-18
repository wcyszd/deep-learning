import numpy as np

# 与门
# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = x1 * w1 + x2 * w2
#     if tmp <= theta:
#         return 0
#     elif tmp > theta:
#         return 1


# 与门 与门仅在两个输入均为 1 时输出 1，其他时候则输出 0
def AND(x1, x2):
    x = np.array([x1, x2]) # w1 和 w2 是控制输入信号的重要性的参数
    w = np.array([0.5, 0.5]) # 偏置是调整神经元被激活的容易程度（输出信号为 1 的程度）的参数
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

AND(0, 0)
AND(1, 0)
AND(0, 1)
AND(1, 1)


# 与非门：与非门就是颠倒了与门的输出，当 x1 和 x2 同时为 1 时输出 0，其他时候则输出 1
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5]) # 仅权重和偏置与AND不同！
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 或门 只要有一个输入信号是 1，输出就为 1
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5]) # 仅权重和偏置与AND不同！
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# 异或门（逻辑异或电路）当 x1 或 x2 中的一方为 1 时，才会输出 1（另外一方是0）
# 感知机（线性空间分割）不能实现异或门（单层感知机无法表示异或门，单层感知机无法分离非线性空间）
# 这样的曲线分割而成的空间称为非线性空间，由直线分割而成的空间称为线性空间。
# 组合与门、或门、与非门可以实现

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

XOR(0, 0) # 输出0
XOR(1, 0) # 输出1
XOR(0, 1) # 输出1
XOR(1, 1) # 输出0

# 异或门是一种多层结构的神经网络（2 层感知机，拥有权重的层实质上只有 2 层）