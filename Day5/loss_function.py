import numpy as np
import matplotlib.pylab as plt


# 均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

# 交叉熵误差
def cross_entropy_error(y, t):
    delta = 1e-7 # 微小值
    return -np.sum(t * np.log(y + delta))
# 因为，当出现 np.log(0) 时，np.log(0) 会变为负无限大的 -inf，这样一来就会导致后续计算无法进行。
# 保护性对策，添加一个微小值可以防止负无限大的发生




# if __name__ == '__main__':

# 设2为正确解
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# “2”的概率最高的情况（0.6）0.0975
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
mean_squared_error(np.array(y1), np.array(t))


# 例2：“7”的概率最高的情况（0.6）0.5975
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
mean_squared_error(np.array(y2), np.array(t))

# "2"的概率最高 0.5108
cross_entropy_error(np.array(y1), np.array(t))

# "7"的概率最高 2.3026
cross_entropy_error(np.array(y2), np.array(t))


