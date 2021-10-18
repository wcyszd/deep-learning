"""
激活函数：
1.阶跃函数
2.sigmoid函数

"""
import numpy as np
import matplotlib.pylab as plt

# 生成一个NumPy数组
x = np.array([-1.0, 1.0, 2.0])
# 对数组进行不等号运算, 数组的各个元素都会进行不等号运算，生成一个布尔型数组
y = x > 0
# stype()方法转换NumPy数组的类型, 把数组y的元素类型从布尔型转换为 int 型
y = y.astype(np.int)

# 阶跃函数生成
def step_function(x):
    return np.array(x > 0, dtype=np.int)

# 在-5.0到5.0的范围内，以0.1为单位，生成NumPy数组（[-5.0, -4.9,…, 4.9]）
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

# 阶跃函数以0为界，输出从0切换为1，值呈阶梯式变化
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # 指定y轴范围
plt.show()


# sigmoid函数：公式 h(x)=1/(1+exp(-x))
# exp(-x)表示e-x，e是纳皮尔常数2.7182 ...
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
# numpy有广播功能，可以和标量做计算
sigmoid(x)

# 画sigmoid函数图像
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# sigmoid平滑曲线、有连续性
# 感知机中神经元之间流动的是0或1的二元信号，而神经网络中流动的是连续的实数值信号
# 输入小时，输出接近 0（为 0）；随着输入增大，输出向 1 靠近（变成 1）
# 非线性函数，sigmoid 函数是一条曲线，阶跃函数是一条像阶梯一样的折线

# ReLU（Rectified Linear Unit）函数
# 数在输入大于0时，直接输出该值；在输入小于等于0时，输出0
def relu(x):
    return np.maximum(0, x) # 选择较大值输出

# 画ReLU函数图像
x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.show()