import numpy as np
# import matplotlib.pyplot as plt

# 生成数组
x = np.array([1.0, 2.0, 3.0])
print(x)

type(x)

# 算术运算
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

x + y

x / 2.0

# N维数组
A = np.array([[1, 2], [3, 4]])
print(A)
A.shape # 矩阵 A 的形状可以通过 shape 查看
A.dtype # 矩阵元素的数据类型可以通过 dtype 查看

B = np.array([[3, 0], [0, 6]])
A + B
A * B

# 广播（标量扩展）
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
A * B

X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)

X[0]
X[0][1]

for row in X:
    print(row)

X = X.flatten()
print(X)

X[np.array([0, 2, 4])]
# array([51, 14, 0])

X > 15
X[X > 15]