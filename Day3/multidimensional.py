# 多维数组运算
import numpy as np

A = np.array([1,2,3,4])
np.ndim(A) # 数组的维数
A.shape #数组的形状,返回值是个元组
A.shape[0]

######################################################################

# 生成了一个 3 × 2 的数组
B = np.array([[1,2], [3,4], [5,6]])
print(B)
np.ndim(B)
B.shape
# 第一个维度有3个元素，第二个维度有2个元素。另外，第一个维度对应第0维，第二个维度对应第1维（Python的索引从 0 开始）
# 数组的横向排列称为行，纵向排列称为列
# 二维数组也称为矩阵

# 矩阵的乘积是通过左边矩阵的行（横向）和右边矩阵的列（纵向）以对应元素的方式相乘后再求和而得到的

######################################################################

A = np.array([[1, 2], [3, 4]])
A.shape
B = np.array([[5, 6], [7, 8]])
B.shape
np.dot(A, B) # 乘积也称为点积，点积计算
# 矩阵的乘积运算中，操作数（A、B）的顺序不同，结果也会不同
# 矩阵 A 的第 1 维的元素个数（列数）必须和矩阵 B 的第 0 维的元素个数（行数）相等

######################################################################

A = np.array([[1, 2], [3, 4], [5, 6]])
A.shape # (3,2)
B = np.array([7,8])
B.shape # (2,)
np.dot(A, B) # A 是二维矩阵、B 是一维数组时，也要保持对应维度的元素个数一致


# 神经网络的内积
X = np.array([1, 2])
X.shape # (2,)
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)
W.shape # (2,3)
# 计算多维数组的点积
Y = np.dot(X, W) # [5, 11, 17]
