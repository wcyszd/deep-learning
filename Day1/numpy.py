import numpy as np

x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))

y = np.array([2.0, 4.0, 6.0])
z = x + y  # 运算
print(z)

A = np.array([[1, 2], [3, 4]])
print(A, "\n")
# print(A.shape)
print(A[0], "\n")

for row in A:
    print(row)

print("")

x = A.flatten()  # 转换为一维数组
print(x, '\n')

print(x[np.array([0, 2])], '\n')  # 获取索引为0，2的元素
