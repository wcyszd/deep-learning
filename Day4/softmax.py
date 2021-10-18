'''

'''
import numpy as np

a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a) #指数函数
print(exp_a)

sum_exp_a = np.sum(exp_a) #指数函数的和
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y) # [ 0.01821127  0.24519181  0.73659691]

# 在计算机的运算上有一定的缺陷, 缺陷就是溢出问题, 此时指数函数的值很容易变得非常大
# 计算机处理“数”时，数值必须在 4 字节或 8 字节的有限数据宽度内。这意味着数存在有效位数，也就是说，可以表示的数值范围是有限的。因此，会出现超大值无法表示的问题。这个问题称为溢出
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# a = np.array([1010, 1000, 990])
# np.exp(a) / np.sum(np.exp(a)) 无法正常计算，会溢出

# c = np.max(a)
# np.exp(a - c) / np.sum(np.exp(a - c))
# 通过减去输入信号中的最大值（上例中的 c），我们发现原本为 nan（not a number，不确定）的地方，现在被正确计算了

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


if __name__ == '__main__':
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(y) # [ 0.01821127  0.24519181  0.73659691]
    # y[0]的概率是0.018（1.8 %），y[1] 的概率是 0.245（24.5 %），y[2] 的概率是 0.737（73.7 %）
    # 因为第 2 个元素的概率最高，所以答案是第 2 个类别
    # 有 74 % 的概率是第 2 个类别，有 25 % 的概率是第 1 个类别，有 1 % 的概率是第 0 个类别
    # 过使用 softmax 函数，我们可以用概率的（统计的）方法处理问题。
    np.sum(y) # 1.0


