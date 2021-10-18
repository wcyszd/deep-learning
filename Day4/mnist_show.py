'''
load_mnist
返回值格式：( 训练图像, 训练标签 )，( 测试图像, 测试标签 )
load_mnist(normalize=True, flatten=True, one_hot_label=False)
三个参数：
    normalize：是否将输入图像正规化为 0.0～1.0 的值
        False（输入图像的像素保持原来的0~255）
    flatten：是否展开输入图像（变成一维数组）
        False 输入图像为 1 × 28 × 28 的三维数组
        True 输入图像会保存为由 784 个元素构成的一维数组
    one_hot_label:pickle 是否将标签保存为 one-hot 表示(one-hot representation)
        True 表示是仅正确解标签为 1，其余皆为 0 的数组，就像 [0,0,1,0,0,0,0,0,0,0]
        False 只是像 7、2 这样简单保存正确解标签

'''
import sys, os
sys.path.append(os.pardir) # .. 上级目录 （父目录加入sys.path - python的搜索模块路径集）
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    # 需要把保存为NumPy数组的图像数据转换为PIL用的数据对象（转换处理由Image.fromarray()来完成）
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
img = x_train[0]
label = x_train[0]
print(label) # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)   # 把图像的形状变成原来的尺寸(转换为指定期望的形状)
print(img.shape)  # (28, 28)

img_show(img)