import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

x = np.arange(0, 6, 0.1)  # 以0.1为单位，生成0到6的数据
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制图形
plt.plot(x, y1, label="sin")
plt.plot(x, y2, label="cos", linestyle="--")
plt.xlabel('x')  # x轴标签
plt.ylabel('y')  # y轴标签
plt.show()

img = imread('E:/picture/1.png')  # windows路径中的\得改为/
plt.imshow(img)
plt.show()
