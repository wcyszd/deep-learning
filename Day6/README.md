## 数值微分
### 导数
- 利用微小的差分求导数的过程称为数值微分（numerical differentiation）
- 基于数学式的推导求导数的过程，则用“解析性”（analytic）一词，称为“解析性求解”或者“解析性求导”。(基本不含误差)

### 偏导数
- 有多个变量的函数的导数称为偏导数
- 偏导数和单变量的导数一样，都是求某个地方的斜率

### 梯度（gradient）
- 由全部变量的偏导数汇总而成的向量
- 神经网络必须在学习时找到最优参数（权重和偏置）。这里所说的最优参数是指损失函数取最小值时的参数。
- 梯度表示的是各点处的函数值减小最多的方向，因此，无法保证梯度所指的方向就是函数的最小值或者真正应该前进的方向
- 虽然梯度的方向并不一定指向最小值，但沿着它的方向能够最大限度地减小函数的值
- ![image](https://user-images.githubusercontent.com/45821845/139241067-7ff90ec1-0602-45c0-8b4a-ffa27c21760f.png)
- 要让loss函数最小，就求loss函数对w，b的梯度，然后来对w，b更新
- ![image](https://user-images.githubusercontent.com/45821845/139248788-f77334f3-52f9-4f9d-8c19-286d715faa18.png)


### 学习步骤
- 前提
- 神经网络存在合适的权重和偏置，调整权重和偏置以便拟合训练数据的
- 过程称为“学习”。神经网络的学习分成下面4个步骤。

- 步骤1（mini-batch）
- 从训练数据中随机选出一部分数据，这部分数据称为mini-batch。我们
- 的目标是减小mini-batch的损失函数的值。

- 步骤2（计算梯度）
- 为了减小mini-batch的损失函数的值，需要求出各个权重参数的梯度。
- 梯度表示损失函数的值减小最多的方向。

- 步骤3（更新参数）
- 将权重参数沿梯度方向进行微小更新。
