## 参数的更新
- 神经网络的学习的目的是找到使损失函数的值尽可能小的参数。解决这个问题的过程称为最优化（optimization）
- **随机梯度下降法（stochastic gradient descent）**，简称 SGD：将参数的梯度（导数）作为了线索。使用参数的梯度，沿梯度方向更新参数，并重复这个步骤多次，从而逐渐靠近最优参数

### SGD
- 需要更新的权重参数记为 W，把损失函数关于 W 的梯度记为xx，η 表示学习率，实际上会取 0.01 或 0.001 这些事先决定好的值。
- SGD 是朝着梯度方向只前进一定距离的简单方法
- 初始化时的参数 lr 表示 learning rate（学习率）

#### 缺点
- 解决某些问题时可能没有效率
- 如果函数的形状非均向（anisotropic），比如呈延伸状，搜索的路径就会非常低效。
- 低效的根本原因：梯度的方向并没有指向最小值的方向。

### Momentum
- 新出现了一个变量 v，对应物理上的速度，表示：物体在梯度方向上受力
- 虽然 x 轴方向上受到的力非常小，但是一直在同一方向上受力，所以朝同一个方向会有一定的加速。反过来，虽然 y 轴方向上受到的力很大，但是因为交互地受到正方向和反方向的力，它们会互相抵消，所以 y 轴方向上的速度不稳定。

### AdaGrad
- 学习率（数学式中记为 η）的值很重要。
- 学习率过小，会导致学习花费过多时间；
- 学习率过大，则会导致学习发散而不能正确进行。

* 学习率衰减（learning rate decay）的方法
- 在更新参数时，通过乘以1/sqrt(h)，就可以调整学习的尺度，参数的元素中变动较大（被大幅更新）的元素的学习率将变小

- 记录过去所有梯度的平方和。因此，学习越深入，更新的幅度就越小
- 遗忘过去的梯度，在做加法运算时将新梯度的信息更多地反映出来。
- 这种操作从专业上讲，称为“指数移动平均”，呈指数函数式地减小过去的梯度的尺度。


### Adam
- 融合了 Momentum 和 AdaGrad 的方法
- 设置 3 个超参数，一个是学习率，一个是一次momentum系数，一个是二次momentum系数（标准的设定值是 β1 为 0.9，β2 为 0.999。设置了这些值后，大多数情况下都能顺利运行）

### 总结4种方法
- 根据使用的方法不同，参数更新的路径也不同
- 一般而言，与 SGD 相比，其他 3 种方法可以学习得更快，有时最终的识别精度也更高。

## 权重的初始值
- 设定什么样的权重初始值，经常关系到神经网络的学习能否成功。

### 问题
- 梯度消失（gradient vanishing）：使用的 sigmoid 函数是 S 型函数，随着输出不断地靠近 0（或者靠近 1），它的导数的值逐渐接近 0。因此，偏向 0 和 1 的数据分布会造成反向传播中梯度的值不断变小，最后消失
- “表现力受限”的问题：如果 100 个神经元都输出几乎相同的值，那么也可以由 1 个神经元来表达基本相同的事情

### 结论
- 当激活函数使用 ReLU 时，权重初始值使用 He 初始值，
- 当激活函数为 sigmoid 或 tanh 等 S 型曲线函数时，初始值使用 Xavier 初始值。

## Batch Normalization
- 为了使各层拥有适当的广度，“强制性”地调整激活值的分布
- 可以使学习快速进行（可以增大学习率）、不那么依赖初始值（对于初始值不用那么神经质）、抑制过拟合（降低 Dropout 等的必要性）
### 操作
- 调整各层的激活值分布使其拥有适当的广度
- 向神经网络中插入对数据分布进行正规化的层，即 Batch Normalization 层（下文简称 Batch Norm 层）
- 进行使数据分布的均值为 0、方差为 1 的正规化