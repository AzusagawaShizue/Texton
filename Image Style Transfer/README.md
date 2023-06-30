# Image Style Transfer

**School**: Ocean University of China

**Class**: 2023OUC Computer Vison

**Author**: 王子杰 巫恩兴

**Video**: https://b23.tv/kGIPTW3



## Background

在神经网络成为主流之前，要实现风格迁移，有一个共同的思路：首先分析一类风格的图像，然后对该类图像建立数学统计模型，最后修改输入图像使得它更符合这个统计模型。但这类方法有个缺点，即只能将特定风格场景的图片迁移到特定风格场景，因此传统方法十分受限，难以得到广泛应用。我们选择使用Leon A. Gatys等人在2015年提出的Neural-Transfer算法。

在迁移结束后我们接着对图像画质进行拔高，我们使用2023CVPR中Zero-Shot Noise2Noise: Efficient Image Denoising without any Data提到的算法进行降噪。该方法在没有成对的训练数据的情况下进行训练和测试。其基于Noise2Noise (N2N)框架，该框架通过对噪声图像对进行训练来学习去噪图像。文中将N2N框架扩展到了零样本情况，即在训练集中只包含大量的噪声图像，而没有对应的干净图像。该方法通过引入一种新的损失函数，鼓励模型在没有看到干净图像的情况下从噪声图像中学习到干净图像的映射。实验结果表明，该方法在多个基准数据集上均取得了最先进的去噪性能。

该方法的实现原理是，首先将一个大型噪声图像数据集用于训练模型，随后使用训练得到的模型对新的噪声图像进行去噪。具体来说，该方法通过对图像中的块进行随机采样，将其作为训练集中的样本，然后使用这些样本进行模型训练。训练过程中，模型仅使用噪声图像进行训练，而不需要干净图像。一旦训练完成，模型就可以应用于新的、未见过的噪声图像，从而实现零样本图像去噪。

## Introduction

该算法接收三张图片：原始图片(Input Images)，内容图片(Content Images)，风格图片(Style Images)，输出图片与Content Images内容主体相同，但又有Style Images的风格。

算法定义了两个损失：用于衡量当前图片X与Content Images在内容上的差距--内容损失Dc、与Style Images在风格上的损失Ds。通过调整X使得两个损失的加权平均值最小，经过若干次梯度下降后最终X就是算法输出。因此算法的核心在于如何计算两个损失，即如何量化两张图片的“Content”和“Style”

项目中使用VGG19这一CNN模型来提取图片语义信息，其拥有多层结构，其中每一层都用上一层的输出来提取更多特征，因此模型中的每一个卷积层都可以看做是很多个局部特征提取器。使用VGG19中的conv1_1,conv2_1,conv3_1,conv4_1,conv5_1的特征图重构图片，发现前三层图片在形状、颜色、细节上与原图差异不大；而深层网络的图片丢失了底层features，但保留了高层的语义信息。

利用这一特征我们可以把“浅层”网络的结果作为Content，而把“深层”网络的结果作为Style，最终合成Output Images，实现风格迁移。

其中内容损失计算，对于某张图片X，将其输入VGG19后，假设卷积层L有$N_L$个卷积核，特征图大小为$M_L$，其中$M_L$为特征图被“压扁”成向量以后的大小，也就是宽度×高度，且假设压扁后的特征图为$F_XL$，所以$F_XL\in{R^{N_L×M_L}}$,同理内容图在该层“压扁”特征图为$F_CL\in{R^{N_L×M_L}}$。

在计算风格损失时，由于其只保留了高层语义，并不能很好表示风格，在此引入Gram矩阵，即特征图与其转置后的矩阵做乘法得到的矩阵。

由于对每一个卷积层都可以计算内容损失和风格损失，所以可以将损失加权求得总的内容损失和风格损失，即$$L_{content}(X,C) = \sum{w_{L}L_{content}(X,C,L)}$$ $$L_{style}(X,S) = \sum{w_{L}L_{style}(X,S,L)}$$，整个网络的损失可以对其进行加权求和，所以总损失为$$L_{total}(X,D,S) = \alpha×L_{content}(X,C)+\beta×L_{style}(X,S)$$

## Key Algorithm

### 1.风格迁移部分

我们定义两个损失，一个用于内容 ($D_C$)，一个用于风格 ($D_S$)。
$D_C$ 用于衡量两张图像在内容方面的差异，而 $D_S$ 用于衡量两张图像在风格方面的差异。
然后选取第三张图像——输入图像，并将其转换为既最小化其与内容图像之间的内容损失，又最小化其与风格图像之间的风格损失。

**内容损失公式：**

$$L_{content}(\hat{y}, y, l) = \frac{1}{2}\sum_{i,j}(F_{ij}^{l} - P_{ij}^{l})^2$$

其中，$L_{content}$是内容损失，$\hat{y}$是生成的图像，$y$是目标图像，$F^{l}$是$\hat{y}$和$y$在第$l$层特征图的特征值，$P^{l}$是$y$在第$l$层特征图的特征值。

**Gram矩阵公式：**

$$G_{ij}^l=\frac{1}{C_{l}H_{l}W_{l}}\sum_{k=1}^{C_{l}}f_{ik}^{l}f_{jk}^{l}$$

其中，$G_{ij}^l$是第$l$层的Gram矩阵中第$i$行$j$列的元素，$C_{l}$是第$l$层的通道数，$H_{l}$和$W_{l}$分别是第$l$层特征图的高度和宽度，$f_{ik}^{l}$和$f_{jk}^{l}$分别是第$l$层特征图中第$k$个通道的第$i$个位置和第$j$个位置的特征值。

**风格损失公式：**

$$L_{style}(\hat{y}, y) = \sum_{l=1}^{L}\omega_l\frac{1}{4N_l^2M_l^2}\sum_{i,j}(G_{ij}^{l} - A_{ij}^{l})^2$$

其中，$L_{style}$是风格损失，$\hat{y}$是生成的图像，$y$是目标图像，$L$是特征图的层数，$\omega_l$是第$l$层特征图的权重，$N_l$和$M_l$分别是第$l$层特征图的通道数和特征图尺寸，$G^{l}$是$\hat{y}$在第$l$层特征图的Gram矩阵，$A^{l}$是$y$在第$l$层特征图的Gram矩阵。

### 2.Zero-Shot降噪部分

**损失函数**

这篇论文提出的新的损失函数被称为"Self-Supervised Noise2Noise (SSN2N) loss"，它鼓励网络从单个噪声图像中学习到干净图像的映射，而不需要真实的干净图像。该损失函数的主要思想是通过自监督学习的方式来进行去噪训练。

具体来说，该方法在每个训练迭代中，使用一个随机采样的噪声图像块作为输入，然后使用该块生成两个噪声图像，记为 A 和 B。这两个图像是从同一个噪声块中随机采样并进行扰动得到的。然后，网络的目标是学习将这两个噪声图像映射到它们对应的干净图像（即原始的噪声块）。为此，SSN2N损失函数定义为噪声图像 A 和 B 之间的均方误差（MSE）。

更具体地，设噪声块为 $X$，则噪声图像 $A$ 和 $B$ 可以表示为：

$$
A = X + N_A, \quad B = X + N_B
$$

其中，$N_A$ 和 $N_B$ 是两个独立的噪声扰动。然后，网络的目标就是最小化 $A$ 和 $B$ 之间的均方误差：

$$
\mathcal{L}_{SSN2N} = \frac{1}{2} \left\lVert f(A) - f(B) \right\rVert_2^2
$$

其中，$f(\cdot)$ 是网络的输出，表示将噪声图像映射到干净图像的函数。通过最小化 SSN2N 损失函数，网络可以学习到一个映射函数，将噪声图像映射到对应的干净图像，从而实现图像去噪。

需要注意的是，该方法的训练过程是自监督的，即网络不需要真实的干净图像作为监督信号。这使得该方法能够在没有成对训练数据的情况下进行训练和测试，并且可以在许多实际应用中进行零样本图像去噪。

如下所示：

$$\mathcal{L}_\mathrm{res.}(\theta) =\frac{1}{2}\left( \|D_1({y}) - f_{\theta}(D_1({y})) - D_2({y})\|_2^2 + \|D_2({y}) - f_{\theta}(D_2({y})) - D_1({y})\|_2^2 \right). $$

$$\mathcal{L}_\mathrm{cons.}(\theta) = \frac{1}{2} \left( \|f_{\theta}(D_1({y})) - D_1(f_{\theta}({y}))\|_2^2 + \|f_{\theta}(D_2({y})) - D_2(f_{\theta}({y}))\|_2^2 \right). $$

$$\mathcal{L}(\theta) = \mathcal{L}_\mathrm{res.}(\theta) + \mathcal{L}_\mathrm{cons.}(\theta), $$

其中 $y$ 是噪声输入图像，$D$ 是图像对下采样器，$f_\theta$ 是网络。

## Tools and Resources

**实现平台**：Pycharm+Jupyter+Anaconda

**计算资源**：AMD3060ti显卡 本地训练

**模型**：VGG19

VGG19是一种卷积神经网络模型，由牛津大学的研究团队于2014年提出，用于参加ImageNet图像分类挑战赛。它是VGGNet的一个变种，包含19个层（包括卷积层、池化层和全连接层），其中16个卷积层和3个全连接层。VGG19的架构比较简单，每个卷积层都使用了3x3的卷积核和ReLU激活函数，而池化层使用了2x2的最大池化。

VGG19优点如下：

1. 较小的卷积核和深度的网络结构可以提高模型的准确性和泛化能力。

2. VGG19的网络结构非常简单，易于理解和实现，也易于在其他任务上进行微调。

3. VGG19的网络结构非常灵活，可以通过改变网络的层数和节点数来适应不同的任务。

VGG19的创新点主要在于其采用了非常深的网络结构，并且使用了小尺寸的卷积核，这种结构可以提高模型的准确性和泛化能力。此外，VGG19还使用了多个相同大小的卷积层堆叠在一起，这种结构也可以提高模型的表达能力。

在图像风格迁移项目中，选择使用VGG19的主要原因是其较浅层的卷积层可以很好地提取图像的风格信息，而较深层的卷积层可以提取图像的内容信息。此外，VGG19的网络结构简单，易于实现，并且已经被广泛使用于计算机视觉领域，因此是一个很好的选择。

**第三方库**

```python
from __future__ import print_function #兼容python2的print格式

import os # 文件读写操作

import matplotlib.pyplot as plt # 绘图

import torch # 深度学习框架
import torch.nn as nn # 最重要模块，提供搭建神经网络的各种类以及函数
import torch.nn.functional as F # 提供非参数化函数(激活函数、池化函数、损失函数等)
import torch.optim as optim # 优化器
import torchvision.models as models # 导入VGG19参数
import torchvision.transforms as transforms # 提供常用图像变换函数，用于对图像预处理和增强
from PIL import Image # 提供函数用于图像读取、剪裁、保存、显示等

from tqdm.notebook import tqdm

```

## 实验结果

**风格迁移效果**

![](images\markdown\1.png)

**下采样，即论文中提到的图像A、B效果：**

![](images\markdown\2.png)

**降噪效果**

![](images\markdown\3.png)



## 成员分工

*王子杰*

**ContentLoss实现**

实际代码中定义了一个非严格意义上的“损失函数”，而是一个网络模块。因为如果定义为损失函数还需要实现反向传播backward函数。

接着将ContentLoss直接加在卷积层的后面，用于计算内容损失。那么在正向传播时，可以自动计算图片与内容图在该层的内容损失值。

由于Torch自动求导机制，它的梯度也会被自动计算出来(后续代码会把self.loss加起来作为损失函数)。为了让ContentLoss层不影响VGG19的正常运行，需要将模块设计为透明的，所有forward函数返回该层的输入。

计算出来的损失值被放到了loss中，将该模块加入VGG19后，对所有的ContentLoss层的Loss加权求和，即可得到总的内容损失值。

**StyleLoss实现**

风格损失模块与内容损失模块类似，需要放到VGG中也是透明的。为了计算风格损失，需要先计算Gram矩阵，它可以在一定程度上量化图片的风格。将风格使用Gram量化后，类比内容图计算风格图与输出图像的Gram矩阵的均方误差。

**重构nn.Sequential模块**

VGG模块的``features``包含卷积层和池化层，具体来说即Conv2d，ReLU，MaxPool2d，Conv2d，ReLU···这样的顺序。
为了将损失模块ContentLoss和StyleLoss插入其中，重构了一个新的nn.Sequential模块。

**模型训练归档**

测试集制作、四组results(ContenImg-StyleImg-OutputImg)训练归档、两个部分的图片可视化及其保存函数实现

**文档撰写、视频录制讲解**

代码注释、项目文档书写、演示视频录制及讲解



*巫恩兴*

**加载训练图片**

导入Content Image与Style Image，原始PIL图像介于[0,255]之间，需要将其转化为[0,1]之间的Tensor格式，可以加快在GPU上的训练速度。
其次要注意图片需要有相同的大小以及通道数，这一点在实验过程中体现为png图片带有alpha通道，即四通道图片，将其与三通道jpg格式训练时报错。
因此导入图片前需要把图片在"画图"功能打开，另存为jpeg格式，去掉透明层信息，即alpha通道，再进行训练。

**预训练模型VGG19导入**

使用torchvision即可快速导入VGG19预训练模型

此处导入的模型有两个``nn.Sequential`` 子模块： ``features`` 和 ``classifier``
其中``features``包含卷积层和池化层，``classifier``包含全连接层。
因为只在卷积层后面插入损失模块，且有些层在训练和评估模式的行为是不一样的，所以把``features``子模块切换到评估模式。

**定义迁移函数**

在每次迭代过程中，将当前图片传入网络，计算得到新的内容损失和风格损失。在损失值上调用backward函数来计算图片的梯度。优化器需要一个"closure"函数，用于重新评估得分和返回损失值。

**降噪模块实现**

采样函数、高斯、泊松噪声函数、网络搭建、训练等

## 总结感悟

实现了上述两篇论文的模型，我们对深度学习的理解更加深入了。通过实现图像风格迁移算法，了解到如何使用卷积神经网络提取图像的特征，以及如何在特征空间中匹配两幅图像的风格和内容。通过实现Zero-Shot Noise2Noise论文中的去噪模块，我们学会了如何使用深度学习模型进行图像降噪，以及如何使用残差学习的思想来训练更加高效的模型。

在实现这个实验的过程中，也遇到了一些困难和挑战。其中最主要的挑战是如何平衡风格迁移算法和去噪模块之间的关系，以得到更好的输出结果。我们尝试了多种不同的参数设置和模型结构，最终找到了一组较为理想的参数和模型结构，使得输出图像在保留原图像的风格和内容的同时，具有更高的清晰度和细节。

通过这个实验，我们深刻认识到了深度学习的潜力和局限性。深度学习模型具有非常强的表达能力，能够从数据中学习到非常复杂的模式和规律。但是，在实际应用中，我们还需要考虑模型的泛化能力、可解释性和鲁棒性等方面的问题。在未来的研究中，我希望能够更加深入地探索深度学习的本质，解决实际问题中的挑战，为计算机视觉的发展贡献一份力量。

## 参考文献

1. Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2414-2423).
2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).
3. Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2018). Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising. IEEE Transactions on Image Processing, 27(9), 3704-3719.
4. Zhao, H., Gallo, O., Frosio, I., & Kautz, J. (2018). Loss functions for image restoration with neural networks. IEEE Transactions on Computational Imaging, 3(1), 47-57.
5. Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., & Karras, T. (2018). Noise2Noise: Learning image restoration without clean data. In Proceedings of the 35th International Conference on Machine Learning (ICML) (pp. 2965-2974).
6. Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. Technical report, University of Toronto.
