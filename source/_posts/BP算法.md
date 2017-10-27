---
title: BP算法
categories:
  - 神经网络
date: 2017-10-26 16:41:39
tags:
---

BP算法——backpropagation algorithm——反向传导算法

<!--more-->

首先介绍一下记号。

样本$ (x_1,y_1),(x_2,y_2),\ldots,(x_n,y_n) $

神经网络层数$L+1$，其中第$0$层为输入层，第$L$层为输出层，激活函数为$\sigma$。

定义$w_{ij}^l$表示连接第$l-1$层第$j$个神经元和第$l$层第$i$个神经元的权重，$b^l_{i}$ 表示第$l$层第$i$个神经元的偏置量，$z^l_i$和$a^l_i$分别表示第$i$个神经元的输入和输出。

根据定义
$$
z^l_i = \sum_j w^l_{ij}a^{l-1}_j + b^l_i \\
a^l_i = \sigma^{\prime} \left( z^l_i \right)
$$
其中$i$枚举第$l-1$层的所有神经元

对样本$i$而言，代价函数
$$
C_i(w,b)=\frac{1}{2} {\lVert y_i-a^L \rVert} ^2
$$
那么样本集的代价函数
$$
C=\frac{1}{n} \sum_{i=1}^{n} C_i
$$

所谓代价函数，就是衡量当前网络的输出与目标之间的差距，代价越小说明越接近正确，因此我们通过改变网络的参数，即$w$和$b$，来最小化$C$ 。

一个自然的想法就是梯度下降法——我们沿着梯度的方向调整参数，这样函数下降的速度最快。所谓求梯度，就是求偏导数，在这里的网络中就是求$\partial C/\partial w$和$\partial C/\partial b$。BP算法就是为了计算这个梯度而提出的。

---

定义残差
$$
\delta^l_i = \frac{\partial C}{\partial z^l_i}
$$
我们可以证明
$$
\delta^L_i = \frac{\partial C}{\partial a^L_i} \sigma^\prime(z^L_i)
$$
这是因为
$$
\begin{align}
  \delta^L_i & = \frac{\partial C}{\partial z^L_i} \\
  & = \sum_j \frac{\partial C}{\partial a^L_j} \frac{\partial a^L_j}{\partial z^L_i} \\
  & = \frac{\partial C}{\partial a^L_i} \frac{\partial a^L_i}{\partial z^L_i} \\
  & = \frac{\partial C}{\partial a^L_i} \sigma^\prime(z^L_i)
\end{align}
$$

我们接下来证明，对于$l=L-1,L-2,\ldots,1$，
$$
\delta^l_i=\sum_j w^{l+1}_{ji}\delta^{l+1}_j\sigma^{\prime}(z^l_i)
$$
因为
$$
\begin{align}
\delta^l_i & = \frac{\partial C}{\partial z^l_i} \\
& = \sum_j \frac{\partial C}{\partial z^{l+1}_j} \frac{\partial z^{l+1}_j}{\partial z^l_i} \\
& = \sum_j \frac{\partial z^{l+1}_j}{\partial z^l_i} \delta^{l+1}_j \\
& = \sum_j \frac{\partial \left( \sum_k w^{l+1}_{jk}\sigma(z^l_k)+b^{l+1}_j \right)}{\partial z^l_i} \delta^{l+1}_j \\
& = \sum_j w^{l+1}_{ji}\delta^{l+1}_j\sigma^{\prime}(z^l_i)
\end{align}
$$
至此我们就能计算出所有的残差$\delta$了，但残差只是对$z$的偏导数，我们的目标其实是对$w$和$b$的偏导数。

下面我们就开始推导。首先考虑$\partial C/\partial b^l_i$
$$
\begin{align}
\frac{\partial C}{\partial b^l_i} & = \frac{\partial C}{\partial z^l_i} \frac{\partial z^l_i}{\partial b^l_i} \\
& = \frac{\partial C}{\partial z^l_i} \\
& = \delta^l_i
\end{align}
$$

然后是$\partial C/\partial w^l_{ij}$
$$
\begin{align}
\frac{\partial C}{\partial w^l_{ij}} & = \frac{\partial C}{\partial z^l_i} \frac{\partial z^l_i}{\partial w^l_{ij}} \\
& = \delta^l_i a^{l-1}_j
\end{align}
$$

大功告成，剩下的就是计算了。可以看到，每一层的梯度是依赖于后一层的，相比于神经网络从前往后的计算，梯度的计算方向是从后往前的，这就是所谓反向传导的意思。

总结一下，也就是四个式子
$$
\delta^L_i=\frac{\partial C}{\partial a^L_i} \sigma^\prime(z^L_i) \\
\delta^l_i=\sum_j w^{l+1}_{ji}\delta^{l+1}_j\sigma^{\prime}(z^l_i) \\
\frac{\partial C}{\partial b^l_i}=\delta^l_i \\
\frac{\partial C}{\partial w^l_{ij}}=\delta^l_i a^{l-1}_j
$$

---

最后还要说一句，hexo对MathJax的支持真是糟糕……
