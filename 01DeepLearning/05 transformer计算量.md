
[分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065?s_r=0)
# 一、模型参数量
transformer模型由 $l$ 个相同的层组成，每个层分为两部分：self-attention块和MLP块。

self-attention块的模型参数有 $Q、K、V$的权重矩阵$W_Q、W_K、W_V$和偏置，输出权重矩阵$W_O$和偏置，4个权重矩阵的形状为[h,h],4个偏置的形状为[h]。self-attention块的参数量为$4h^2+4h$。

MLP块由2个线性层组成，一般地，第一个线性层是先将维度从$h$映射到$4h$，第二个线性层再将维度从$4h$映射到$h$。第一个线性层的权重矩阵$W_1$的形状为[h,4h],偏置的形状为[4h]。第二个线性层权重矩阵$W_2$的形状为[4h,h]，偏置形状为[h]。MLP块的参数量为$8h^2+5h$。

self-attention块和MLP块各有一个layer normalization，包含了2个可训练模型参数：缩放参数$\gamma $和平移参数 $\beta$，形状都是 [h]。2个layer normalization的参数量为 $4h$。

总的，每个transformer层的参数量为 $12h^2+13h$。当隐藏维度 $h$ 较大时，可以忽略一次项，模型参数量近似为 $12lh^2$。

# 二、模型计算量
## 2.1 矩阵FLOPS
FLOPs，floating point operations，表示浮点数运算次数，衡量计算量的大小。
矩阵乘法的计算量：
对于矩阵$A\in R^{m \times n}, B \in R^{n \times p}$, $AB$相乘后结果为$C \in R^{m \times p}$。 对于矩阵$C$中的每个元素，由$A$中的每一行($1\times n$)与$B$中的每一列($n \times 1$)对应元素的乘积求和而得，总共需要 n 次乘法和（n-1）次加法,当矩阵维度较大时，我们忽略加法计算的 -1，故计算矩阵$C$的总计算量为$2mnp$。
## 2.2 Attention FLOPS
attention
$$Q=xW_Q, K = xW_k, V = xW_v \\
x_{out} = softmax(\frac{QK^T}{\sqrt{d_k}} + mask)\times V\times W_o + x
$$
在此我们只考虑单个输入，不考虑batch_size。假设输入数据维度为[s, h]。

1、计算Q、K、V， 矩阵乘法的输入输出为：[s,h]x[h,h]->[s,h]，计算量为$3\times 2sh^2$

2、$QK^T$，矩阵乘法的输入输出为:[s,h] x [h,s] ->[s,s]，计算量为 $2s^2h$

3、$scores\times V$，矩阵乘法的输入输出为：[s,s]x[s,h]->[s,h]，计算量为$2s^2h$

4、$ (...)\times W_o$, 矩阵乘法的输入输出为:[s,h]x[h,h]->[s,h]，计算量为$2sh^2$

总的计算量是$8sh^2 + 4s^2h$
## 2.3、FeedForward FLOPS
$$output = (f_{silu}(xW_1) * (xW_2)) \times W_3$$
其中，$W_1, W_3 \in R^{h \times 4h}, W_2 \in R^{4h \times h}$.

1、$xW_1$ 和 $xW_2$，矩阵乘法的输入输出是 [s,h] x [h,4h] -> [s, 4h]，计算量是$2\times 8sh^2$

2、$f_{silu}(xW_1) * (xW_2)$ 点乘的输入输出是 [s,4h] * [s,4h]->[s,4h]，计算量是$4sh$

3、$(...)\times W_3$, 矩阵乘法的输入输出是 [s,4h] x [4h,h]->[s,h]，计算量是$8sh^2$

总的计算量是$24sh^2 + 4sh$
## 2.4、logits proj
$$ output = xW_{vocab}$$
矩阵乘法的的输入输出是：[s,h]x[h,v] -> [s,v]，计算量是$2shv$.

故对于一个$l$层的模型，总的计算量为：$l * (4s^2h + 32sh^2 + 4sh) + 2shv$
