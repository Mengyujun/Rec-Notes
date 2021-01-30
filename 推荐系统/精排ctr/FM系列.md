# **FM**（因子分解机）

可以参见前深度学习CTR演化.md 文件

在线性回归模型的基础上

![img](https://pic2.zhimg.com/v2-0a7e0845c0f6cd3b8eaaf868d975487c_b.jpg)



FM的模型方程：

![img](https://pic4.zhimg.com/v2-cd7402c41afd7debf7e165da55007277_b.jpg)

FM公式计算简化： 复杂度即为 kn

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%26+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Csum_%7Bj%3Di%2B1%7D%5E%7Bn%7D%5Cleft%5Clangle%5Cmathbf%7Bv%7D_%7Bi%7D%2C+%5Cmathbf%7Bv%7D_%7Bj%7D%5Cright%5Crangle+x_%7Bi%7D+x_%7Bj%7D+%5C%5C%3D%26+%5Cfrac%7B1%7D%7B2%7D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%5Cleft%5Clangle%5Cmathbf%7Bv%7D_%7Bi%7D%2C+%5Cmathbf%7Bv%7D_%7Bj%7D%5Cright%5Crangle+x_%7Bi%7D+x_%7Bj%7D-%5Cfrac%7B1%7D%7B2%7D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Cleft%5Clangle%5Cmathbf%7Bv%7D_%7Bi%7D%2C+%5Cmathbf%7Bv%7D_%7Bi%7D%5Cright%5Crangle+x_%7Bi%7D+x_%7Bi%7D+%5C%5C%3D%26+%5Cfrac%7B1%7D%7B2%7D%5Cleft%28%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Csum_%7Bj%3D1%7D%5E%7Bn%7D+%5Csum_%7Bf%3D1%7D%5E%7Bk%7D+v_%7Bi%2C+f%7D+v_%7Bj%2C+f%7D+x_%7Bi%7D+x_%7Bj%7D-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Csum_%7Bf%3D1%7D%5E%7Bn%7D+v_%7Bi%2C+f%7D+v_%7Bi%2C+f%7D+x_%7Bi%7D+x_%7Bi%7D%5Cright%29+%5C%5C%3D%26+%5Cfrac%7B1%7D%7B2%7D+%5Csum_%7Bf%3D1%7D%5E%7Bk%7D%5Cleft%28%5Cleft%28%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+v_%7Bi%2C+f%7D+x_%7Bi%7D%5Cright%29%5Cleft%28%5Csum_%7Bj%3D1%7D%5E%7Bn%7D+v_%7Bj%2C+f%7D+x_%7Bj%7D%5Cright%29-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+v_%7Bi%2C+f%7D%5E%7B2%7D+x_%7Bi%7D%5E%7B2%7D%5Cright%29+%5C%5C%3D%26+%5Cfrac%7B1%7D%7B2%7D+%5Csum_%7Bf%3D1%7D%5E%7Bk%7D%5Cleft%28%5Cleft%28%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+v_%7Bi%2C+f%7D+x_%7Bi%7D%5Cright%29%5E%7B2%7D-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+v_%7Bi%2C+f%7D%5E%7B2%7D+x_%7Bi%7D%5E%7B2%7D%5Cright%29+%5Cend%7Baligned%7D+%5C%5C)





# FFM

FFM在FM的基础上增加了 “域”的概念，FM中一个特征只有一种隐向量的表达，FFM将特征按照事先的规则分为多个场(Field)，特征 ![[公式]](https://www.zhihu.com/equation?tex=x_i) 属于某个特定的场f。每个特征将被映射为多个隐向量 ![[公式]](https://www.zhihu.com/equation?tex=V_%7Bi1%7D%2C%E2%80%A6%2CV_%7Bif%7D) ，每个隐向量对应一个场。

当两个特征 ![[公式]](https://www.zhihu.com/equation?tex=x_i%2C+x_j) ,组合时，用对方对应的场对应的隐向量做内积:

![img](https://pic3.zhimg.com/v2-dc7400deb795c8605dd0280275b451bb_b.jpg)

其余参见 前深度学习CTR演化.md 文件



# AFM

Attentional Factorization Machines, 2017 —— 引入Attention机制的FM

参考链接- CTR预估模型发展过程与关系图谱 - 天雨粟的文章 - 知乎 https://zhuanlan.zhihu.com/p/104307718

（读论文）推荐系统之CTR预估-AFM模型 - Jesse的文章 - 知乎 https://zhuanlan.zhihu.com/p/82299967

我们知道FM模型枚举了所有的二阶交叉特征（second-order interactions），即 ![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%3D1%7D%5En%5Csum_%7Bj%3Di%2B1%7D%5En+%5Clangle+v_i%2Cv_j%5Crangle+x_i+x_j) ，实际上有一些交叉特征可能与我们的预估目标关联性不是很大；AFM就是通过Attention机制来学**习不同二阶交叉特征的重要性**（这个思路与FFM中不同field特征交叉使用不同的embedding实际上是一致的，都是通过引入额外信息来表达不同特征交叉的重要性）。



举例 

> 我们认为当性别与ad_category交叉时，重要性应该要高于性别与ad_size的交叉

FFM中通过引入Field-aware的概念来量化这种与不同特征交叉时的重要性，AFM则是通过加入Attention机制，赋予重要交叉特征更高的重要性。



## 模型结构

![img](https://pic1.zhimg.com/80/v2-280a60b1ad44bdf3b193c6b9d818ff0c_720w.jpg)



### **Pair-wise Interaction Layer:**

是对组合特征进行建模，原来的m个嵌入向量，通过**element-wise product（两两做内积）**操作得到了**m(m-1)/2个组合向量**，这些向量的维度和嵌入向量的维度相同均为k。形式化如下：

> 其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Codot) 代表element-wise的向量相乘，下同。
>
> 其中 element-wise 相乘表示 同位对应元素相乘

![img](https://pic1.zhimg.com/v2-eb2fe6583577543c08d596727f90d1c9_b.png)PI层输出

也就是说Pair-wise Interaction Layer的输入是所有嵌入向量，输出也是一组向量。输出是任意两个嵌入向量的element-wise product。任意两个嵌入向量都组合得到一个Interacted vector，所以m个嵌入向量得到m(m-1)/2个向量。



### Attention-based Pooling Layer:

Attention机制的核心思想在于：当把不同的部分压缩在一起的时候，让不同的部分的贡献程度不一样。**AFM通过在Interacted vector后增加一个weighted sum来实现Attention机制。**形式化如下：

![img](https://pic1.zhimg.com/v2-36aecc64a27c2ffa9b577ebe9e16568a_b.png)

这里aij是交互特征的Attention score，表示不同的组合特征对于最终的预测的贡献程度。可以看到：

\1. Attention-based Pooling Layer的输入是Pair-wise Interaction Layer的输出。它包含m(m-1)/2个向量，每个向量的维度是k。（k是嵌入向量的维度，m是Embedding Layer中嵌入向量的个数）

\2. Attention-based Pooling Layer的输出是一个k维向量。它对Interacted vector使用Attention score进行了weighted sum pooling（加权求和池化）操作。



### **Attention score的学习**

一个常规的想法就是随着最小化loss来学习，但是存在一个问题是：对于训练集中从来没有一起出现过的特征组合的Attention score无法学习。

**为了解决泛化问题**，引入多层感知机（MLP），这里称为Attention network，其形式化定义如下：

![img](https://pic3.zhimg.com/v2-49878ed4c93a95e7a97c3f47b11633f5_b.jpg)

可以看到，Attention network实际上是一个one layer MLP，激活函数使用ReLU，网络大小用attention factor表示，就是神经元的个数。它的输入是两个嵌入向量element-wise product之后的结果(interacted vector，用来在嵌入空间中对组合特征进行编码)；它的输出是组合特征对应的Attention score（ ![[公式]](https://www.zhihu.com/equation?tex=a_%7Bi%2Cj%7D) ）。最后，使用softmax对得到的Attention score进行规范化。



AFM在FM的二阶交叉特征上引入Attention权重，公式如下：

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D%3Dw_0%2B%5Csum_%7Bi%3D1%7D%5En+w_i+x_i%2Bp%5ET%5Csum_%7Bi%3D1%7D%5En+%5Csum_%7Bj%3Di%2B1%7D%5En+%5Calpha_%7Bij%7D%28v_i%5Codot+v_j%29x_ix_j)

**优势：**

- 在FM的二阶交叉项上引入Attention机制，赋予不同交叉特征不同的重要度，增加了模型的表达能力
- Attention的引入，一定程度上增加了模型的可解释性

**不足：**

- 仍然是一种浅层模型，模型没有学习到高阶的交叉特征





# DeepFm

