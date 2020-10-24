# 谷歌、阿里、微软等10大深度学习CTR模型最全演化图谱【推荐、广告、搜索领域】

![img](https://pic1.zhimg.com/80/v2-763b523bd17349cd6cfecae2765db3d5_720w.jpg)



先记录几个常用的，简历中，以及提及到的模型

wide&deep dcn deepfm fm ffm  deepMCP+mmoe  双塔dnn 



从以下3个方面来考虑深度学习CTR的不同模型

**“记忆与扩展”、“类别特征”和“特征交叉”** 

参见- 看Google如何实现Wide & Deep模型(1) - 石塔西的文章 - 知乎 https://zhuanlan.zhihu.com/p/47293765

> 相比于实数型特征，**稀疏的类别/ID类特征，才是推荐、搜索领域的“一等公民”**，被研究得更多。即使有一些实数值特征，比如历史曝光次数、点击次数、CTR之类的，也往往通过**bucket**的方式，变成categorical特征，才喂进模型。
>
> 
>
> 推荐、搜索喜欢稀疏的类别/ID类特征，我觉得有三方面的原因：
>
> - LR, DNN在底层还是一个线性模型，但是现实生活中，**标签y与特征x之间较少存在线性关系，而往往是分段的**。以“点击率~历史曝光次数”之间的关系为例，之前曝光过1、2次的时候，“点击率~历史曝光次数”之间一般是正相关的，再多曝光1、2次，用户由于好奇，没准就点击了；但是，如果已经曝光过8、9次了，由于用户已经失去了新鲜感，越多曝光，用户越不可能再点，这时“点击率~历史曝光次数”就表现出负相关性。因此，categorical特征相比于numeric特征，更加符合现实场景。
> - 推荐、搜索一般都是基于用户、商品的标签画像系统，而标签天生就是categorical的
> - 稀疏的类别/ID类特征，可以**稀疏地存储、传输、运算，提升运算效率**。



## wide&deep

一句话概括： **W&D由浅层（或单层）的Wide部分神经网络和深层的Deep部分多层神经网络组成，输出层采用softmax或logistics regression综合Wide和Deep部分的输出。**



模型图：

![img](https://pic1.zhimg.com/v2-a203aa626f77d0510bbffa5535c34d7d_b.jpg)



**优点：**

**Wide**部分有利于增强模型的“**记忆能力**”，**Deep**部分有利于增强模型的“**泛化能力**”

Wide部分只需补充Deep模型的缺点，即记忆能力，这部分主要通过**小规模的交叉特征**实现



细节：

1. **使用带L1正则化项的FTRL作为wide部分的优化方法，而使用AdaGrad作为deep部分的优化方法** 

   - **FTRL with L1非常注重模型的稀疏性**，即想让wide部分变得更加稀疏，即L1 FTRL会让Wide部分的大部分权重都为0，压缩了模型权重，也压缩了特征向量的维度
   - Deep部分呢： 输入主要是**数值型特征** or 已经**降维并稠密化的Embedding向量**, 不会有过度稀疏的特征向量， 即不存在严重的**特征稀疏问题**， **深度学习模型不需要稀疏性，稀疏性太强影响效果**

2. 记忆能力和泛化能力如何理解？

   - 记忆能力 指 “直接的”、“暴力的”、“显然的”关联规则的能力 ------即很容易理解强关联的规则
   - 泛化能力 指 所有特征交给网络去学习是否关联， 泛化成一些间接的，可能的相关性
   - 其实分别对应了 推荐结果**准确性（记忆能力）和扩展性（泛化能力）**

3. 两部分的输入有何区别呢 

   - wide侧说白了就是记忆用的，还是**one-hot**更合适，能够很明确的告诉模型去记忆什么  --**one-hot 后的离散型特征和等频分桶后的连续性特征**
   - wide侧肯定是**独热输入**，embedding其实就是一层全连接神经网络
   - Deep侧  可以泛化学习到样本中多个特征之间与目标看不到的潜在关联-- **Embedding 后的离散型特征和归一化后的连续型特征**

4. wide部分具体

   wide部分主要就是一个**广义线性模型（如LR）**，具体模型如下：

   ![img](https://pic2.zhimg.com/v2-863d3a42105cebc322937f86eba5dec2_b.jpg)Wide部分模型

   这里值得注意的是，这部分的输入特征包括两部分，一部分是原始的输入特征，另一部分是交叉特征，

   **Wide侧还需要人工构造特征交叉**（deepfm对wide&deep的巨大改进， 不需要再手动交叉）

   其中一种方式就是cross-product，具体定义如下：

   ![img](https://pic3.zhimg.com/v2-c158203bee5245b4e967f574b055d1fd_b.jpg)

5. deep部分具体

   Deep部分简单理解就是Embedding+MLP这种非常普遍的结构

   具体模型结构如下所示：

   ![img](https://pic2.zhimg.com/v2-93c215a506c1a52580611742afc9a1d3_b.jpg)Deep部分模型

   从上图的网络结构可以看出，中间隐藏层的激活函数都是ReLu，最后一层的激活函数是sigmoid

6. 训练过程为联合训练

   即同时训练wide 和deep部分的模型参数，wide部分主要采用带L1正则的FTRL算法进行优化,deep部分采用AdaGrad进行优化 

   好处如下：

   - 整体最优化
   - 有效降低整个网络大小

7. 




## DeepFm

DeepFM模型是将Wide部分替换为了FM模型，增强了模型的低阶特征交互的能力

模型图：

![img](https://pic2.zhimg.com/v2-e668f7ba05aad55f684e06af5a19a45d_b.jpg)

目标是**共同学习低阶和高阶特征交互**



### wide部分

即为FM模型 



### **Deep部分**

Deep部分是一个前向传播的神经网络，用来学习高阶特征交互。



### **Output层**

FM层与Deep层的输出相拼接，最后通过一个逻辑回归返回最终的预测结果：

![[公式]](https://www.zhihu.com/equation?tex=%5Chat+y%3Dsigmoid%28y_%7BFM%7D%2By_%7BDNN%7D%29+%5C%5C)

DeepFM模型的优势：

1. 端到端；
2. Wide部分与Deep部分共享参数；
3. 无需手动设计特征工程（相对于w&d中 wide部分需要手动设计交叉特征）



### deepfm常见面试问题：

1. deepfm的wide & deep部分的输入是什么呢，分别输入什么特征好

   - FM 和 Deep 部分**共享Embedding 层**，FM 训练得到的参数既作为 wide 部分的输出，也作为 MLP 部分的输入 
   - 一方面是和wide&deep一样的情况 还是可以考虑 对离散型特征 和 连续型特征分开讨论  

2. dnn可以学到特征交叉，为什么还需要fm侧 

   参考-DNN可以进行高阶特征交互，为什么Wide&Deep和DeepFM等模型仍然需要显式构造Wide部分？ - 王鸿伟的回答 - 知乎 https://www.zhihu.com/question/364517083/answer/961734991 

   - 从wide和deep的 记忆能力和泛化能力展开 

   - 理论上来说DNN可以拟合任意函数，实际从3个方面来提高拟合能力：

     - 一是根据具体的问题场景提出更好的模型，这些特定的模型会比较适合各自的场景，从而降低拟合难度，比如各类CNN，RNN，GNN等。
     - 二是提出更好的优化方法，使得学习拟合的过程更快速高效。
     - 三是提取更好的特征，从而在源头上降低拟合的难度，提升性能的上限。

     在推荐系统中，特征提取的影响非常大，因此显示的选择一些特征，可以降低拟合难度

   - Wide&Deep和FM里手动构造二阶项，也只是想为DNN提供更多的“**输入素材”**，让DNN可以更好地发挥而已。这些二阶项能不能真的提高模型性能，并没有理论上的保证，因为这取决于具体的推荐系统场景中是否真的有很多这种二阶相关性



## DCN

 Wide & Deep工作的一个后续研究， 将Wide部分替换为由特殊网络结构实现的Cross，**自动构造有限高阶的交叉特征**



背景：

w&d, 在Wide部分，仍然需要人工地设计特征叉乘

deepfm, FM可以自动组合特征，但也仅限于二阶叉乘

dcn, **自动学习高阶的特征组合**



网络模型：

模型的结构非常简洁，从下往上依次为：Embedding和Stacking层、Cross网络层与Deep网络层并列、输出合并层，得到最终的预测结果。

![img](https://pic4.zhimg.com/80/v2-d80939a10d6dde95859fa1f13866f02f_720w.jpg)



### **Embedding and Stacking Layer**

Embedding操作将高维稀疏特征转化为低维密集型特征 

然后stacking（联合操作） ： 将密集型【连续】特征和embedding之后的特征进行stacking



### **Cross Network**

**目的**是增加特征之间的交互力度



假设第![[公式]](https://www.zhihu.com/equation?tex=l)层交叉层的输出向量为![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_l+)，那么对于第![[公式]](https://www.zhihu.com/equation?tex=l%2B1)层交叉层输出向量![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_%7Bl%2B1%7D)为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_%7Bl%2B1%7D%3D%5Cmathbf%7Bx%7D_0%5Cmathbf%7Bx%7D_%7Bl%7D%5ET%5Cmathbf%7Bw%7D_l%2B%5Cmathbf%7Bb%7D_l%2B%5Cmathbf%7Bx%7D_l%3Df%28%5Cmathbf%7Bx%7D_l%2C%5Cmathbf%7Bw%7D_l%2C%5Cmathbf%7Bb%7D_l%29%2B%5Cmathbf%7Bx%7D_l+%5C%5C)

其中![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_%7Bl%7D%2C%5Cmathbf%7Bx%7D_%7Bl%2B1%7D+%5Cin+%5Cmathbb%7BR%7D%5Ed)是第![[公式]](https://www.zhihu.com/equation?tex=l)和![[公式]](https://www.zhihu.com/equation?tex=l%2B1)交叉层的输出向量，![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bw%7D_l%2C%5Cmathbf%7Bb%7D_l%5Cin+%5Cmathbb%7BR%7D%5Ed)是权重参数和偏置

其中x0 即为最开始的输入

![img](https://picb.zhimg.com/80/v2-80bc51c7906c1f10f514b851afbcc190_720w.jpg)

举例：令![[公式]](https://www.zhihu.com/equation?tex=x_0%3D%5Bx_0%5E1%2Cx_0%5E2%5D%5ET)，![[公式]](https://www.zhihu.com/equation?tex=b_i%3D0)则：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Barray%7D%7Bl%7D%5C%5C+%5Cmathbf%7Bx%7D_1+%26%3D%5Cmathbf%7Bx%7D_0%2A%5Cmathbf%7Bx%7D_0%5ET%2A%5Cmathbf%7Bw%7D_0%2B%5Cmathbf%7Bx%7D_0%5C%5C+%26%3D%5Bx_0%5E1%2Cx_0%5E2%5D%5ET%2A%5Bx_0%5E1%2Cx_0%5E2%5D%2A%5Bw_0%5E1%2Cw_0%5E2%5D%2B%5Bx_0%5E1%2Cx_0%5E2%5D%5ET%5C%5C+%26%3D%5Bw_0%5E1%28x_0%5E1%29%5E2%2Bw_0%5E2x_0%5E1x_0%5E2%2Bx_0%5E1%2Cw_0%5E1x_0%5E1x_0%5E2%2Bw_0%5E2%28x_0%5E2%29%5E2%2Bx_0%5E2%5D%5ET%5C%5C+%5Cmathbf%7Bx%7D_2+%26%3D%5Cmathbf%7Bx%7D_0%2A%5Cmathbf%7Bx%7D_1%5ET%2A%5Cmathbf%7Bw%7D_1%2B%5Cmathbf%7Bx%7D_1%5C%5C+%26%3D%5Bw_1%5E1x_0%5E1x_1%5E1%2Bw_1%5E2x_0%5E1x_1%5E2%2Bx_1%5E1%2Cw_1%5E1x_0%5E2x_1%5E1%2Bw_1%5E2x_0%5E2x_1%5E2%2Bx_1%5E2%5D%5ET%5C%5C+%26%3D%5Bw_1%5E1x_0%5E1%28w_0%5E1%28x_0%5E1%29%5E2%2Bw_0%5E2x_0%5E1x_0%5E2%2Bx_0%5E1%29%2Bw_1%5E2x_0%5E1%28w_0%5E1x_0%5E1x_0%5E2%2Bw_0%5E2%28x_0%5E2%29%5E2%2Bx_0%5E2%29%2Bw_0%5E1%28x_0%5E1%29%5E2%2Bw_0%5E2x_0%5E1x_0%5E2%2Bx_0%5E1%2C......%5D%5ET+%5Cend%7Barray%7D+%5C%5C)

相对于FM的二阶特征交叉，DCN可以构建高阶的特征交互，阶数由网络深度决定，并且交叉网络的参数只依据输入的维度线性增长。

**Cross网络带来的优势：**

1.  **有限高阶**：叉乘**阶数由网络深度决定**，深度 ![[公式]](https://www.zhihu.com/equation?tex=L_c) 对应最高 ![[公式]](https://www.zhihu.com/equation?tex=L_c%2B1+) 阶的叉乘  

    ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_1)中包含了包含了所有的![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_0)的1、2阶特征交互，![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_2)包含了所有![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_0%2C%5Cmathbf%7Bx%7D_1)的1、2、3阶特征交互，那么![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_%7Bl%2B1%7D)包含了所有的![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_0%2C%5Cmathbf%7Bx%7D_1%2C...%2C%5Cmathbf%7Bx%7D_%7Bl%7D)的![[公式]](https://www.zhihu.com/equation?tex=1%EF%BD%9El%2B2)阶特征交互

2. **自动叉乘**：Cross输出包含了原始特征从一阶（即本身）到 ![[公式]](https://www.zhihu.com/equation?tex=L_c%2B1+) 阶的**所有叉乘组合，**而模型参数量仅仅随输入维度成**线性增长**： ![[公式]](https://www.zhihu.com/equation?tex=2%2Ad%2AL_c)

3. **参数共享**：不同叉乘项对应的权重不同，但并非每个叉乘组合对应独立的权重（指数数量级）， 通过参数共享，Cross有效**降低了参数量**。

> 这里有一点很值得留意，前面介绍过，文中将dense特征和embedding特征拼接后作为Cross层和Deep层的共同输入。这对于Deep层是合理的，但我们知道**人工交叉特征**基本是对**原始sparse特征**进行叉乘，那为何不直接用原始sparse特征作为Cross的输入呢？联系这里介绍的Cross设计，每层layer的节点数都与Cross的输入维度一致的，直接使用大规模高维的sparse特征作为输入，会导致极大地增加Cross的参数量。当然，可以畅想一下，其实直接拿原始sparse特征喂给Cross层，才是论文真正宣称的“省去人工叉乘”的更完美实现，但是现实条件不太允许。所以将高维sparse特征转化为低维的embedding，再喂给Cross，实则是一种trade-off的可行选择。