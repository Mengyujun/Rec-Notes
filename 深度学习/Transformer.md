## Transformer 

参考：

- 【经典精读】Transformer模型深度解读 - <em>潘小小</em>的文章 - 知乎 https://zhuanlan.zhihu.com/p/104393915



### 1. Transformer整体架构

整体架构图： 

![image-20210309214117289](https://i.loli.net/2021/03/09/76KsFaoPpvIwzE9.png)

### 2. Attention的背景溯源

机器翻译领域的模型演进历史：

**Simple RNN** -》 Contextualize RNN -》 Contextualized RNN with attention -》 **Transformer**



**Simple RNN：** 

encoder-decoder模型结构，  **encoder**将整个**源端序列**(不论长度)压缩成一个**向量(encoder output)**， encoder output会作为decoder的initial states的输入

![image-20210309214716660](https://i.loli.net/2021/03/09/QiLaOw83MNdpFK9.png)

主要存在缺陷： 

- 需要将输入的源端序列 统一压缩成**固定维度向量**， 末尾信息更多，若初始信息序列很长，可能导致起始位置信息丢失。
- RNN特性导致的，  随着decoder timestep的信息的增加，无法传递长序列 长依赖  decoder会逐渐“遗忘”**源端序列的信息**（最原始的信息 rnn的最开始），而更多地关注目标序列中在该timestep之前的token的信息



**Contextualized RNN**：

为解决上述问题2（encoder output随着decoder timestep增加而信息衰减的问题）, 提出了**Contextualized RNN**

即decoder在每个timestep的input上都会加上一个context（即手动添加了 源端信息，防止源端的context信息随着timestep的增长而衰减）

![image-20210309215415368](https://i.loli.net/2021/03/09/vjhbL9HkmY4sIC1.png)



 主要存在问题：

每次timestamp增加的context信息都是**静态**的（encoder端的final hidden states，或者是所有timestep的output的平均值）但实际在不同时间点，所需要的**context信息是不同的** 



**Contextualized RNN with soft align (Attention)**：

用**当前的输入token**的vector与encoder **output中的每一个position**的vector作一个"attention"操作，然后做加权来作为 context信息

![image-20210309220640535](https://i.loli.net/2021/03/09/LtG5E1zfg3uTwOo.png)



### 3. attention的细节

#### 点击attention

其公式为： 

![image-20210309220817953](https://i.loli.net/2021/03/09/EjCg3mKpRntcHaL.png)

其中 Q*K 因此要求这两个矩阵有相同的 维度 **d**  

公式中出现放缩的原因： 

![image-20210309221210136](https://i.loli.net/2021/03/09/CnHWa9Pyv6kutcG.png)

总之就是调节 A的分布，使之与维度解耦 





### 面试问到的一些问题：

介绍一下 Transformer 

self-attention如何作用 为什么要除以 根号下d

为什么要做 multihead-attention?

transformer 中序列长度如何影响  

