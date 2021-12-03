# 主流Embedding技术解析



主要是对 [腾讯技术文章-embedding](https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247513629&idx=2&sn=afba7c17c3f59a9ad04598584aaf181a&chksm=fbd70471cca08d67fd0a4feb6febf613746e1bce6b03e72f6a47c82886f9008e4d1a0dec2683&mpshare=1&scene=24&srcid=1201XU4V6KMXKbzbu2ZKZJhT&sharer_sharetime=1606807200926&sharer_shareid=a1eba9ee6af76371282ef03db822bd93&key=1e7b385c415ba6befcd33c025869008fc3489cfdc7fd0a5842fab835816782e82ca82fb021cbd9bf6d06b880f57009874007b63988a89ac2c0be0cacd0dac4b0097853fdabebb714d64f87d4647def981761fb8b2a006afe7cdbda151113cabbfc8253677d0bf83014bb8d5835efa02b075d0a5d8d966a93685141377eb93d7c&ascene=14&uin=NzQxNjE1MjE4&devicetype=Windows+10+x64&version=6300002f&lang=zh_CN&exportkey=A9ndCTWhedtA1X9jhIZlOdE%3D&pass_ticket=2MKqBy%2BUnJb4dph7JIKg77OELcpVeCUO5z6JOxkXpzaI1p4C%2BklF9rdTj5kbvap%2B&wx_header=0)  的总结



## 引言

### 在推荐系统深度学习中embedding的主要使用方式

- 排序阶段：
  - 作为 **Embedding 层**嵌入到深度模型中，实现将高维稀疏特征到低维稠密特征的转换（如 Wide&Deep、DeepFM 等模型）；
  - 作为**预训练的 Embedding 特征向量**，与其他特征向量拼接后，一同作为深度学习模型输入进行训练（如 FNN）；
- 召回阶段，通过计算用户和物品的 Embedding 向量相似度，作为召回策略（比 Youtube 推荐模型等）；
- Serve 阶段， 实时计算用户和物品的 Embedding 向量，并将其作为实时特征输入到深度学习模型中（比 Airbnb 的 embedding 应用）。

 	

### 当前主流的embedding技术

- 经典的矩阵分解方法：这里主要是介绍 SVD 方法 其他的就是mf矩阵分解 以及变形后的 fm 

- 基于内容的 embedding 方法：这部分主要涉及到 NLP 相关的文本 embedidng 方法，包括静态向量 embedding（如 word2vec、GloVe 和 FastText）和动态向量 embedding（如 ELMo、GPT 和 BERT）

  ![img](https://pic1.zhimg.com/v2-18144af9c14e4121f810368593d51698_r.jpg)

- 基于 Graph 的 embedding 方法：这部分主要是介绍图数据的 embedding 方法，包括浅层图模型（如 DeepWalk、Node2vec 和 Metapath2vec）和深度图模型（如基于谱的 GCN 和基于空间的 GraphSAGE）



## 经典矩阵分解法



### SVD奇异值分解

svd分解的缺陷：

1. SVD分解要求矩阵是稠密 而实际的推荐评分矩阵是极度稀疏的 - 无论使用何种方法补全 效果不理想
2. 计算复杂度（时间复杂度是 n^3，空间复杂度是）



解决传统SVD的稀疏数据不适用和高计算复杂度的问题，其中主要的有**隐语义模型**（Latent Factor Model） --  FunkSVD、BiasSVD和SVD++算法。





## 基于内容的Embedding方法



基于内容的embedding方法，主要是针对**文本类型数据**，从最开始的静态向量方法（如word2vec、GloVe和FastText）发展为能根据上下文语义实现动态向量化的方法如（ELMo、GPT和BERT）。



### 静态向量

#### Word2vec

详情可以参见 论文学习之 word2vec.md 



#### Glove

GloVe（Global Vectors for Word Representation）是2014年由斯坦福大学提出的无监督词向量表示学习方法，是一个基于**全局词频统计**（count-based & overall statistics）的词表征工具。由它得到的词向量捕捉到单词之间一些语义特性，比如相似性、类比性等。GloVe主要分为三步：

- 基于语料构建词的共现矩阵

  表示词和词在特定大小的窗口内共同出现的次数。如对于语料：`I play cricket, I love cricket and I love football`，窗口为2的的共现矩阵可以表示为：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/j3gficicyOvaveaR5KgZQYDocLwgm53DyEfmNqgcZ3my0VQhLNKwOvCsNHClIInua5oR2YWLkl0aH6oZUqiaG5MWg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

构造词向量和贡献矩阵之间的关系：

![img](https://mmbiz.qpic.cn/mmbiz_png/j3gficicyOvaveaR5KgZQYDocLwgm53DyEJicGNqSqlNh3PxR4KnHUXhjXpZ7QJCHqNcb47L9Sib68WgUXlN8WFchQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

其中，和是要求解的词向量，和是两个词向量的偏差项。

- 最终 GloVe 的 loss function 如下：

![image-20201207160157058](https://i.loli.net/2020/12/07/mJXOtKH3rowFbhW.png)

其中，表示语料库中词个数。在语料库中，多个单词会一起出现多次，f(x)表示权重函数主要有以下原则：

- - 非递减函数，用于确保多次一起出现的单词的权重要大于很少在一起出现的单词的权重
  - 权重不能过大，达一定程度之后应该不再增加
  - f(0)=0，确保没有一起出现过的单词不参与loss的计算

在论文中，作者给出了如下分段函数：

![image-20201207160253385](https://i.loli.net/2020/12/07/hrLgeXO1yUPf3kl.png)

通过实验，作者得到了效果相对较好的a=3/4 ，x_max =100 ，此时对应 曲线如下图：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/j3gficicyOvaveaR5KgZQYDocLwgm53DyEQvAuf3iax8sDibxoR30tsyP8icMPsuZDLAGNxeaff6MP5vHtDpzw6vDfQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

CBOW和Skip-gram是local context window的方法，缺乏了整体的词和词的关系，负样本采样会缺失词的关系信息。此外，直接训练Skip-gram类型的算法，很容造成高曝光词汇得到过多的权重。

Global Vector融合了矩阵分解Latent Semantic Analysis (LSA)的全局统计信息和local context window优势。融入全局的先验统计信息，可以加快模型的训练速度，又可以控制词的相对权重。



####  **FastText**

FastText是FaceBook在2017年提出的**文本分类模型**（有监督学习）。词向量则是FastText的一个副产物。FastText模型结果如下图所示：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/j3gficicyOvaveaR5KgZQYDocLwgm53DyEknjOuj72anusicvY0e4D5NykLnibfZeA2jQQdqgaA0mglywBbDaVhgTQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

其中x1, x2, x3, xn 表示一个文本中的n-gram向量，每个特征是词向量的平均值。从模型结构可以看出，FastText与CBOW模型的结构相似，不同在于FastText预测的是全部的n-gram去预测指定类别，而CBOW预测的是中间词。



### 动态向量

由于静态向量表示中每个词被表示成一个固定的向量，无法有效解决**一词多义**的问题。在动态向量表示中，模型不再是向量对应关系，而是一个**训练好的模型**。在使用时，将文本输入模型中，模型根据上下文来推断每个词对应的意思，从而得到该文本的词向量。在对词进行向量表示时，能结合当前语境对多义词进行理解，实现不同上下文，其向量会有所改变。下面介绍三种主流的动态向量表示模型：ELMo、GPT和BERT。



后续可以结合 张俊林的文章重新整理以下 



#### ELMO

模型结构图： 

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/j3gficicyOvaveaR5KgZQYDocLwgm53DyEWLBicLpnGuw5iaTlMvUxRkRM2KbVjbBFyIad4Wn2h2EVmMjKZ6AiaFR5Q/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="img" style="zoom: 50%;" />

**多层双向 + LSTM** 尚未使用 transformer



#### GPT

**单向+ transformer** 

GPT-1 采用pre-training和fine-tuning的下游统一框架，将预训练和finetune的结构进行了统一，解决了之前两者分离的使用的不确定性（例如ELMo）。此外，GPT使用了Transformer结构克服了LSTM不能捕获远距离信息的缺点。GPT主要分为两个阶段：pre-training和fine-tuning

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/j3gficicyOvaveaR5KgZQYDocLwgm53DyE5ZjU7K62bDbGaDIgSOypb4MSgzVWwRBicQ79Tl1NGZkicHE80bdZUxfQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="img" style="zoom: 50%;" />



#### BERT

BERT、ELMo和GPT模型结构对比图如下图所示：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/j3gficicyOvaveaR5KgZQYDocLwgm53DyEbdAXibTSVcPnBFyTDPCMfWVZNLhQqviaSA8Vx1YoqIIJQC4iaSicWL8kCw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



相较于**ELMo**，BERT采用句子级负采样来得到句子表示/句对关系，使用**Transformer模型代替LSTM**，提升模型表达能力，Masked LM解决“自己看到自己”的问题。

相较于**GPT**，BERT采用了**双向的Transformer**，使得模型能够挖掘左右两侧的语境。此外，BERT进一步增强了词向的型泛化能力，充分描述**字符级、词级、句子级甚至句间的关系特征**。



BERT的输入的编码向量（长度为512）是3种Embedding特征element-wise和，如下图所示：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/j3gficicyOvaveaR5KgZQYDocLwgm53DyEGnN5RB6w0hbKIkO6y8ygzic4aMEyTqiaIDo3N4wV8Iev1bUIsDibRsYEQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

这三种Embedding特征分别是：

- Token Embedding (WordPiece)：将单词划分成一组有限的公共词单元，能在单词的有效性和字符的灵活性之间取得一个折中的平衡。如图中的“playing”被拆分成了“play”和“ing”；
- **Segment Embedding：用于区分两个句子**，如B是否是A的下文（对话场景，问答场景等）。对于句子对，第一个句子的特征值是0，第二个句子的特征值是1；
- Position Embedding：将单词的位置信息编码成特征向量，Position embedding能有效将单词的位置关系引入到模型中，提升模型对句子理解能力；

在模型预训练阶段，BERT采用两个自监督任务采用来实现模型的多任务训练：1）**Masked Language Model**；2）**Next Sentence Prediction**



##### **Masked Language Model (MLM)**

MLM的核心思想早在1953年就被Wilson Taylor[Wilson]提出，是指在训练时随机从输入语料中mask掉一些单，然后通过该词上下文来预测它（非常像让模型来做完形填空），如下图所以：

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/j3gficicyOvaveaR5KgZQYDocLwgm53DyEjO6KW2cibnRnWhagnPTKCO5gFfekESGf66QkwtFHYD91Fn4NXQIl8RQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="img" style="zoom:50%;" />

在论文实验中，只有**15%的Token会被随机Mask掉**。在训练模型时，一个句子会被多次输入模型中用于参数调优，对于某个要被Mask的Token并不是每次都一定会被Mask掉：

- `80%`概率直接替换为`[MASK]`，如`my dog is hairy -> my dog is [mask]`
- **`10%`概率替换为其他任意Token**，如`my dog is hairy -> my dog is apple`
- `10%`概率保留为原始Token，如`my dog is hairy -> my dog is hairy`

这样做的好处主要有：

- 如果某个Token100%被mask掉，在fine-tuning的时候会这些被mask掉的Token就成为OOV，反而影响模型的泛化性;
- 加入随机Token是因为Transformer要保持对每个输入Token的分布式表征，否则模型就会记住这个[MASK]=“hairy”
- 虽然加入随机单词带来的负面影响，但由于单词被随机替换掉的概率只有15%*10% =1.5%，负面影响可以忽略不计



##### **Next Sentence Prediction (NSP)**

许多重要的下游任务譬如QA、NLI需要语言模型理解两个句子之间的关系，而传统的语言模型在训练的过程没有考虑句对关系的学习。BERT采用NSP任务来增强**模型对句子关系的理解**，**即给出两个句子A、B，模型预测B是否是A的下一句**，如下图所示：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/j3gficicyOvaveaR5KgZQYDocLwgm53DyEH5JXdPK99gQkJlW8BwyQwBnCEU4xP6gymKB8cWKrABbb6cYN1eBjJg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

训练数据集构造上，从平行语料中随机抽取连续的两句话：50%保留抽取的两句话（label=IsNext）；50%的第二句话随机从语料中抽取（label=NotNext）



##### **Fine-tuning**

BERT提供了4中不同的下游任务的微调方案，大家根据自己的语料在预训练好的模型上采用这些任务来微调模型：

- 句对关系判断：第一个起始符号[CLS]经过编码后，增加Softmax层，即可用于分类；
- 单句分类任务：实现同“句对关系判断”；
- 问答类任务：问答系统输入文本序列的question和包含answer的段落，并在序列中标记answer，让BERT模型学习标记answer开始和结束的向量来训练模型；
- 序列标准任务：识别系统输入标记好实体类别（人、组织、位置、其他无名实体）文本序列进行微调训练，识别实体类别时，将序列的每个Token向量送到预测NER标签的分类层进行识别。



## 基于graph的embedding方法

基于内容的Embedding方法（如word2vec、BERT等）都是针对“**序列**”样本（如句子、用户行为序列）设计的，但在互联网场景下，数据对象之间更多呈现出**图结构**，如1）有用户行为数据生成的物品关系图；2）有属性和实体组成的只是图谱。

<img src="https://mmbiz.qpic.cn/mmbiz_jpg/j3gficicyOvaveaR5KgZQYDocLwgm53DyEZzH2xHhsQAxMLX1Wcrs4TjQG6GWJaQ7OcMLsPg09QoJP8s6G0tdU2A/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="img" style="zoom:50%;" />

Graph Embedding是一种将**图结构数据**映射为**低微稠密向量**的过程，从而捕捉到**图的拓扑结构、顶点与顶点的关系、以及其他的信息**。目前，Graph Embedding方法大致可以分为两大类：1）浅层图模型；2）深度图模型。



### 浅层图模型

浅层图模型主要是采用`random-walk + skip-gram`模式的embedding方法。主要是通过在图中采用**随机游走策略来生成多条节点列表**，然后将每个列表相当于含有多个单词（图中的节点）的句子，再用skip-gram模型来训练每个节点的向量。这些方法主要包括**DeepWalk、Node2vec、Metapath2vec**等。  ---- 随机游走 形成多条序列 -》 转化为文本类型的 embedding 



#### Deepwalk 

输入是一张图，输出是图中节点的向量表示， **目标**是使得图中两个点共有的邻居节点（或者高阶邻近点）越多，则对应的两个向量之间的距离就越近。

![img](https://mmbiz.qpic.cn/mmbiz_jpg/j3gficicyOvaveaR5KgZQYDocLwgm53DyEksaOZe9mQ1TUVEyP8SQJiaibHZNnelRGXibcTgyCqOTD0QGfVT9Ompib5A/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

DeepWalk得本质可以认为是：`random walk + skip-gram`。在DeepWalk算法中，需要形式化定义的是random walk的跳转概率，即到达节点后，下一步遍历其邻居节点的概率：

![image-20201207170009031](https://i.loli.net/2020/12/07/8G5uBbDSYC2Eo4w.png)

其中，N+(vi)表示及节点的所有出边连接的节点集合，M_ij表示由v_i节点连接至v_j节点的边的权重。由此可见，原始DeepWalk算法的跳转概率是跳转边的权重占所有相关出边权重之和的比例。

DeepWalk采用的**游走策略过于简单**（BFS），**无法有效表征图的节点的结构信息**。



#### Node2vec 

该模型通过**调整random walk权重**的方法使得节点的embedding向量更倾向于**体现网络的同质性或结构性**。

- 同质性 ： 指的是距离相近的节点的embedding向量应近似 用**DFS**

- 结构性： 结构相似的节点的embedding向量应近似 应该用**BFS**，BFS更能描绘出周围的结构 

  <img src="https://mmbiz.qpic.cn/mmbiz_jpg/j3gficicyOvaveaR5KgZQYDocLwgm53DyEtWMVOJ1j7ObJW0lXHIgpKDs0ibYED6zxehEjTED1JS5L7Emal6T9UDg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="img" style="zoom:50%;" />

在Node2vec中，同样是通过控制节点间的**跳转概率**来控制BFS和DFS倾向性的。

<img src="https://i.loli.net/2020/12/07/oOwBbnZgWIi6cEt.png" alt="image-20201207171911358" style="zoom:50%;" />

相较于DeepWalk，

Node2vec通过设计biased-random walk策略，能对图中节点的**结构相似性和同质性**进行权衡，使**模型更加灵活**。

但与DeepWalk一样，Node2vec**无法指定游走路径**，**且仅适用于解决只包含一种类型节点的同构网络**，无法有效表示包含多种类型节点和边类型的复杂网络。



#### **Metapath2vec**

为了解决Node2vec和DeepWalk**无法指定游走路径、处理异构网络**的问题

Metapath2vec方法，用于对**异构信息网络**（Heterogeneous Information Network, HIN）的节点进行embedding。



Metapath2vec**总体思想**跟Node2vec和DeepWalk相似，主要是在随机游走上使用**基于meta-path的random walk**来构建节点序列，然后用Skip-gram模型来完成顶点的Embedding。