# Word2Vec

涉及3篇文章

1.  [[Word2Vec\] Efficient Estimation of Word Representations in Vector Space (Google 2013)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%5BWord2Vec%5D%20Efficient%20Estimation%20of%20Word%20Representations%20in%20Vector%20Space%20%28Google%202013%29.pdf)
2.  [[Word2Vec\] Distributed Representations of Words and Phrases and their Compositionality (Google 2013)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%5BWord2Vec%5D%20Distributed%20Representations%20of%20Words%20and%20Phrases%20and%20their%20Compositionality%20%28Google%202013%29.pdf)
3. [[Word2Vec\] Word2vec Parameter Learning Explained (UMich 2016)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%5BWord2Vec%5D%20Word2vec%20Parameter%20Learning%20Explained%20%28UMich%202016%29.pdf)



参考的博客：

从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史 - 张俊林的文章 - 知乎https://zhuanlan.zhihu.com/p/49271699

word2vec一篇细致的讲解 -https://www.cnblogs.com/peghoty/p/3857839.html

[NLP] 秒懂词向量Word2vec的本质 - 穆文的文章 - 知乎https://zhuanlan.zhihu.com/p/26306795

理解 Word2Vec 之 Skip-Gram 模型 - 天雨粟的文章 - 知乎 https://zhuanlan.zhihu.com/p/27234078

全面理解word2vec - Evan的文章 - 知乎 https://zhuanlan.zhihu.com/p/33799633 -- 里面有两种输入的解释 



一些不同方法之间的关联：

基础假设： 分布式假设- 其核心思想为出现于上下文情景中的词汇都有相类似的语义 (上下文相似，其语义也相似)

- 基于计数的方法 (e.g. 潜在语义分析， Glove)  -- 统计该词与附近词汇的词频 再映射低维稠密向量
- 预测方法 (e.g. 神经概率化语言模型，word2vec) -- 直接从某词汇的邻近词汇 对其进行预测 在此过程中学习 embedding向量 



一些注解： 

CBOW模型以及Skip-gram  context为单个词的结构，可以看到是有**两个网络参数**的， 

网络结构- 隐层没有使用任何激活函数，但是输出层使用了sotfmax

![image-20201101222139061](https://i.loli.net/2020/11/01/OBrmXN54uQJalyp.png)

1. word2vec的主要目的和之前的nnlm不同，之前是为了学习到语言模型，而词向量是副产物，此时word2vec的主要目的就是为了**学习模型参数--神经网络的权重(参数)**，这些权重即上图所示的**两个参数矩阵**（输入层到隐含层的权重  以及  隐含层到输出层的权重），得到这两个参数矩阵之后，由one-hot相乘即可以得到一个**词向量** 

   <img src="https://i.loli.net/2020/11/02/kcdw83ACzVhuq1K.png" alt="img" style="zoom:50%;" />

   实际为了有效地进行计算，这种稀疏状态下不会进行矩阵乘法计算，而是直接去查找矩阵的第x行， 模型中的隐层权重矩阵便成了一个”查找表“（lookup table），进行矩阵计算时，直接去查输入向量中取值为1的维度下对应的那些权重值。隐层的输出就是每个输入单词的“嵌入词向量”。

2. 最终每一个word都会有两个向量：V_word和V_context ？ （是不是上文所说的呢？ ）

3. **词向量的维度**（embedding size）往往极大小于词的数量 所以Word2vec可以理解为一种降维操作, 把词语从 one-hot encoder 形式的表示(高维稀疏)降维到 Word2vec 形式（低维稠密）的表示。

4. 有关训练过程种的一些理解：

   - 输入即是N-维的one-hot 向量， 输出是 --神经网络基于这些训练数据将会输出一个概率分布，这个概率代表着我们的词典中的每个词是output word的可能性。 or 模型的输出概率代表着到我们词典中每个词有多大可能性跟input word同时出现。 

5. 上述第一条是word2vec的精髓--学习网络参数，然后参数矩阵的值既可以做为词向量来代表词， 而hierarchical softmax 和 negative sampling只是他的训练技巧

   - hierarchical softmax ----- 本质是把 N 分类问题变成 log(N)次二分类  （于huffman树相关） 
     - Hierarchical Softmax是一种对输出层进行优化的策略，输出层从原始模型的利用softmax计算概率值改为了利用Huffman树计算概率值。   
   - negative sampling（NEG） ------本质是一个预测全部分类的变成**预测总体类别的子集**的方法
     - 优化目标变为了：最大化正样本的概率，同时最小化负样本的概率。

6. 在原来第二篇论文中的三个创新 

   - 将常见的单词组合（word pairs）或者词组作为单个“words”来处理。
   - 对高频次单词进行抽样来减少训练样本的个数。
   - 对优化目标采用“negative sampling”方法，这样每个训练样本的训练只会更新一小部分的模型权重，从而降低计算负担--提高训练速度并且改善所得到词向量的质量



word2vec流行的原因（优点）：

1. 训练速度-- 不是训练语言模型 而是单纯的去学词向量，从而不需要隐藏层 结合Hsoftmax和负采样加速 训练速度快
2. 在语义上面的优势 





word2vec的常见面试题：

1. 主要是负采样 其中的负采样的常用方法 
2. 介绍一下word2vec 

