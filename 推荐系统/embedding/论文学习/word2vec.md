# Word2Vec

涉及3篇文章

1.  [[Word2Vec\] Efficient Estimation of Word Representations in Vector Space (Google 2013)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%5BWord2Vec%5D%20Efficient%20Estimation%20of%20Word%20Representations%20in%20Vector%20Space%20%28Google%202013%29.pdf)
2.  [[Word2Vec\] Distributed Representations of Words and Phrases and their Compositionality (Google 2013)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%5BWord2Vec%5D%20Distributed%20Representations%20of%20Words%20and%20Phrases%20and%20their%20Compositionality%20%28Google%202013%29.pdf)
3. [[Word2Vec\] Word2vec Parameter Learning Explained (UMich 2016)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%5BWord2Vec%5D%20Word2vec%20Parameter%20Learning%20Explained%20%28UMich%202016%29.pdf)



参考的博客：

从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史 - 张俊林的文章 - 知乎https://zhuanlan.zhihu.com/p/49271699

word2vec一篇细致的讲解 -https://www.cnblogs.com/peghoty/p/3857839.html

[NLP] 秒懂词向量Word2vec的本质 - 穆文的文章 - 知乎https://zhuanlan.zhihu.com/p/26306795



一些注解： 

CBOW模型以及Skip-gram  context为单个词的结构，可以看到是有**两个网络参数**的， 

![image-20201101222139061](https://i.loli.net/2020/11/01/OBrmXN54uQJalyp.png)

1. word2vec的主要目的和之前的nnlm不同，之前是为了学习到语言模型，而词向量是副产物，此时word2vec的主要目的就是为了**学习模型参数--神经网络的权重(参数)**，这些权重即上图所示的**两个参数矩阵**（输入层到隐含层的权重  以及  隐含层到输出层的权重），得到这两个参数矩阵之后，由one-hot相乘即可以得到一个**词向量**

2. 最终每一个word都会有两个向量：V_word和V_context ？ （是不是上文所说的呢？ ）

3. **词向量的维度**（embedding size）往往极大小于词的数量 所以Word2vec可以理解为一种降维操作, 把词语从 one-hot encoder 形式的表示(高维稀疏)降维到 Word2vec 形式（低维稠密）的表示。

4. 上述第一条是word2vec的精髓--学习网络参数，然后参数矩阵的值既可以做为词向量来代表词， 而hierarchical softmax 和 negative sampling只是他的训练技巧

   - hierarchical softmax ----- 本质是把 N 分类问题变成 log(N)次二分类  （于huffman树相关）
   - negative sampling ------本质是预测总体类别的一个子集 

   





word2vec流行的原因（优点）：

1. 训练速度-- 不是训练语言模型 而是单纯的去学词向量，从而不需要隐藏层 结合Hsoftmax和负采样加速 训练速度快
2. 在语义上面的优势 

