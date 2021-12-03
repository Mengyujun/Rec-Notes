# item2vec

涉及论文： 

[[Item2Vec\] Item2Vec-Neural Item Embedding for Collaborative Filtering (Microsoft 2016)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%5BItem2Vec%5D%20Item2Vec-Neural%20Item%20Embedding%20for%20Collaborative%20Filtering%20%28Microsoft%202016%29.pdf)



### 主要思想： 

将SGNS（Skip-gram with Negative Sampling） 的思想应用到 item的表示， 从而获得item的embedding，用来计算item-CF

把item视为word，用户的**行为序列视为一个集合**，**item间的共现为正样本**，并按照item的频率分布进行负样本采样



### Skip-gram的模型架构：



![img](https://pic4.zhimg.com/v2-50987411c5b403620bdedfd60c8970fb_b.png)

Skip-gram是利用当前词预测其上下文词。给定一个训练序列![[公式]](https://www.zhihu.com/equation?tex=w_{1}),![[公式]](https://www.zhihu.com/equation?tex=w_{2}+),...,![[公式]](https://www.zhihu.com/equation?tex=w_{T}+)，模型的目标函数是最大化平均的log概率：

![img](https://pic4.zhimg.com/v2-f166c2b4891c7ee4f27a0f7b877489d3_b.png)

目标函数中c中context的大小。c越大，训练样本也就越大，准确率也越高，同时训练时间也会变长。

在skip-gram中，![[公式]](https://www.zhihu.com/equation?tex=P(w_{t%2Bj}|w_{t})) 利用softmax函数定义如下：

![img](https://pic1.zhimg.com/v2-28932813495efde94920a29ace6ea374_b.png)

W是整个语料库的大小。上式的梯度的计算量正比如W，W通常非常大，直接计算上式是不现实的。为了解决这个问题，google提出了两个方法，一个是hierarchical softmax，另一个方法是negative sample。

negative sample的思想本身源自于对Noise Contrastive Estimation的一个简化，具体的，把目标函数修正为：



![img](https://pic1.zhimg.com/v2-a78307e5ce97eb6cacbb9d97a52718a8_r.jpg)

![[公式]](https://www.zhihu.com/equation?tex=P_{n}(w))是噪声分布 ( noise distribution )。即训练目标是使用Logistic regression区分出目标词和噪音词。具体的Pn(w)方面有些trick，google使用的是unigram的3/4方，即![[公式]](https://www.zhihu.com/equation?tex=U(w)^{3%2F4}%2FZ+)，好于unigram，uniform distribution。