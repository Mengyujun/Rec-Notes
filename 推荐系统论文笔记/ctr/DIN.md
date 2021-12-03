有疑问： 

现在的所有的特征是大杂烩扔进去的

然后要找到所有的用户的特征-和历史特征的交叉 所以要找到然后进行soft-match 

各种不同的product 到底是如何工作的 是什么含义呢 

是否需要一个阶段 将所有的特征分个类 之后再做dnn汇合 



感觉自己对特征的感知度不够 自己想想是否有影响 举例想想 



为啥要取消分段特征 

- 分段特征是冗余的
- 分段特征其实 也是有位置信息的 不同分段 top3 top20 之内的肯定是不一样的 但是分段内部没有权重 



din中item要和带位置信息的分段计算权重么 



为啥要特征加位置信息 就拿top20的分类 来说

举例： 军事#1 和军事#20  发现军事#1 和item-军事匹配时 更容易发生点击 即军事-1 军事-20 实际上是不一样的  



有关特征是否有用 只需要举个例子 不同的情况下是否发生点击的概率不同 ；增加某个特征是否具有区分度，是否因为这个特诊更容易发生点击，**举例**想想



 

到底哪个有用呢？ 等待实验看看 自己需要更认真的看看 

**用户的top20 其实是长期画像**

**而最近点击 展示的是用户的短期画像**  



用户历史向量表达： 

是将 用户的点击id， 点击cat，点击media，点击topic等汇总在一起 进行concat 之后作为一个整个的 某个点击的embedding表达 ，还是分开几个list分别计算呢？ 



分别计算感觉会好一点，即分别构造不同的list

比如说 cat和tag ，topic以及media 的优先级是 ：



![img](https://pic3.zhimg.com/80/v2-ba0d0fafc33999e9af1877cf6087523e_720w.jpg)

这张图感觉更靠谱点 : 

- base - 不同类别自己内部做embedding的pooling 如商品id和shopid 分别sum pooling 
- din - 相对于之前的base的区别只是 说他用了一个新的pooling 方式 其他的还是相同的 
- 其中权重的计算方式为:**u和v以及u v的element wise差值向量合并起来作为输入，然后喂给全连接层，最后得出权重**

一些想法： 

- 后续可以用 点积 迅速试一下？ 

- 对了 之前按照忠儒的说法 相减是为了 之间的差异 相乘表示正负方向？ 怎么个说法呢？

  >  减法表示两者差异，加法相当于pooling，乘法表示相似性。 
  >
  > 一个外积一个差，差很容易理解成两向量之间的距离（相似度），外积可以按照向量的方向来理解，





是不是还要改造 

之前的根本没有具体的名称，没有名称的映射，只有hash（带filed信息）> index > embedding 

如何改造 



尝试构建新的result值 即 不仅仅有一个hash值》index 值 还应当有

当前的做法实际上是 一种 one-hot-full-embedding 







之前是否需要统计 不同field的大小关系  数量关系 即一共有多少个num

即 不同field的原始大小应该是多少的



然后考虑 不同field的embedding维度应该是多少

即目前先统计一下 vocab_size 然后设计一下 dim_size 

``` text
layers.Embedding(vocab_size, dim_size, dtype='float32')
```

> embedding的size我一般采用个经验值，假如embedding对应的原始feature的取值数量为 ![[公式]](https://www.zhihu.com/equation?tex=n) ，那么我一般会采用 ![[公式]](https://www.zhihu.com/equation?tex=log_2%7B%28n%29%7D) 或者 ![[公式]](https://www.zhihu.com/equation?tex=k%5Csqrt%5B4%5D%7Bn%7D)![[公式]](https://www.zhihu.com/equation?tex=%28k+%5Cle+16%29) 来做初始的size，然后2倍扩大或缩小，实验几次，一般就能得到一个相对较好的值。另外，embedding的初始化也是非常重要的，需要精细调参。
>
> 
>
> 作者：严林
> 链接：https://www.zhihu.com/question/283167457/answer/429884110
> 来源：知乎
> 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

对啊 那如果有新的id是怎么处理的呢？ 之前的embedding table 是如何处理的?  就是单纯的hash处理？ 



忠儒说的 trace 是啥意思 尽量不要在流程中有 map map会出现问题 ？ 

```
# 是因为保存模型方式的原因么
t.jit.trace() 后续可以观察一下 
```



先依据长期画像来做 直接使用之前的top特征







先进行长期画像实验： 

cat的特征已经很有区分度 不需要cat  因此进行 topic 和 tags的实验 认为media的效果不太好 后续可以尝试



暂时不需要mask， 因为自己是固定长度的计算， 如果涉及到可变长序列，就需要用到mask来屏蔽一些关键

具体可以看deepctr_torch 中 din 代码的实现



DIN中多个序列如何输入din 

-  方案一 : 先按照常规的方式 concat 然后一起输入一个dnn网络中去 

```
"att_pair": "n_news_tags_top3#n_user_tags_20|n_news_topics_top3#n_user_topics_20",
```

出现一个问题，之前train的部分并没有生效，重新开始下一小时的eval之后，又是在0.5左右波动 

感觉像是模型参数没有保存下来 所以下一小时又重新开始训练 （问题解决， 自己在din模型中没有加载参数 使用base_model的_update_state_dict 函数）



注意一下 base实验中 drop_out的设置 

dnn-base 是0.2？  注意现在和drop-out为0 的进行对比 

是exp1的实验是对比 



2个序列一共有几种方式呢？  

- 单独每个序列都有一个独立的attention-mlp网络
- for循环的方式 公用一个 
- 像其他论文一样  concat之后 再 公用一个网络 

