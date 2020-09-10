# LSTM



## RNN

循环神经网络（Recurrent Neural Network，RNN）是一种用于处理**序列数据**的神经网络

结构图如下：

![img](https://pic1.zhimg.com/v2-71652d6a1eee9def631c18ea5e3c7605_b.jpg)

![image-20200908211320000](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200908211320000.png)





## LSTM

lstm是一种特殊的RNN， 主要是为了解决长序列训练过程中的***梯度消失和梯度爆炸***问题。

简单来说，相较于普通rnn，LSTM可以在更长的序列中更好的表现 



结合xmind中的结构图 具体了解 

LSTM结构（图右）和普通RNN的主要输入输出区别如下所示。

![img](https://pic3.zhimg.com/v2-e4f9851cad426dfe4ab1c76209546827_b.jpg)

具体结构： 

有3个门控状态，分别是**forget，input， output**，计算方式类似，只有参数不同

![[公式]](https://www.zhihu.com/equation?tex=z) 则是将结果通过一个 ![[公式]](https://www.zhihu.com/equation?tex=tanh) 激活函数将转换成-1到1之间的值（这里使用 ![[公式]](https://www.zhihu.com/equation?tex=tanh) 是因为这里是将其做为输入数据，而不是门控信号）

<img src="https://pic3.zhimg.com/v2-15c5eb554f843ec492579c6d87e1497b_b.jpg" alt="img" style="zoom:50%;" />

<img src="https://pic2.zhimg.com/v2-d044fd0087e1df5d2a1089b441db9970_b.jpg" alt="img" style="zoom:50%;" />

![img](https://pic3.zhimg.com/v2-556c74f0e025a47fea05dc0f76ea775d_b.jpg)

具体解释lstm流程：

1. 忘记阶段 --- 对上一个节点传进来的输入进行**选择性**忘记。简单来说就是会 “忘记不重要的，记住重要的”。

   具体通过 forget gate来作为忘记门控 来控制上一个状态的 ![[公式]](https://www.zhihu.com/equation?tex=c%5E%7Bt-1%7D) 哪些需要留哪些需要忘。

2. 选择记忆阶段 --- 将这个阶段的输入有选择性地进行“记忆”。主要是会对输入 ![[公式]](https://www.zhihu.com/equation?tex=x%5Et) 进行选择记忆。哪些重要则着重记录下来，哪些不重要，则少记一些

3. 输出阶段 --- 这个阶段将决定哪些将会被当成当前状态的输出。主要是通过 ![[公式]](https://www.zhihu.com/equation?tex=z%5Eo) 来进行控制的。并且还对上一阶段得到的 ![[公式]](https://www.zhihu.com/equation?tex=c%5Eo) 进行了放缩（通过一个tanh激活函数进行变化）



### **LSTM优点**

1. 利用乘法门，来控制输入信息、状态转移，以及输出信息的有选择性地表达。
2. **选择性角度**模型选择记住（或遗忘）它想要记住（或遗忘）的部分，从而更有效地利用其隐层单元。



### LSTM缺点

1. 引入了很多参数，训练难度加大--》 使用效果相当而参数更少的GRU来替代LSTM



### 关于LSTM如何解决梯度消失和梯度爆炸

LSTM如何来避免梯度弥散和梯度爆炸？ - Towser的回答 - 知乎 https://www.zhihu.com/question/34878706/answer/665429718

里面公式推导比较多 - superbrother的回答 - 知乎 https://www.zhihu.com/question/34878706/answer/654501152



1. 首先，主要解决的是**梯度消失**的问题，梯度爆炸问题相对容易解决，只要有**梯度截断** 就可以

2. lstm解决的是----RNN 所谓梯度消失并**不是说梯度为0** **其真正含义**是，梯度被**近距离梯度**主导，导致模型**难以学到远距离的依赖关系。**

   rnn中梯度消失的本质： **由于时间维度共享了参数矩阵，导致计算隐态** ![[公式]](https://www.zhihu.com/equation?tex=h_t)时会循环计算矩阵乘法，所以BPTT算法求解梯度时出现了**参数矩阵的累乘**

3. **LSTM 中梯度的传播有很多条路径**，![[公式]](https://www.zhihu.com/equation?tex=c_%7Bt-1%7D+%5Crightarrow+c_t+%3D+f_t%5Codot+c_%7Bt-1%7D+%2B+i_t+%5Codot+%5Chat%7Bc_t%7D) 这条路径上只有逐元素相乘和相加的操作，梯度流最稳定；

   具体：

   > 上面是类比了一下，现在我们数学推导看一看
   >
   > ![[公式]](https://www.zhihu.com/equation?tex=c_t%3Df_t+%5Codot+c_%7Bt-1%7D%2Bi_t+%5Codot+g_t+++++++)
   >
   > ![[公式]](https://www.zhihu.com/equation?tex=%5Cquad+%3Df_t%5Codot+f_%7Bt-1%7D+%5Codot+c_%7Bt-2%7D%2Bf_t%5Codot+i_%7Bt-1%7D+%5Codot+g_%7Bt-1%7D%2Bi_t+%5Codot+g_t)
   >
   > ![[公式]](https://www.zhihu.com/equation?tex=%5Cquad+%3Df_t+%5Codot+f_%7Bt-1%7D+%5Ccdots+%5Codot+f_%7B1%7D+%5Codot+c_0+%2B%5Csum_%7B%5Ctau%3D0%7D%5E%7Bt%7D+f_t+%5Codot+f_%7Bt-1%7D+%5Ccdots+%5Codot+f_%7B%5Ctau%7D+%5Codot+i_%7B%5Ctau%7D+%5Codot+g_%7B%5Ctau%7D+%2Bi_t+%5Codot+g_t)
   >
   > 其中现在是f（遗忘门）的连乘

4. 在**其他路径**上，LSTM 的梯度流和普通 RNN 没有太大区别，依然会爆炸或者消失， 由于总的远距离梯度 = 各条路径的远距离梯度之和，即便其他远距离路径梯度消失了，只要保证有一条远距离路径（就是上面说的那条高速公路）梯度不消失，总的远距离梯度就不会消失（正常梯度 + 消失梯度 = 正常梯度）。因此 LSTM 通过改善**一条路径**上的梯度问题拯救了**总体的远距离梯度**。

5.  **LSTM 仍然有可能发生梯度爆炸**。不过，由于 LSTM 的其他路径非常崎岖，和普通 RNN 相比多经过了很多次激活函数（导数都小于 1），因此 **LSTM 发生梯度爆炸的频率要低得多**。实践中梯度爆炸一般通过梯度裁剪来解决。

