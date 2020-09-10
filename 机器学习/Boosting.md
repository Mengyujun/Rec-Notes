



# GBDT和Xgboost学习 

非常好的链接 -重点是其中的pdf

机器学习算法中 GBDT 和 XGBOOST 的区别有哪些？ - wepon的回答 - 知乎 https://www.zhihu.com/question/41354392/answer/98658997



## 系统学习

主要是总结上述链接中pdf的内容：

![image-20200909211450764](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909211450764.png)



### 泰勒公式

**定义**： 泰勒公式是一个用函数在某点的信息描述其附近取值的公式。**局部有效性** 就是用**多项式函数去逼近光滑函数**

![image-20200909212938584](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909212938584.png)

迭代形式：

![image-20200909213634833](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909213634833.png)



### 最优化方法之梯度下降法

在机器学习任务中，需要最小化损失函数L(theta)，其中是要求解的模型参数。梯度下降法常用来求解这种无约束最优化问题，它是一种迭代方法：选取初值theta_0 ，不断迭代，更新 theta 的值，进行损失函数的极小化

![image-20200909214734540](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909214734540.png)



### 最优化方法之牛顿法

对L(theta) 进行二阶泰勒展开：

迭代公式同上

![image-20200909220455133](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909220455133.png)

![image-20200909221032015](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909221032015.png)

![image-20200909221105935](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909221105935.png)



实际上就是 二阶展开多了一个二阶导，目的还是让 L(theta_t) < L(theta_t-1) 尽可能小 



### 从参数空间到函数空间（加法模型）

**GBDT**是 在函数空间利用 **梯度下降** 来优化

**XGboost** 是在 函数空间利用 **牛顿法** 来优化 



GBDT和梯度下降的对比：

![image-20200909221904999](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909221904999.png)

![image-20200909222707227](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909222707227.png)

因此 boosting算法是一种加法模型 ：

![image-20200909222802286](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909222802286.png)



### GradientBoostingTree算法原理

由上文可知，GBDT是一种加法模型，

![image-20200909223256960](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909223256960.png)

模型学习策略：

![image-20200909223447332](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909223447332.png)

贪心算法： 即每次都局部最优化即可 

GBDT实际算法原理：

![image-20200909223928018](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909223928018.png)

注： 

1.  2.1中 F(x) 是上一次得到的 F_t-1(x) 

2.  2.2中利用 CART决策树 去学习 损失函数的负梯度（对上一次的F_t-1   道理类似于 梯度下降的 f'(t-1)）

3. 2.3 中利用**线性搜索**  其实也是 **贪心算法** 就是直接意义上 去**尝试** 参数为多少时， 使得当前的loss最小 

4. 如果问题是 **回归问题**+loss为**平方误差损失函数** ， 即每次拟合的就是实际的**残差** 

   残差的意义如公式：残差 = 真实值 - 预测值 
   
   （**背会**）*对于 **一般损失函数**， 每次将 **损失函数的负梯度在当前模型的的值**（即2.1中的式子） 作为 **残差的近似值***

![img](https://upload-images.jianshu.io/upload_images/967544-1502996028c98f08.png?imageMogr2/auto-orient/strip|imageView2/2/w/666/format/webp)



### NewtonBoostingTree算法原理：详解XGBoost

#### 模型函数形式

![image-20200909232342259](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909232342259.png)

注意结果是多颗树的结果的累加



#### 目标函数

![image-20200909232859316](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909232859316.png)

即经验损失（训练数据误差）和结构损失（正则化项）



##### 正则化

有关正则化的一种解释：

![image-20200909233244965](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909233244965.png)

![image-20200909233334399](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909233334399.png)

不同点： 相比原始的GBDT，XGBoost的目标函数多了正则项，使得学习出来的模型更加不容易过拟合。

衡量树的复杂度： **树的深度**，内部节点个数，**叶子节点个数(T)**，**叶节点分数(w)**

xgboost即使用了 叶子节点以及叶节点分数两个来控制正则化

![image-20200909233725349](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909233725349.png)

##### 误差函数的二阶泰勒展开

首先是常规的 加法模型的展开：

![image-20200909234437053](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909234437053.png)

然后利用泰勒二阶展开：

![image-20200909234606925](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909234606925.png)

去掉其中的常数项

![image-20200909235137441](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909235137441.png)

![image-20200909235323311](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909235323311.png)

其中 前面1-n 是对样本的累加  后面的 1-T 是对叶节点的累加 



然后按照**叶节点**对 样本进行整合，  按照**叶节点累加方式**

![image-20200909235757197](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909235757197.png)

得到目标函数之后，就是令其最小，即可以求导为0 ：

如果能够确定树的结构，即每个叶节点中包含哪些样本确定，整个树确定，此时的**变量变为w_j 即叶节点分数**

![image-20200909235919889](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200909235919889.png)

由上可知，当树的结构确定时，则w_j 和损失函数都可知，那么如何确定树结构？



#### 回归树的学习策略

**贪心法**，每次尝试分裂一个叶节点，计算分裂前后的增益，选择增益最大的



##### **分裂增益的选择**：

ID3算法采用信息增益 C4.5算法采用信息增益比 CART采用Gini系数

XGBoost呢？

![image-20200910002236856](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200910002236856.png)

##### 树节点分裂方法（SplitFinding）

精确法： 遍历所有特征的所有可能划分点 

近似法： 对于每个特征，只考察分位点



##### 稀疏值处理



##### 其他特性

![image-20200910002914630](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200910002914630.png)

##### 系统设计（待补充）



### LightGBM

大致优点：

速度更快 内存占用更少 准确率更高（优势不明显，与XGBoost相当）



#### 改进

##### 直方图算法

1. 即 利用直方图类似的来**分桶**， 并直接统计直方图（单桶）的值，方便后续**分割**
2. **直方图差加速**， 叶子的直方图可以由它的父亲节点的直方图与它兄弟节点的直方图做差



##### 建树过程

XGBoost： Level-wise 类似于广度优先  同一层所有节点都做分裂，最后剪枝

LightGBM： Leaf-wise 类似于深度优先   选取具有最大增益的节点分裂容易过拟合，通过max_depth限制



##### 并行优化

###### 传统方法- 特征并行（切分特征）

1. 垂直切分数据，每个worker只有部分特征
2. 各找各的最优切分点，之后**通信**再找全局的，再**广播**

![image-20200910003755979](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200910003755979.png)



LightGBM的优化 

![image-20200910004405310](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200910004405310.png)





###### 传统方法-- 数据并行（切分数据）

1. 每个worker部分数据 
2. 本地自己的 数据统计直方图 再**汇合全局** 再分裂

![image-20200910003820489](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200910003820489.png)

lightGBM优化数据并行：

![image-20200910004543307](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200910004543307.png)



## GBDT常见细节问题汇总

1. 分类还是回归

   主要是解决回归预测，在做一些调整之后亦可以做分类问题

2. 





## GBDT优势

1. 天然的区分**有效特征** 以及 进行 **特征组合**
2. ![image-20200910115534173](E:\github\面经\机器学习\image-20200910115534173.png)