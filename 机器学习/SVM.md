# SVM



基础：

![](C:\Users\MYJ\AppData\Roaming\Typora\typora-user-images\image-20200830162515509.png)

## 硬间隔

### 求解目标

目标1：间隔最大化

两条平行直线的距离公式推广到高维可求得图2.1中margin的![[公式]](https://www.zhihu.com/equation?tex=%5Crho)

![[公式]](https://www.zhihu.com/equation?tex=margin+%3D+%5Crho+%3D+%5Cfrac+2+%7B%7C%7CW%7C%7C%7D+%5Ctag%7B2.2.1%7D)

我们的目标是使![[公式]](https://www.zhihu.com/equation?tex=%5Crho)最大, 等价于使![[公式]](https://www.zhihu.com/equation?tex=%5Crho%5E2)最大:![[公式]](https://www.zhihu.com/equation?tex=%5Cunderset%7BW%2Cb%7D%7Bmax%7D+%5Crho+%5Ciff+%5Cunderset%7BW%2Cb%7D%7Bmax%7D+%5Crho%5E2+%5Ciff+%5Cunderset%7BW%2Cb%7D%7Bmin%7D%5Cfrac+1+2+%7C%7CW%7C%7C%5E2+%5Ctag%7B2.2.2%7D)

上式的![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac+1+2)是为了后续求导后刚好能消去，没有其他特殊意义

​	

![[公式]](https://www.zhihu.com/equation?tex=%E7%9B%AE%E6%A0%87%E4%B8%80%EF%BC%88%E4%BD%BF%E9%97%B4%E9%9A%94%E6%9C%80%E5%A4%A7%E5%8C%96%EF%BC%89%EF%BC%9A%7B%5Cmin+_%7B%5Cmathbf%7Bw%7D%2C+b%7D+%5Cfrac%7B1%7D%7B2%7D%5C%7C%5Cmathbf%7Bw%7D%5C%7C%5E%7B2%7D%7D%5C%5C)2

目标2： 样本正确分类 即线性可分

![[公式]](https://www.zhihu.com/equation?tex=%E7%9B%AE%E6%A0%87%E4%BA%8C%EF%BC%88%E4%BD%BF%E6%A0%B7%E6%9C%AC%E6%AD%A3%E7%A1%AE%E5%88%86%E7%B1%BB%EF%BC%89%EF%BC%9Ay_%7Bi%7D%5Cleft%28%5Cmathbf%7Bw%7D%5E%7BT%7D%5Cmathbf%7Bx%7D_i%2Bb%5Cright%29+%5Cgeq+1%2C+i%3D1%2C2%2C+%5Cldots%2C+m%5C%5C)



最后整合在一起即： 

![[公式]](https://www.zhihu.com/equation?tex=%E7%BB%88%E6%9E%81%E7%9B%AE%E6%A0%87%EF%BC%9A%5Cbegin%7Barray%7D%7Bc%7D%7B%5Cmin+_%7Bw%2C+b%7D+%5Cfrac%7B1%7D%7B2%7D%5C%7Cw%5C%7C%5E%7B2%7D%7D+%5C%5C+%7B%5Ctext+%7Bs.t.+%7D+y_%7Bi%7D%5Cleft%28w%5E%7BT%7D+x_%7Bi%7D%2Bb%5Cright%29+%5Cgeq+1%2C+%5Cforall+i%7D%5Cend%7Barray%7D%5C%5C)

### 求解过程

2.2.4 为原始问题，有约束条件的最优化问题， 用拉格朗日函数解决

![[公式]](https://www.zhihu.com/equation?tex=min_%7Bw%2Cb%7Dmax_%5Calpha+L%28%5Comega%2C+b%2C+%5Calpha%29%3D%5Cfrac%7B1%7D%7B2%7D%5C%7C%5Comega%5C%7C%5E%7B2%7D%2B%5Csum_%7Bi%3D1%7D%5E%7Bm%7D+%5Calpha_%7Bi%7D%5Cleft%281-y_%7Bi%7D%5Cleft%28%5Comega%5E%7BT%7D+x_%7Bi%7D%2Bb%5Cright%29%5Cright%29%5C%5C+s.t.+%5Calpha_i+%5Cgeq+0++%2C+++%5Cforall+i)

即 max L(w,b,a) 就是原问题，解释：

> ​	符合正确分类条件时， maxL 的结果即为 1/2 w^2 
>
> ​    不符合时， max的结果 时 正无穷 
>
>  外围再 有min函数 则和 式2.2.4 相同 

在满足Slater定理的时候，且过程满足KKT条件的时候，原问题转换成对偶问题：

极小极大问题 转化为 极大极小问题 （极大极小问题给出了下限）

![[公式]](https://www.zhihu.com/equation?tex=max_%5Calpha+min_%7Bw%2Cb%7D+L%28%5Comega%2C+b%2C+%5Calpha%29%3D%5Cfrac%7B1%7D%7B2%7D%5C%7C%5Comega%5C%7C%5E%7B2%7D%2B%5Csum_%7Bi%3D1%7D%5E%7Bm%7D+%5Calpha_%7Bi%7D%5Cleft%281-y_%7Bi%7D%5Cleft%28%5Comega%5E%7BT%7D+x_%7Bi%7D%2Bb%5Cright%29%5Cright%29%5C%5C+s.t.+%5Calpha_i+%5Cgeq0+++%2C++%5Cforall+i)



此时先求内部的 min函数 ，即求导（最小处梯度为0） 

先求内部最小值，对 ![[公式]](https://www.zhihu.com/equation?tex=%5Comega) 和 ![[公式]](https://www.zhihu.com/equation?tex=b) 求偏导并令其等于 ![[公式]](https://www.zhihu.com/equation?tex=0) 可得：

![[公式]](https://www.zhihu.com/equation?tex=w%3D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%7B%5Calpha_%7Bi%7Dy_%7Bi%7Dx_%7Bi%7D%7D%2C%5C%5C+0%3D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%7B%5Calpha_%7Bi%7Dy_%7Bi%7D%7D.%5C%5C)

将其代入到上式中去可得到

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+max_%5Calpha++L%28%5Comega%2C+b%2C+%5Calpha%29%3D%26+%5Csum_%7Bi%3D1%7D%5E%7Bm%7D+%5Calpha_%7Bi%7D-%5Cfrac%7B1%7D%7B2%7D+%5Csum_%7Bi%3D1%7D%5E%7Bm%7D+%5Csum_%7Bj%3D1%7D%5E%7Bm%7D+%5Calpha_%7Bi%7D+%5Calpha_%7Bj%7D+y_%7Bi%7D+y_%7Bj%7D+x_%7Bi%7D%5E%7BT%7D+x_%7Bj%7D+%5C%5C+%26+s+.+t+.+%5Csum_%7Bi%3D1%7D%5E%7Bm%7D+%5Calpha_%7Bi%7D+y_%7Bi%7D%3D0++%EF%BC%88+%5Calpha_%7Bi%7D+%5Cgeq+0%2C+i%3D1%2C2%2C+%5Cldots%2C+m+%EF%BC%89%5Cend%7Baligned%7D%5C%5C)

且要求满足KKT条件：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bcases%7D+%E4%B9%98%E5%AD%90%E9%9D%9E%E8%B4%9F%3A+%5Calpha_i+%5Cge+0+%28i%3D1%2C2%2C...n.%E4%B8%8B%E5%90%8C%29+%5C%5C+%E7%BA%A6%E6%9D%9F%E6%9D%A1%E4%BB%B6%3A+y_i%28X_i%5ETW%2Bb%29+-+1%5Cge+0+%5C%5C+%E4%BA%92%E8%A1%A5%E6%9D%A1%E4%BB%B6%3A+%5Calpha_i+%28y_i%28X_i%5ETW%2Bb%29+-+1%29%3D0+%5Cend%7Bcases%7D+%5C%5C)

此时再求解a, 可以增加负号转化为最小值问题， 之后利用 SMO算法 

利用SMO算法求得a之后， 可得W，继而可求得b

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7BW%7D%3D+%5Csum_%7Bi%3D1%7D%5En+%5Chat%7B%5Calpha%7D_i+y_i+X_i+%5Ctag%7B2.4.11%7D)

所以至少存在一个![[公式]](https://www.zhihu.com/equation?tex=j), 使![[公式]](https://www.zhihu.com/equation?tex=y_j%28X_j%5ET+%5Chat%7BW%7D%2B%5Chat%7Bb%7D%29+-+1%3D0), 即可求得最优![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bb%7D):![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%5Chat%7Bb%7D+%26+%3D+%5Cfrac+1+%7By_j%7D+-X_j%5ET+%5Chat%7BW%7D+%5C%5C+%26+%3D+y_j+-X_j%5ET+%5Chat%7BW%7D+%5C%5C+%26+%3D+y_j-%5Csum_%7Bi%3D1%7D%5En+%5Chat%7B%5Calpha%7D_i+y_i+X_j%5ET+X_i+%5Cend%7Baligned%7D+%5Ctag%7B2.4.12%7D)

所以我们就求得了整个线性可分SVM的解。求得的分离超平面为:![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%3D1%7D%5En+%5Chat%7B%5Calpha%7D_i+y_i+X%5ETX_i+%2B+%5Chat%7Bb%7D%3D0+%5Ctag%7B2.4.13%7D)

则分类的决策函数为![[公式]](https://www.zhihu.com/equation?tex=f%28X%29+%3D+sign%28%5Csum_%7Bi%3D1%7D%5En+%5Chat%7B%5Calpha%7D_i+y_i+X%5ETX_i+%2B+%5Chat%7Bb%7D%29+%5Ctag%7B2.4.14%7D)



## 软间隔

主要思路： 在硬间隔的基础上， **我们放宽对样本的要求，允许少量样本分类错误**



### 求解目标

现在给之前的目标函数加上一个误差，就相当于允许原先的目标出错，引入松弛变量 ![[公式]](https://www.zhihu.com/equation?tex=%5Cxi_i+%5Cge+0) ，公式变为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmin+_%7Bw%2C+b%2C+%5Cxi%7D+%5Cfrac%7B1%7D%7B2%7D%5C%7Cw%5C%7C%5E%7B2%7D%2B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Cxi_%7Bi%7D%5C%5C)

松弛变量的计算：

试图用0，1损失去计算，但0，1损失函数并不连续，求最值时求导的时候不好求，所以引入合页损失（hinge loss）：

![[公式]](https://www.zhihu.com/equation?tex=l_%7Bh+i+n+g+e%7D%28z%29%3D%5Cmax+%280%2C1-z%29%5C%5C)

函数图张这样：

![img](https://pic1.zhimg.com/80/v2-bf35e95b541f12bdc22235e7666f827d_720w.jpg)

即max(0, 1-z) 即当z《1 不符合条件时，松弛变量为1-z 否则为0 

所以目标变为（正确分类的话损失为0，错误的话付出代价）但这个代价需要一个控制的因子，引入C>0，惩罚参数，即：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmin+_%7Bw%2C+b%7D+%5Cfrac%7B1%7D%7B2%7D%5C%7Cw%5C%7C%5E%7B2%7D%2BC%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+max%280%2C1+-+y_%7Bi%7D%5Cleft%28w%5E%7BT%7D+x_%7Bi%7D%2Bb%5Cright%29%29%5C%5C)

C 实际和 1/lamda 类似

C越大 对错误容忍越低 过拟合 C越小 对误差错误不敏感 则欠拟合

**所以软间隔的目标函数为：**

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Barray%7D%7Bc%7D%7B%5Cmin+_%7Bw%2C+b%2C+%5Cxi%7D+%5Cfrac%7B1%7D%7B2%7D%5C%7Cw%5C%7C%5E%7B2%7D%2BC+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Cxi_%7Bi%7D%7D+%5C%5C+%7B%5Ctext+%7B+s.t.+%7D+y_%7Bi%7D%5Cleft%28x_%7Bi%7D%5E%7BT%7D+w%2Bb%5Cright%29+%5Cgeq+1-%5Cxi_%7Bi%7D%7D+%5C%5C+%7B%5Cquad+%5Cxi_%7Bi%7D+%5Cgeq+0%2C+i%3D1%2C2%2C+%5Cldots+n%7D%5Cend%7Barray%7D%5C%5C)

其中：

![[公式]](https://www.zhihu.com/equation?tex=%5Cxi_%7Bi%7D%3Dmax%280%2C1+-+y_%7Bi%7D%5Cleft%28w%5E%7BT%7D+x_%7Bi%7D%2Bb%5Cright%29%29%5C%5C)

### 求解过程

与硬间隔类似：

上式的拉格朗日函数为：

![[公式]](https://www.zhihu.com/equation?tex=min_%7Bw%2Cb%2C%5Cxi%7Dmax_%7B%5Calpha%2C%5Cbeta%7D++L%28%5Comega%2C+b%2C+%5Calpha%2C%5Cxi%2C%5Cbeta%29%3D%5Cfrac%7B1%7D%7B2%7D%5C%7C%5Comega%5C%7C%5E%7B2%7D%2BC+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Cxi_%7Bi%7D%2B%5Csum_%7Bi%3D1%7D%5E%7Bm%7D+%5Calpha_%7Bi%7D%5Cleft%281-y_%7Bi%7D%5Cleft%28%5Comega%5E%7BT%7D+x_%7Bi%7D%2Bb%5Cright%29-%5Cxi_%7Bi%7D%5Cright%29-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Cbeta_%7Bi%7D+%5Cxi_%7Bi%7D%5C%5C+s.t.+%5Calpha_i+%5Cgeq+0++%E4%B8%94%5Cbeta_%7Bi%7D%5Cgeq+0%2C+++%5Cforall+i)

在满足Slater定理的时候，且过程满足KKT条件的时候，原问题转换成对偶问题：

![[公式]](https://www.zhihu.com/equation?tex=max_%7B%5Calpha%2C%5Cbeta%7D+min_%7Bw%2Cb%2C%5Cxi%7D+L%28%5Comega%2C+b%2C+%5Calpha%2C%5Cxi%2C%5Cbeta%29%3D%5Cfrac%7B1%7D%7B2%7D%5C%7C%5Comega%5C%7C%5E%7B2%7D%2BC+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Cxi_%7Bi%7D%2B%5Csum_%7Bi%3D1%7D%5E%7Bm%7D+%5Calpha_%7Bi%7D%5Cleft%281-y_%7Bi%7D%5Cleft%28%5Comega%5E%7BT%7D+x_%7Bi%7D%2Bb%5Cright%29-%5Cxi_%7Bi%7D%5Cright%29-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Cbeta_%7Bi%7D+%5Cxi_%7Bi%7D%5C%5C+s.t.+%5Calpha_i+%5Cgeq+0++%E4%B8%94%5Cbeta_%7Bi%7D%5Cgeq+0%2C+++%5Cforall+i)

先求内部最小值，对 ![[公式]](https://www.zhihu.com/equation?tex=%5Comega) , ![[公式]](https://www.zhihu.com/equation?tex=b) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Cxi) 求偏导并令其等于 ![[公式]](https://www.zhihu.com/equation?tex=0) 可得：

![[公式]](https://www.zhihu.com/equation?tex=w%3D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%7B%5Calpha_%7Bi%7Dy_%7Bi%7Dx_%7Bi%7D%7D%2C%5C%5C+0%3D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%7B%5Calpha_%7Bi%7Dy_%7Bi%7D%7D.%5C%5CC%3D%5Calpha_%7Bi%7D%2B%5Cbeta_%7Bi%7D%5C%5C)

将其代入到上式中去可得到，注意 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta) 被消掉了：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+max_%7B%5Calpha%2C%5Cbeta%7D++L%28%5Comega%2C+b%2C+%5Calpha%2C%5Cxi%2C%5Cbeta%29%3D%26+%5Csum_%7Bi%3D1%7D%5E%7Bm%7D+%5Calpha_%7Bi%7D-%5Cfrac%7B1%7D%7B2%7D+%5Csum_%7Bi%3D1%7D%5E%7Bm%7D+%5Csum_%7Bj%3D1%7D%5E%7Bm%7D+%5Calpha_%7Bi%7D+%5Calpha_%7Bj%7D+y_%7Bi%7D+y_%7Bj%7D+x_%7Bi%7D%5E%7BT%7D+x_%7Bj%7D+%5C%5C+%26+s+.+t+.+%5Csum_%7Bi%3D1%7D%5E%7Bm%7D+%5Calpha_%7Bi%7D+y_%7Bi%7D%3D0++%EF%BC%880+%5Cleq++%5Calpha_%7Bi%7D+%5Cleq+C%2C++i%3D1%2C2%2C+%5Cldots%2C+m+%EF%BC%89%5Cend%7Baligned%7D%5C%5C)

此时需要求解 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) ，同样**利用SMO（序列最小优化）算法。**



## 非线性SVM-核技巧

### 核函数

如下图所示，核技巧的基本思路分为两步:*使用一个变换将原空间的数据映射到新空间(例如更高维甚至无穷维的空间)；*然后在新空间里用线性方法从训练数据中学习得到模型。

![img](https://picb.zhimg.com/v2-87d10648809c0b6c5e473bd4565c2c08_b.jpg)

即 非线性变换将 原始输入空间对应到 高维的特征空间 ，在更高维的空间内找到一个分离超平面来解决分类问题 



有关核函数（偷吃步）：

核函数的定义：K(x,y)=<ϕ(x),ϕ(y)>， 

即应该有两步操作，1- 先进行x的映射转换，2- 再进行点积计算

此时引入核函数，这两步的计算便成了一步计算

如下图所示：

![img](https://pic4.zhimg.com/80/v2-3795f9ba5f773bfd0f13b286da40cbdc_720w.jpg)

### 核函数的选择

pass



## 损失函数

软间隔的基本型形式：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmin+_%7Bw%2C+b%7D+%5Cfrac%7B1%7D%7B2%7D%5C%7Cw%5C%7C%5E%7B2%7D%2BC%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+max%280%2C1+-+y_%7Bi%7D%5Cleft%28w%5E%7BT%7D+x_%7Bi%7D%2Bb%5Cright%29%29%5C%5C)

稍微做一点变化：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmin+_%7B%5Cboldsymbol%7Bw%7D%2C+b%7D+%5Cfrac%7B1%7D%7Bm%7D+%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28+%5Cmax+0%2C1-y_%7Bi%7D%5Cleft%28%5Cboldsymbol%7Bw%7D%5E%7B%5Ctop%7D+%5Cboldsymbol%7B%5Cphi%7D%5Cleft%28%5Cboldsymbol%7Bx%7D_%7Bi%7D%5Cright%29%2Bb%5Cright%29%2B%5Cfrac%7B%5Clambda%7D%7B2%7D%5C%7C%5Cboldsymbol%7Bw%7D%5C%7C%5E%7B2%7D%5C%5C)

转变为标准的**损失函数+正则化**的样子，其中, 第一项称为经验风险, 度量了模型对训练数据的拟合程度; 第二项称为结构风险, 也称为正则化项, 度量 了模型自身的复杂度. 



## **SVM对缺失数据敏感**

这里说的缺失数据是指缺失某些特征数据，向量数据不完整。SVM 没有处理缺失值的策略。而 SVM 希望样本在特征空间中线性可分，所以特征空间的好坏对SVM的性能很重要。缺失特征数据将影响训练结果的好坏。



##  **SVM的优缺点：**

**优点：**

1. 由于SVM是一个凸优化问题，所以求得的解一定是全局最优而不是局部最优。
2. 不仅适用于线性线性问题还适用于非线性问题(用核技巧)。
3. 拥有高维样本空间的数据也能用SVM，这是因为数据集的复杂度只取决于支持向量而不是数据集的维度，这在某种意义上避免了“维数灾难”。
4. 理论基础比较完善(例如神经网络就更像一个黑盒子)。

**缺点：**

1. 二次规划问题求解将涉及m阶矩阵的计算(m为样本的个数), 因此SVM不适用于超大数据集。(SMO算法可以缓解这个问题)
2. 只适用于二分类问题。(SVM的推广SVR也适用于回归问题；可以通过多个SVM的组合来解决多分类问题)



## LR和SVM的联系区别

联系：

1. 分类模型，一般处理二分类问题
2. 不考虑核函数 都是线性模型 
3. 都属于判别模型 



区别：

1. ​	LR是参数模型，SVM是非参数模型。
2. 从目标函数来看，区别在于逻辑回归采用的是交叉熵，SVM采用的是hinge loss
3. SVM不直接依赖数据分布，而LR则依赖，因为SVM只与支持向量那几个点有关系，而LR和所有点都有关系。
4. SVM本身是结构风险最小化模型，而LR是经验风险最小化模型

