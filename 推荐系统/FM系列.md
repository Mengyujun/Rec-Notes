# **FM**（因子分解机）

在线性回归模型的基础上

![img](https://pic2.zhimg.com/v2-0a7e0845c0f6cd3b8eaaf868d975487c_b.jpg)



FM的模型方程：

![img](https://pic4.zhimg.com/v2-cd7402c41afd7debf7e165da55007277_b.jpg)

FM公式计算简化： 复杂度即为 kn

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%26+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Csum_%7Bj%3Di%2B1%7D%5E%7Bn%7D%5Cleft%5Clangle%5Cmathbf%7Bv%7D_%7Bi%7D%2C+%5Cmathbf%7Bv%7D_%7Bj%7D%5Cright%5Crangle+x_%7Bi%7D+x_%7Bj%7D+%5C%5C%3D%26+%5Cfrac%7B1%7D%7B2%7D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%5Cleft%5Clangle%5Cmathbf%7Bv%7D_%7Bi%7D%2C+%5Cmathbf%7Bv%7D_%7Bj%7D%5Cright%5Crangle+x_%7Bi%7D+x_%7Bj%7D-%5Cfrac%7B1%7D%7B2%7D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Cleft%5Clangle%5Cmathbf%7Bv%7D_%7Bi%7D%2C+%5Cmathbf%7Bv%7D_%7Bi%7D%5Cright%5Crangle+x_%7Bi%7D+x_%7Bi%7D+%5C%5C%3D%26+%5Cfrac%7B1%7D%7B2%7D%5Cleft%28%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Csum_%7Bj%3D1%7D%5E%7Bn%7D+%5Csum_%7Bf%3D1%7D%5E%7Bk%7D+v_%7Bi%2C+f%7D+v_%7Bj%2C+f%7D+x_%7Bi%7D+x_%7Bj%7D-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Csum_%7Bf%3D1%7D%5E%7Bn%7D+v_%7Bi%2C+f%7D+v_%7Bi%2C+f%7D+x_%7Bi%7D+x_%7Bi%7D%5Cright%29+%5C%5C%3D%26+%5Cfrac%7B1%7D%7B2%7D+%5Csum_%7Bf%3D1%7D%5E%7Bk%7D%5Cleft%28%5Cleft%28%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+v_%7Bi%2C+f%7D+x_%7Bi%7D%5Cright%29%5Cleft%28%5Csum_%7Bj%3D1%7D%5E%7Bn%7D+v_%7Bj%2C+f%7D+x_%7Bj%7D%5Cright%29-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+v_%7Bi%2C+f%7D%5E%7B2%7D+x_%7Bi%7D%5E%7B2%7D%5Cright%29+%5C%5C%3D%26+%5Cfrac%7B1%7D%7B2%7D+%5Csum_%7Bf%3D1%7D%5E%7Bk%7D%5Cleft%28%5Cleft%28%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+v_%7Bi%2C+f%7D+x_%7Bi%7D%5Cright%29%5E%7B2%7D-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+v_%7Bi%2C+f%7D%5E%7B2%7D+x_%7Bi%7D%5E%7B2%7D%5Cright%29+%5Cend%7Baligned%7D+%5C%5C)





## FFM

FFM在FM的基础上增加了 “域”的概念，FM中一个特征只有一种隐向量的表达，FFM将特征按照事先的规则分为多个场(Field)，特征 ![[公式]](https://www.zhihu.com/equation?tex=x_i) 属于某个特定的场f。每个特征将被映射为多个隐向量 ![[公式]](https://www.zhihu.com/equation?tex=V_%7Bi1%7D%2C%E2%80%A6%2CV_%7Bif%7D) ，每个隐向量对应一个场。

当两个特征 ![[公式]](https://www.zhihu.com/equation?tex=x_i%2C+x_j) ,组合时，用对方对应的场对应的隐向量做内积:

![img](https://pic3.zhimg.com/v2-dc7400deb795c8605dd0280275b451bb_b.jpg)



## DeepFm

