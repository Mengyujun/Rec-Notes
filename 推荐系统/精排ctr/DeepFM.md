# DeepFm

DeepFM模型是将Wide部分替换为了FM模型，增强了模型的低阶特征交互的能力

模型图：

![img](https://pic2.zhimg.com/v2-e668f7ba05aad55f684e06af5a19a45d_b.jpg)

目标是**共同学习低阶和高阶特征交互**



### wide部分

即为FM模型 ，学习低阶特征交叉 



![image-20201118215337066](https://i.loli.net/2020/11/18/GbHqAVzNip1daEX.png)

FM部分公式为：输出由两部分组成：一个 **Addition Unit**，多个 **内积单元**。

**Addition Unit** 反映的是1阶的特征。**内积单元**反映的是2阶的组合特征对于预测结果的影响。

![image-20201118215412124](https://i.loli.net/2020/11/18/DEorZ79VJejpqC1.png)

### **Deep部分**

Deep部分是一个前向传播的神经网络，用来学习高阶特征交互。

![image-20201118215447848](https://i.loli.net/2020/11/18/TxGfSEDuVmckQY3.png)

就是常规 的mlp



### **Output层**

FM层与Deep层的输出相拼接，最后通过一个逻辑回归返回最终的预测结果：

![[公式]](https://www.zhihu.com/equation?tex=\hat+y%3Dsigmoid(y_{FM}%2By_{DNN})+\\)

DeepFM模型的优势：

1. 端到端；不需要任何预训练
2. 同时学习低阶特征交叉和高阶特征交叉
3. Wide部分与Deep部分共享参数 **共享feature embedding**；无需手动设计特征工程（相对于w&d中 wide部分需要手动设计交叉特征）



### deepfm常见面试问题：

1. deepfm的wide & deep部分的输入是什么呢，分别输入什么特征好

   - FM 和 Deep 部分**共享Embedding 层**，embedding训练得到的参数既作为 wide 部分的输出，也作为 MLP 部分的输入 
   - 一方面是和wide&deep一样的情况 还是可以考虑 对离散型特征 和 连续型特征分开讨论  
   - 连续型特征的处理： 用其**连续值**或者**离散化one hot encoding向量**表示
   - 对于实际输入， 如果是sparse特征，先进行embedding化，之后拼接， 若果是dense特征，则直接输入

   **感觉靠谱的答案：**

   **dnn部分**： dense数据 以及  经过embedding处理的sparse数据 **concat**之后的 数据输入

   **wide部分（）**：**sparse feature的embedding**  以及 dense数据经过离散化，embedding之后的embedding向量 一起参与fm的交叉

2. dnn可以学到特征交叉，为什么还需要fm侧  

   参考-DNN可以进行高阶特征交互，为什么Wide&Deep和DeepFM等模型仍然需要显式构造Wide部分？ - 王鸿伟的回答 - 知乎 https://www.zhihu.com/question/364517083/answer/961734991 

   - 从wide和deep的 记忆能力和泛化能力展开 

   - 理论上来说DNN可以拟合任意函数，实际从3个方面来提高拟合能力：

     - 一是根据具体的问题场景提出更好的模型，这些特定的模型会比较适合各自的场景，从而降低拟合难度，比如各类CNN，RNN，GNN等。
     - 二是提出更好的优化方法，使得学习拟合的过程更快速高效。
     - 三是提取更好的特征，从而在源头上降低拟合的难度，提升性能的上限。

     在推荐系统中，特征提取的影响非常大，因此显示的选择一些特征，可以降低拟合难度

   - Wide&Deep和FM里手动构造二阶项，也只是想为DNN提供更多的“**输入素材”**，让DNN可以更好地发挥而已。这些二阶项能不能真的提高模型性能，并没有理论上的保证，因为这取决于具体的推荐系统场景中是否真的有很多这种二阶相关性