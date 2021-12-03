尝试做一个修改 



将deepfm修改为 双塔模式 将deep部分的输出和fm部分的输出 同时输入一个全连接网络中 去最终学习deep和fm的加权和 



同时一阶项的问题， 要不要共享embedding？ 还是单独的embedding 

> 根据原始输入得到一阶交互特征，根据embedding向量得到二阶交互特征向量
>
> 正确么？ 



DeepFM技术细节 - panda爱瞎想的文章 - 知乎 https://zhuanlan.zhihu.com/p/148836639

注意其中的 公式中的 vi*xi 即将原始输入 转换成embedding的过程 

```
vi*xi * vj*xj = <vi,vj>xi*xj 
# vi*xi 就是做embedding的过程 ’
# 因此dense层可以看作是一个乘矩阵V的操作，实现了embedding的目的。
```

![image-20210928145430070](https://i.loli.net/2021/09/28/ujU381v52FzqGST.png)



目前的 一阶项就是 做一个1维的embedding 其实也就是一个全连接层 然后求和 因此现有代码没毛病 只是比较简单， 不同field的内部依据index 找到对应的 embedding（只有1维 不就是全连接层么-不同特征值有一个自己的系数） 最后系数相加 就是1阶和 



[DeepFM(3) torch实现 - 木叶流云 - 博客园 (cnblogs.com)](https://www.cnblogs.com/leimu/p/14606583.html)



共享权重 是如何更新的呢？ 会出现相互干扰么？ 



先不考虑 一阶项的问题 尝试用不同的





deepfm 的问题 竟然是 优化器的问题？  换不同的优化器的结果肯可能大不相同 

下次出现这种问题 可以考虑**打印出梯度** 观察一下 不同部分的 梯度可能是不同的  观察梯度的方法？ 





k is mlp.mlp.0.weight torch.Size([128, 705])
k is mlp.mlp.0.bias torch.Size([128])
k is mlp.mlp.1.alpha torch.Size([128])
k is mlp.mlp.1.bn.weight torch.Size([128])
k is mlp.mlp.1.bn.bias torch.Size([128])
k is mlp.mlp.1.bn.running_mean torch.Size([128])
k is mlp.mlp.1.bn.running_var torch.Size([128])
k is mlp.mlp.1.bn.num_batches_tracked torch.Size([])
k is mlp.mlp.2.weight torch.Size([64, 128])
k is mlp.mlp.2.bias torch.Size([64])
k is mlp.mlp.3.alpha torch.Size([64])
k is mlp.mlp.3.bn.weight torch.Size([64])
k is mlp.mlp.3.bn.bias torch.Size([64])
k is mlp.mlp.3.bn.running_mean torch.Size([64])
k is mlp.mlp.3.bn.running_var torch.Size([64])
k is mlp.mlp.3.bn.num_batches_tracked torch.Size([])
k is mlp.mlp.4.weight torch.Size([32, 64])
k is mlp.mlp.4.bias torch.Size([32])
k is mlp.mlp.5.alpha torch.Size([32])
k is mlp.mlp.5.bn.weight torch.Size([32])
k is mlp.mlp.5.bn.bias torch.Size([32])
k is mlp.mlp.5.bn.running_mean torch.Size([32])
k is mlp.mlp.5.bn.running_var torch.Size([32])
k is mlp.mlp.5.bn.num_batches_tracked torch.Size([])
k is mlp.mlp.6.weight torch.Size([1, 32])
k is mlp.mlp.6.bias torch.Size([1])
k is embedding.embedding.weight torch.Size([1000000, 15])
k is linear.bias torch.Size([1])
k is linear.fc.weight torch.Size([1000000, 1])
k is mlp.mlp.0.weight torch.Size([64, 705])
k is mlp.mlp.0.bias torch.Size([64])
k is mlp.mlp.2.weight torch.Size([32, 64])
k is mlp.mlp.2.bias torch.Size([32])
k is mlp.mlp.4.weight torch.Size([1, 32])
k is mlp.mlp.4.bias torch.Size([1])
k is embedding.embedding.weight torch.Size([1000000, 15])



单独dfm的参数是： 

k is linear.bias torch.Size([1])
k is linear.fc.weight torch.Size([2000000, 1])
k is mlp.mlp.0.weight torch.Size([64, 705])
k is mlp.mlp.0.bias torch.Size([64])
k is mlp.mlp.2.weight torch.Size([32, 64])
k is mlp.mlp.2.bias torch.Size([32])
k is mlp.mlp.4.weight torch.Size([1, 32])
k is mlp.mlp.4.bias torch.Size([1])
k is embedding.embedding.weight torch.Size([2000000, 15])



dfm使用dice是没有bn相关参数的 但是dnn使用dice就会多出很多参数 （？）



解决了一个bug 就是无法复制 自动扩容的问题 主要是参数名称的问题 因此修改一下就好了 



dfm 模型尝试优化调参 

