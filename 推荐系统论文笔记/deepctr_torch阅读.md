dnn部分的特征其实可以分为 sparse 以及 varlen_sparse(即序列)



即构建dict 为 name到位置的映射  build_input_features 

```
# OrderedDict: {feature_name:(start, start+dimension)} 
self.feature_index = build_input_features(
     linear_feature_columns + dnn_feature_columns)
```



注意 构建 index和位置之间的映射时，同时保存了 长度的名称？ feat.length_name？你的输入和模型之间的输入是如何一一对应在一起的  ？ 

```
        elif isinstance(feat, VarLenSparseFeat):
            features[feat_name] = (start, start + feat.maxlen)
            start += feat.maxlen
            if feat.length_name is not None and feat.length_name not in features:
                features[feat.length_name] = (start, start + 1)
                start += 1
```



知道了 变长序列是一种处理方式；历史点击序列是变长序列的子集， 在deepctr中如果是历史点击序列了，就不走 普通变长序列的max或者mean的处理方式，而是去使用attention来做 

即 普通的变长序列和 历史点击序列是不同的处理方式 

然后历史点击序列走的 是 **query&key**的方式



**测试一下 VarLenSparseFeat.name 是个啥？** 

所有的测试din特征 run_din中  主要观察 特征名称有哪些 

['user', 'gender', 'item', 'item_gender', 'score', 'hist_item', 'seq_length', 'hist_item_gender'] 



重新梳理一下流程： 

- 普通的sequence 是用普通的mean或者sum pooling 
- history_sequence 是用attention来进行操作