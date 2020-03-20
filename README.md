# Exact-K
An unofficial PyTorch implementation of paper “Exact-K Recommendation via Maximal Clique Optimization”

## 说明
数据（`data`目录以及`data_load_ml.py`）及超参（`hyperparams.py`）来源于[官方实现](https://github.com/pangolulu/exact-k-recommendation.git)。网络按照论文描述的大体框架进行搭建，细节有一定出入。

## 细节

- `PonterRNN.forward`：RNN部分使用上一步的ground truth作为输入，
类似NLP里的生成模型。按照原文的意思，这种训练方式在inference阶段，如果上一步
预测错误，会导致这一步也发生错误。在论文中对应`Learning from Demonstrations`经测试movelens的precison@10大约在0.3多一点。
- `PointerRNN.sample`：RNN部分根据上一步的softmax采样得到item作为这一步的
输入。在论文中对应`SUPERVISE with Policy-sampling`。
- `PointerRNN.beam_search`：批量的beam search实现，
参考[pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq.git)，细节有所改造，主要是需要更新mask用于屏蔽出现过的item。
- 强化学习：按照论文，通过sample得到的card整体应该有个得分，即card中存在
被点击的item时reward应为1，不存在则为-1，然后通过policy gradient进行学习。
经测试movelens的precison@10大概在0.486左右，原文是0.4815。

## TODO
原文使用另一个CTR预估网络来估计card被点击的概率，作为reward来进行强化学习。

## Reference
```
@inproceedings{gong2019exact,
  title={Exact-k recommendation via maximal clique optimization},
  author={Gong, Yu and Zhu, Yu and Duan, Lu and Liu, Qingwen and Guan, Ziyu and Sun, Fei and Ou, Wenwu and Zhu, Kenny Q},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={617--626},
  year={2019}
}
```


