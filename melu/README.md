# MeLU
An unofficial TensorFlow implementation of paper [MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation](https://arxiv.org/pdf/1908.00413v1.pdf).


# 说明
- `preprocess.py`：预处理参照[官方实现](https://github.com/hoyeoplee/MeLU.git)
- `movielens`数据：[官方实现](https://github.com/hoyeoplee/MeLU.git)
- 训练：端到端的方式实现，无中间过程保存。
- NDCG指标：NDCG实现方式似乎版本比较多，官方实现没有给出，尝试了几种实现方式似乎都远远达不到论文给出的结果。
本实现方式如下，仅供参考：
```python
import numpy as np
def NDCG(pred, target, k=1):
    sorted_idx = np.argsort(target)[::-1][:k]
    idcg = 0
    for r, idx in enumerate(sorted_idx):
        idcg += (2 ** target[idx] - 1) / np.log2(1 + r + 1)
    # the rank of item in predicted list, for example: [4, 3, 2, 9, 7] -> [2, 3, 4, 0, 1]
    pred_rank = np.argsort(np.argsort(pred)[::-1])
    dcg = 0
    for r, idx in enumerate(sorted_idx):
        # 'pred_rank[idx]' is the predicted rank of the item 'idx'
        dcg += (2 ** target[idx] - 1) / np.log2(1 + pred_rank[idx] + 1)
    return dcg / idcg
```
## Reference
```
@inproceedings{lee2019melu,
  title={MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation},
  author={Lee, Hoyeop and Im, Jinbae and Jang, Seongwon and Cho, Hyunsouk and Chung, Sehee},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1073--1082},
  year={2019},
  organization={ACM}
}
```