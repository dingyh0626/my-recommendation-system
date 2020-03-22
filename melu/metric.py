import numpy as np


def MAE(pred, target):
    return np.mean(np.fabs(np.subtract(pred, target)))


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



if __name__ == '__main__':
    print(NDCG([3, 2, 3, 0, 1, 2], [2, 3, 3, 0, 1, 2], k=6))