from tdm.tree import TreeNode
import numpy as np
from tdm.data_generator import iterate_minibatches_val
from tdm.metric import precision, recall, f1, novelty
from tqdm import tqdm
from tdm.networks.multi_windows_din import MWDin


def predict(model: MWDin, root:  TreeNode, history, k=200):
    candidates = [root]
    history = np.array(history, dtype='int')
    result = []
    i = 0
    while len(candidates) > 0:
        i += 1
        candidates_updated = []
        for node in candidates:
            if node.is_leaf:
                result.append(node.id)
            else:
                candidates_updated.append(node)
        candidates = candidates_updated
        if len(candidates) == 0:
            break
        query = np.array([node.id for node in candidates], dtype='int')
        key = np.repeat(history, query.shape[0], 0)
        _, rank = model.predict(query, key)

        topk = np.array(candidates)[rank[:k]]

        candidates_updated = []

        for node in topk:
            if node.left is not None:
                candidates_updated.append(node.left)
            if node.right is not None:
                candidates_updated.append(node.right)
        candidates = candidates_updated
    # result = [r for r in result if r[0] == 15]
    result = np.array(result)
    # print([r[0] for r in result])
    key = np.repeat(history, result.shape[0], 0)
    _, rank = model.predict(result, key)
    result = result[rank[:k]]
    # result = list(np.array(result)[rank[:k]])
    # print([r[0] for r in result])
    return result


def validate(val, model, root: TreeNode, k=200, window_size=6, seq_len=6, gap='2 day'):
    n_user = len(val)
    generator = iterate_minibatches_val(val, window_size=window_size, seq_len=seq_len, gap=gap)
    tbar = tqdm(range(n_user))
    PRECISION = 0
    RECALL = 0
    F1 = 0
    NOVELTY = 0

    for i in tbar:
        history, ground_truth = next(generator)
        fetch = predict(model, root, history, k=k)

        PRECISION += precision(ground_truth, fetch)
        RECALL += recall(ground_truth, fetch)
        F1 += f1(ground_truth, fetch)
        NOVELTY += novelty(ground_truth, fetch)
        tbar.set_description('precision: %.3f, recall: %.3f, f1: %.3f, novelty: %.3f' % (
            PRECISION / (i + 1), RECALL / (i + 1), F1 / (i + 1), NOVELTY / (i + 1)
        ))
    return {
        'precision': PRECISION / n_user,
        'recall': RECALL / n_user,
        'f1': F1 / n_user,
        'novelty': NOVELTY / n_user
    }





if __name__ == '__main__':
    pass


