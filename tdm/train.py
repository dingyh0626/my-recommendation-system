import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from tdm.data_generator import iterate_minibatches
from tdm.tree import Tree
import torch
import numpy as np
from tdm.networks.multi_windows_din import MWDin
from tdm.prediction import validate
import pandas as pd
from tqdm import tqdm
import pickle


def reset_embedding(weight, tree: Tree):
    for level in tree.levels[1:]:
        for node in level:
            if node.is_leaf:
                continue
            else:
                node_id = node.id
                c = node.c
                node.c = None
                c = torch.from_numpy(c).float().cuda()
                weight.data[node_id].copy_(c)


def train(df_history_path='./datasets/movielensHistory.pkl',
          item_path='./datasets/movieSorted.npy', checkpoints_dir='/data/data_dyh/recsys_data/',
          neg_samples=100, num_epochs=10, window_size=10, seq_len=70, gap='30 min', k=10, resume=0, n=80):
    df_history = pd.read_pickle(df_history_path)
    df_train = df_history[df_history.type == 'train']
    df_val = df_history[df_history.type == 'val']
    n_iters = len(df_train['history'])
    # n_iters = int(np.ceil(df_train['history'].apply(lambda x: (len(x) - min_seq_len)).sum() / n))
    all_items = np.load(item_path)
    tree = Tree()
    tree.initialize_tree(all_items)
    net = MWDin(tree.n_nodes, num_windows=window_size, weighted_att=False, dropout=0.4)
    if resume > 0:
        net.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'model-epoch{}.pth'.format(resume + 1))))
        with open(os.path.join(checkpoints_dir, 'tree-epoch{}.pkl'.format(resume + 1)), 'rb') as f:
            tree = pickle.load(f)

    for epoch in range(0, num_epochs):
        if epoch == 2:
            tree = Tree()
            embed = net.get_embeddings(all_items)
            tree.cluster_tree(all_items, embed)
            del net
            torch.cuda.empty_cache()
            net = MWDin(tree.n_nodes, num_windows=window_size, weighted_att=False, dropout=0.4)
            # reset_embedding(net.net.embed.weight, tree)

        generator = iterate_minibatches(df_train, tree,
                                        neg_samples=neg_samples, num_workers=4,
                                        window_size=window_size, seq_len=seq_len, gap=gap, n=n)
        tbar = tqdm(range(n_iters))
        losses = 0
        net.train()
        for i in tbar:
            nodes, history, y = next(generator)
            loss = net.fit_batch(nodes, history, y)
            losses += loss
            tbar.set_description('Epoch: {}/{}, Loss: {:.3f}'.format(epoch + 1, num_epochs, losses / (i + 1)))
            if (i + 1) % 10000 == 0:
                net.eval()
                metric = validate(df_val, net, tree.root, k=k, window_size=window_size, seq_len=seq_len, gap=gap)
                print('epoch: %d, iter: %d, precision: %.3f, recall: %.3f, f1: %.3f, novelty: %.3f' % (
                    epoch + 1, i + 1, metric['precision'], metric['recall'], metric['f1'], metric['novelty']
                ))
                net.train()
                torch.save(net.state_dict(), os.path.join(checkpoints_dir, 'model-epoch{}-iter{}.pth'.format(epoch + 1, i + 1)))

        net.eval()
        metric = validate(df_val, net, tree.root, k=k, window_size=window_size, seq_len=seq_len, gap=gap)
        print('epoch: %d, precision: %.3f, recall: %.3f, f1: %.3f, novelty: %.3f' % (
            epoch + 1, metric['precision'], metric['recall'], metric['f1'], metric['novelty']
        ))

        torch.save(net.state_dict(), os.path.join(checkpoints_dir, 'model-epoch{}.pth'.format(epoch + 1)))
        with open(os.path.join(checkpoints_dir, 'tree-epoch{}.pkl'.format(epoch + 1)), 'wb') as f:
            pickle.dump(tree, f)


if __name__ == '__main__':
    train(resume=0)
