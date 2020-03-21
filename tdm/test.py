import torch
from tdm.networks.multi_windows_din import MWDin
from tdm.prediction import validate
import pandas as pd
import pickle
import os


if __name__ == '__main__':
    df_history_path = 'datasets/movielensHistory.pkl'
    item_path = 'datasets/movieSorted.npy'
    checkpoints_dir = '/data/data_dyh/recsys_data/'
    resume = 7
    df_history = pd.read_pickle(df_history_path)
    df_val = df_history[df_history.type == 'test']
    # all_items = np.load(item_path)
    with open(os.path.join(checkpoints_dir, 'tree-epoch{}.pkl'.format(resume)), 'rb') as f:
        tree = pickle.load(f)
    net = MWDin(tree.n_nodes, num_windows=10, weighted_att=False)
    net.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'model-epoch{}.pth'.format(resume, 30000))))
    net.eval()
    metric = validate(df_val, net, tree.root, k=10, window_size=10, seq_len=70, gap='30 min')
    print(metric)
