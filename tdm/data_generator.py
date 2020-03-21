from tdm.tree import Tree
import numpy as np
import pandas as pd
from multiprocessing import Process, JoinableQueue, Queue
from prefetch_generator import background
import random


def build_time_window(t, hist, max_window_size=8, gap='2 day'):
    t = np.array(t)
    hist = np.array(hist)
    delta = t[1:] - t[:-1]
    is_gap = delta < pd.Timedelta(gap)
    i = len(is_gap)
    max_len = 0
    history_window = []
    while len(history_window) < max_window_size:
        window = []
        window.append(hist[i])
        i -= 1
        while i >= 0 and is_gap[i]:
            window.append(hist[i])
            i -= 1
        history_window.append(window)
        max_len = max(len(window), max_len)
        if i < 0:
            break
    while len(history_window) < max_window_size:
        history_window.append([0])

    for i in range(len(history_window)):
        history_window[i] += [0] * (max_len - len(history_window[i]))
    return history_window, max_len



def train_data_generate_worker(queue: Queue, df_history, tree: Tree, neg_samples=600, n=1, window_size=8, seq_len=60, gap='2 day'):
    padding = 0
    for _, row in df_history.iterrows():
        batch_nodes = []
        batch_history = []
        batch_y = []
        # MAX_LEN = 0
        _history = row['history']
        index_set = list(range(8, len(_history) - 1))
        if len(index_set) > n:
            index_set = random.sample(index_set, k=n)
        for index in index_set:
            # history = row['history']
            # index = random.choice(range(max(int(len(history) * 0.5), 7), len(history) - 1))
            excluded_items = _history[index:]
            item_id = random.choice(excluded_items)
            history = _history[:index]
            # t = row['timestamp'][:index]

            # history, max_len = build_time_window(t, history, max_window_size=max_window_size, gap=gap)
            if len(history) > seq_len:
                history = history[-seq_len:]
            else:
                history = [0] * (seq_len - len(history)) + history
            # MAX_LEN = max(max_len, MAX_LEN)
            positive_nodes, negative_nodes = tree.generate_dataset(item_id, excluded_items=excluded_items, neg_samples=neg_samples)

            batch_history.append(np.reshape(history, (window_size, seq_len // window_size)))
            batch_nodes.append(positive_nodes + negative_nodes)
            batch_y.append([1] * len(positive_nodes) + [0] * len(negative_nodes))


        for i in range(len(batch_history)):
            # for j in range(len(batch_history[i])):
            #     batch_history[i][j] += [padding] * (MAX_LEN - len(batch_history[i][j]))
            array = np.array(batch_history[i])[None, :]
            array = np.repeat(array, len(batch_y[i]), 0)
            batch_history[i] = array
            batch_nodes[i] = np.array(batch_nodes[i])
            batch_y[i] = np.array(batch_y[i])

        batch_nodes = np.concatenate(batch_nodes, 0)
        batch_history = np.concatenate(batch_history, 0)
        batch_y = np.concatenate(batch_y, 0).astype(np.float32)
        # length = np.array(length)
        queue.put((batch_nodes, batch_history, batch_y))


@background(10)
def iterate_minibatches(df_history, tree: Tree, neg_samples=600, n=20, num_workers=4,
                        window_size=6, seq_len=60, gap='2 day'):
    n_hist = len(df_history)
    idx = np.arange(n_hist)
    np.random.shuffle(idx)
    df_history = df_history.iloc[idx]
    workers = []
    queue = Queue(1000)
    partition_size = n_hist // num_workers
    for i in range(num_workers):
        if i == num_workers - 1:
            partition = df_history.iloc[i * partition_size:]
        else:
            partition = df_history.iloc[i * partition_size: (i + 1) * partition_size]
        p = Process(target=train_data_generate_worker, args=(queue, partition, tree, neg_samples, n, window_size, seq_len, gap))
        p.daemon = True
        p.start()
        workers.append(p)
    for _ in range(n_hist):
        nodes, history, y = queue.get()
        yield nodes, history, y

    for p in workers:
        p.join()




@background()
def iterate_minibatches_val(df_history, window_size=6, seq_len=60, gap='2 day'):
    for _, row in df_history.iterrows():
        history = row['history']
        t = row['timestamp']
        mid = len(history) // 2
        ground_truth = history[mid:]
        history = history[:mid]
        if len(history) > seq_len:
            history = history[-seq_len:]
        else:
            history = [0] * (seq_len - len(history)) + history
        history = np.reshape(history, (window_size, seq_len // window_size))
        # t = t[:mid]
        # history, max_len = build_time_window(t, history, max_window_size=max_window_size, gap=gap)
        yield np.array(history)[None, :], ground_truth


if __name__ == '__main__':
    history_path = 'datasets/movielensHistory.pkl'
    item_path = 'datasets/movieSorted.npy'
    tree = Tree()
    tree.initialize_tree(np.load(item_path))
    df_history = pd.read_pickle(history_path)
    train = df_history[df_history.type == 'train']
    length = len(train)
    gen = iterate_minibatches(df_history, tree, neg_samples=200, n=2)
    for i in range(length):
        nodes, history, y = next(gen)
        print(nodes.shape)
        print(history.shape)
        print(y.shape)
        break