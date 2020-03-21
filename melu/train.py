import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from melu.modules.model import MeLU, SparseFeature
import tensorflow as tf
from melu.config import config
import random
import pickle
import numpy as np
from tqdm import tqdm

def collect(x):
    output = {}
    for k in x[0].keys():
        output[k] = []
    for t in x:
        for k, v in t.items():
            output[k].append(v)
    for k, v in output.items():
        output[k] = np.concatenate(output[k], 0)
    output['rate'] = np.reshape(output['rate'], (-1,))
    output['gender'] = np.reshape(output['gender'], (-1,))
    output['age'] = np.reshape(output['age'], (-1,))
    output['occupation'] = np.reshape(output['occupation'], (-1,))
    output['zipcode'] = np.reshape(output['zipcode'], (-1,))
    return output

if __name__ == '__main__':
    path = './datasets'
    features = []
    features.append(SparseFeature(config['num_rate'], config['embedding_dim'], 'rate', one_hot=False))
    features.append(SparseFeature(config['num_genre'], config['embedding_dim'], 'genre'))
    features.append(SparseFeature(config['num_director'], config['embedding_dim'], 'director'))
    features.append(SparseFeature(config['num_actor'], config['embedding_dim'], 'actor'))
    features.append(SparseFeature(config['num_gender'], config['embedding_dim'], 'gender', one_hot=False))
    features.append(SparseFeature(config['num_age'], config['embedding_dim'], 'age', one_hot=False))
    features.append(SparseFeature(config['num_occupation'], config['embedding_dim'], 'occupation', one_hot=False))
    features.append(SparseFeature(config['num_zipcode'], config['embedding_dim'], 'zipcode', one_hot=False))
    batch_size = config['batch_size']
    melu = MeLU(features, batch_size=batch_size, hiddens=(config['first_fc_hidden_dim'],
                                   config['second_fc_hidden_dim']),
                num_local_update=config['inner'], local_lr=config['local_lr'], global_lr=config['lr'])

    save_path = './checkpoints'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())


        training_set_size = int(len(os.listdir("{}/warm_state".format(path))) / 4)
        supp_xs_s = []
        supp_ys_s = []
        query_xs_s = []
        query_ys_s = []
        for idx in range(training_set_size):
            supp_xs_s.append(pickle.load(open("{}/warm_state/supp_x_{}.pkl".format(path, idx), "rb")))
            supp_ys_s.append(pickle.load(open("{}/warm_state/supp_y_{}.pkl".format(path, idx), "rb")))
            query_xs_s.append(pickle.load(open("{}/warm_state/query_x_{}.pkl".format(path, idx), "rb")))
            query_ys_s.append(pickle.load(open("{}/warm_state/query_y_{}.pkl".format(path, idx), "rb")))
        total_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
        del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)


        for j in range(config['num_epoch']):
            random.shuffle(total_dataset)
            num_batch = int(training_set_size / batch_size)
            a, b, c, d = zip(*total_dataset)
            tbar = tqdm(range(num_batch))
            losses = 0
            for i in tbar:
                supp_xs = list(a[batch_size * i:batch_size * (i + 1)])
                if len(supp_xs) < batch_size:
                    continue
                supp_ys = list(b[batch_size * i:batch_size * (i + 1)])
                query_xs = list(c[batch_size * i:batch_size * (i + 1)])
                query_ys = list(d[batch_size * i:batch_size * (i + 1)])
                batch_support_x = []
                batch_support_y = []
                batch_query_x = []
                batch_query_y = []
                for _, (s_xs, s_ys, q_xs, q_ys) in enumerate(zip(supp_xs, supp_ys, query_xs, query_ys)):
                    batch_support_x.append(collect(s_xs))
                    batch_support_y.append(s_ys)
                    batch_query_x.append(collect(q_xs))
                    batch_query_y.append(q_ys)
                loss, _ = sess.run([melu.train_loss, melu.train_step],
                                   feed_dict=melu.generate_train_feed_dict(batch_support_x, batch_support_y,
                                                                           batch_query_x, batch_query_y))
                losses += loss
                tbar.set_description('epoch: %d, loss: %.3f' % (j + 1, losses / (i + 1)))





        tf.train.Saver().save(sess, save_path=save_path)



