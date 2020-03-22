import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from melu.modules.model import MeLU, SparseFeature
import tensorflow as tf
from melu.config import config
import random
import pickle
from melu.metric import MAE, NDCG
import numpy as np
from tqdm import tqdm




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
                num_local_update=config['inner'], local_lr=config['local_lr'], global_lr=config['lr'], train=False)

    save_path = './checkpoints/'

    # state = 'user_and_item_cold_state'
    state = 'user_cold_state'

    test_set_size = int(len(os.listdir("{}/{}".format(path, state))) / 4)
    supp_xs_s = []
    supp_ys_s = []
    query_xs_s = []
    query_ys_s = []
    for idx in range(test_set_size):
        supp_xs_s.append(pickle.load(open("{}/{}/supp_x_{}.pkl".format(path, state, idx), "rb")))
        supp_ys_s.append(pickle.load(open("{}/{}/supp_y_{}.pkl".format(path, state, idx), "rb")))
        query_xs_s.append(pickle.load(open("{}/{}/query_x_{}.pkl".format(path, state, idx), "rb")))
        query_ys_s.append(pickle.load(open("{}/{}/query_y_{}.pkl".format(path, state, idx), "rb")))
    total_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
    del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)

    saver = tf.train.Saver(tf.global_variables())
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    save_path = './checkpoints/'
    with tf.Session(config=tf_config) as sess:
        saver.restore(sess, tf.train.latest_checkpoint(save_path))
        # sess.run(tf.global_variables_initializer())
        # sess.run(tf.global_variables_initializer())
        tbar = tqdm(total_dataset)
        mae_total = 0
        ndcg1_total = 0
        ndcg3_total = 0
        for i, (supp_xs, supp_ys, query_xs, query_ys) in enumerate(tbar):
            feed_dict = melu.generate_infer_feed_dict(melu.sup_infer_placeholders, supp_xs, supp_ys)
            feed_dict.update(melu.generate_infer_feed_dict(melu.query_infer_placeholders, query_xs, query_ys))
            pred = sess.run(melu.infer_result, feed_dict=feed_dict)
            mae_total += MAE(pred, query_ys)
            ndcg1_total += NDCG(pred, query_ys, k=1)
            ndcg3_total += NDCG(pred, query_ys, k=3)
            tbar.set_description('MAE: %.3f, NDCG@1: %.3f, NDCG@3: %.3f' %
                                 (mae_total / (i + 1), ndcg1_total / (i + 1), ndcg3_total / (i + 1)))


