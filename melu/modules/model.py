import tensorflow as tf
from melu.modules.feature import SparseFeature
from tensorflow.contrib.layers import xavier_initializer


class MeLU(object):
    def __init__(self, features, batch_size=2, hiddens=(128, 64, 32), loss_type='mse', num_local_update=4,
                 local_lr=5e-6, global_lr=5e-5, reuse=False, train=True):
        assert loss_type in ('mse', 'bce')
        self.features = features
        self.hiddens = hiddens
        self.loss_type = loss_type
        self.reuse = reuse
        self.num_local_updates = num_local_update
        self.local_lr = local_lr
        self.global_lr = global_lr
        sup_placeholders = []
        query_placeholders = []
        self.batch_size = batch_size
        for i in range(batch_size):
            sup_placeholders.append(self.generate_placeholders('sup_'.format(i)))
            query_placeholders.append(self.generate_placeholders('query_'.format(i)))
        self.sup_placeholders = sup_placeholders
        self.query_placeholders = query_placeholders

        self.global_weights, self.local_weights = self.generate_weights(reuse)
        if train:
            self.train_loss, self.train_step = self.global_update()
        else:
            self.sup_infer_placeholders = self.generate_placeholders('sup_infer')
            self.query_infer_placeholders = self.generate_placeholders('query_infer')
            self.grad_score = self.get_grad_score()
            self.infer_result = self.infer()


    def infer(self):
        weights = self.local_update(self.sup_infer_placeholders)
        pred, _ = self.forward(self.query_infer_placeholders, weights, return_loss=False)
        return pred


    def get_grad_score(self):
        weights = dict(self.global_weights)
        score = 0
        for i in range(self.num_local_updates):
            _, loss = self.forward(self.sup_infer_placeholders, weights)
            loss = loss / tf.stop_gradient(tf.norm(loss))
            grad = tf.gradients(loss, [v for v in self.local_weights.values()])
            for g in grad:
                score += tf.norm(g)
            for k, g in zip(self.local_weights.keys(), grad):
                weights[k] = weights[k] - self.local_lr * g
        return score / self.num_local_updates


    def local_update(self, sup_placeholders):
        weights = dict(self.global_weights)
        for i in range(self.num_local_updates):
            pred, loss = self.forward(sup_placeholders, weights)
            grad = tf.gradients(loss, [v for v in self.local_weights.values()])
            for k, g in zip(self.local_weights.keys(), grad):
                weights[k] = weights[k] - self.local_lr * g
        return weights

    def global_update(self):
        loss = []
        for sup_placeholder, query_placeholders in zip(self.sup_placeholders, self.query_placeholders):
            weights = self.local_update(sup_placeholder)
            _, l = self.forward(query_placeholders, weights)
            loss.append(l)
        loss = tf.stack(loss, 0)
        loss = tf.reduce_mean(loss)
        # grad = tf.gradients(loss, [v for v in self.global_weights.values()])
        # return grad
        train_step = tf.train.AdamOptimizer(self.global_lr).minimize(loss)
        return loss, train_step

    def generate_placeholders(self, prefix='sup'):
        placeholders = {}
        target_placeholder = tf.placeholder(tf.float32, (None,), '{}_target'.format(prefix))
        placeholders['target'] = target_placeholder
        for f in self.features:
            name = f.name
            if f.one_hot:
                input = tf.placeholder(tf.float32, (None, f.vocab_size), '{}_input_{}'.format(prefix, name))
            else:
                input = tf.placeholder(tf.int32, (None,), '{}_input_{}'.format(prefix, name))
            placeholders[name] = input
        return placeholders

    def generate_weights(self, reuse=False):
        global_weights = {}
        local_weights = {}
        first_dim = 0
        for f in self.features:
            first_dim += f.dim
            with tf.variable_scope(f.name, reuse=reuse):
                weights = tf.get_variable(f.name, (f.vocab_size, f.dim), tf.float32, initializer=xavier_initializer())
                global_weights['weights_embed_{}'.format(f.name)] = weights

        hiddens = [first_dim] + list(self.hiddens)
        for i in range(len(hiddens)):
            if i == 0:
                continue
            with tf.variable_scope('dense_{}'.format(i), reuse=reuse):
                weights = tf.get_variable('weights_{}'.format(i), (hiddens[i - 1], hiddens[i]), dtype=tf.float32,
                                          initializer=xavier_initializer())
                bias = tf.get_variable('bias_{}'.format(i), (hiddens[i],), dtype=tf.float32,
                                       initializer=tf.initializers.zeros())
                global_weights['weights_{}'.format(i)] = weights
                local_weights['weights_{}'.format(i)] = weights
                global_weights['bias_{}'.format(i)] = bias
                local_weights['bias_{}'.format(i)] = bias
        with tf.variable_scope('dense_output', reuse=reuse):
            weights = tf.get_variable('weights_output', (hiddens[-1], 1), dtype=tf.float32,
                                      initializer=xavier_initializer())
            bias = tf.get_variable('bias_output', (1,), dtype=tf.float32,
                                   initializer=tf.initializers.zeros())
            global_weights['weights_output'] = weights
            local_weights['weights_output'] = weights
            global_weights['bias_output'] = bias
            local_weights['bias_output'] = bias
        return global_weights, local_weights


    def forward(self, placeholders, weights, return_loss=True):
        embedding_output = []
        for f in self.features:
            name = f.name
            x = placeholders[name]
            if f.one_hot:
                embedding_output.append(tf.matmul(x, weights['weights_embed_{}'.format(name)]) / tf.reduce_sum(x))
            else:
                embedding_output.append(tf.nn.embedding_lookup(weights['weights_embed_{}'.format(name)], x))

        output = tf.concat(embedding_output, 1)

        for i in range(len(self.hiddens) + 1):
            if i == 0:
                continue
            output = tf.matmul(output, weights['weights_{}'.format(i)]) + weights['bias_{}'.format(i)]
            output = tf.nn.relu(output)
        output = tf.matmul(output, weights['weights_output']) + weights['bias_output']
        output = tf.reshape(output, (-1,))
        if return_loss:
            if self.loss_type == 'mse':
                loss = tf.losses.mean_squared_error(placeholders['target'], output)
            else:
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=output, logits=placeholders['target'])
            return output, loss
        else:
            return output, None


    def generate_train_feed_dict(self, input_sup, target_sup, input_query, target_query):
        feed_dict = {}
        for i in range(self.batch_size):
            feed_dict[self.sup_placeholders[i]['target']] = target_sup[i]
            feed_dict[self.query_placeholders[i]['target']] = target_query[i]
            for k, v in input_sup[i].items():
                feed_dict[self.sup_placeholders[i][k]] = v
            for k, v in input_query[i].items():
                feed_dict[self.query_placeholders[i][k]] = v
        return feed_dict

    def generate_infer_feed_dict(self, placeholders, input, y):
        feed_dict = {}
        for k, v in input.items():
            feed_dict[placeholders[k]] = v
        feed_dict[placeholders['target']] = y
        return feed_dict






if __name__ == '__main__':
    feat1 = SparseFeature(20, 16, 'feat1')
    feat2 = SparseFeature(20, 16, 'feat2', one_hot=False)
    net = MeLU([feat1, feat2])
    # net = MeLu([feat1, feat2], loss='bce', reuse=True)
    # print(net.output)

