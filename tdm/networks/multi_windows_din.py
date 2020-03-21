import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import apex
from apex import amp
from torch.nn import init


class AttentionScore(nn.Module):
    def __init__(self, hidden_size=12):
        super(AttentionScore, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(hidden_size * 4, 36),
            nn.PReLU(),
            nn.Linear(36, 1)
        )

    def forward(self, query, key):
        '''
        :param query: (batch_size, hidden_size)
        :param key: (batch_size, window_size, seq_len, hidden_size)
        :return: (batch_size, window_size, seq_len, 1)
        '''
        query = query.unsqueeze(1).unsqueeze(1)
        query = query.repeat(1, key.size(1), key.size(2), 1)
        out = torch.cat([query, key, query - key, query * key], -1)
        out = self.dense(out)
        return out


class TimeWindowBlock(nn.Module):
    def __init__(self, hidden_size, weighted_att=False):
        super(TimeWindowBlock, self).__init__()
        self.attention_score = AttentionScore(hidden_size)
        self.weighted_att = weighted_att
        # init.kaiming_normal_(self.positional_embed_W)

    def forward(self, query, key, mask=None):
        """
        :param query: (batch_size, hidden_size)
        :param key: (batch_size, window_size, seq_len, hidden_size)
        :return: (batch_size, window_size, hidden_size)
        """
        key_score = self.attention_score(query, key)
        if mask is not None:
            key_score.masked_fill_(mask, -1e4 if self.weighted_att else 0)
        if self.weighted_att:
            key_score = torch.softmax(key_score, 2)
        out = key * key_score
        out = out.sum(2)
        return out


class _MWDin(nn.Module):
    def __init__(self, num_nodes, embed_size=24, num_windows=10, dnn_hidden_size=(128, 64, 24), weighted_att=True, dropout=0.4):
        super(_MWDin, self).__init__()
        self.embed = nn.Embedding(num_nodes + 1, embed_size, padding_idx=0)
        self.padding_idx = 0
        self.num_windows = num_windows
        self.time_window_block = TimeWindowBlock(embed_size, weighted_att=weighted_att)

        # for i in range(num_windows):
            # self.add_module('window_block{}'.format(i), TimeWindowBlock(embed_size, weighted_att=weighted_att))
        self.dnn = self.build_dnn(num_windows, embed_size, dnn_hidden_size, dropout)

    @staticmethod
    def build_dnn(num_windows, embed_size, dnn_hidden_size, dropout):
        dnn_hidden_size_in = num_windows * embed_size + embed_size
        hidden_layers = []
        for i, size in enumerate(dnn_hidden_size):
            if i == 0:
                hidden_layers.append(nn.Sequential(
                    nn.Linear(dnn_hidden_size_in, size, bias=False),
                    nn.BatchNorm1d(size),
                    nn.PReLU(),
                    nn.Dropout(dropout)
                ))
            else:
                hidden_layers.append(nn.Sequential(
                    nn.Linear(dnn_hidden_size[i - 1], size, bias=False),
                    nn.BatchNorm1d(size),
                    nn.PReLU(),
                    nn.Dropout(dropout)
                ))
        hidden_layers.append(nn.Linear(dnn_hidden_size[-1], 1))
        return nn.Sequential(*hidden_layers)

    def forward(self, query, key):
        """
        :param query: (batch_size,)
        :param key: (batch_size, num_windows, seq_len)
        :return: (batch_size, 1)
        """
        query = self.embed(query)
        mask = torch.eq(key, self.padding_idx)
        mask = mask.unsqueeze_(-1)
        key = self.embed(key)
        key_pooling = self.time_window_block(query, key, mask)
        key_pooling = key_pooling.view(query.size(0), -1)
        dnn_in = [query, key_pooling]

        # for i in range(self.num_windows):
        #     # window_block = getattr(self, 'window_block{}'.format(i))
        #     dnn_in.append(self.time_window_block(query, key[:, i], mask[:, i]))
        dnn_in = torch.cat(dnn_in, 1)
        logits = self.dnn(dnn_in)
        return logits.squeeze_(-1)


class MWDin(object):
    def __init__(self, num_nodes, embed_size=24, num_windows=9, dnn_hidden_size=(128, 64, 24), weighted_att=False, dropout=0.4, lr=1e-3, weight_decay=1e-6):
        self.net = _MWDin(num_nodes, embed_size=embed_size, num_windows=num_windows,
                          dnn_hidden_size=dnn_hidden_size, weighted_att=weighted_att, dropout=dropout).cuda()
        self.optimizer = Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        # self.net, self.optimizer = amp.initialize(self.net, self.optimizer, verbosity=False)

    def get_embeddings(self, items):
        with torch.autograd.no_grad():
            items = torch.from_numpy(items).long().cuda()
            embed = self.net.embed(items).data.cpu().numpy()
        return embed

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def fit_batch(self, query, key, y):
        self.optimizer.zero_grad()
        query = torch.from_numpy(query).cuda()
        key = torch.from_numpy(key).cuda()
        y = torch.from_numpy(y).cuda()
        logits = self.net(query, key)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, query, key):
        with torch.autograd.no_grad():
            query = torch.from_numpy(query).cuda()
            key = torch.from_numpy(key).cuda()
            logits = self.net(query, key)
            # prob = torch.sigmoid(logits)
            rank = torch.argsort(logits, descending=True)
            return logits.data.cpu().numpy(), rank.data.cpu().numpy()

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict['net'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def state_dict(self):
        return {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }



if __name__ == '__main__':
    query = torch.LongTensor([1, 2])
    key = torch.LongTensor([[[0,1, 1],[2,3, 1]], [[4,5, 7],[5, 6, 1]]])
    net = _MWDin(4142579, num_windows=2)
    out = net(query, key)
    out.detach()
    print(out.size())


