import torch
from torch import nn



class MultiHeadSelfAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, heads, dropout=0.1):
        super(MultiHeadSelfAttentionLayer, self).__init__()
        assert hidden_dim % heads == 0
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, hidden_dim)
        self.heads = heads
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, X, mask=None):
        '''
        :param X: (batch_size, seq_len, hidden_dim)
        :param mask: (batch_size, seq_len, seq_len)
        :return: (batch_size, seq_len, hidden_dim)
        '''
        batch_size, seq_len, hidden_dim = X.size()
        Q = self.W_Q(X).view(batch_size, seq_len, self.heads, hidden_dim // self.heads)
        K = self.W_K(X).view(batch_size, seq_len, self.heads, hidden_dim // self.heads)
        V = self.W_V(X).view(batch_size, seq_len, self.heads, hidden_dim // self.heads)

        e = torch.einsum('bihd,bjhd->bihj', [Q, K])
        e /= (hidden_dim // self.heads) ** 0.5
        if mask is not None:
            mask = mask.unsqueeze(2)
            e = e.masked_fill_(mask, -float('inf'))
        e = torch.softmax(e, -1)
        out = torch.einsum('bihj,bjhd->bihd', [e, V])
        out = out.reshape(batch_size, seq_len, -1)
        out = self.W_out(out)
        out = self.norm1(X + self.dropout1(out))
        out = self.norm2(out + self.dropout2(self.ff(out)))
        return out


class Encoder(nn.Module):
    def __init__(self, hidden_dim, heads, layers, dropout=0.1):
        assert hidden_dim % heads == 0
        super(Encoder, self).__init__()
        self.encode_layers = nn.ModuleList(
            [MultiHeadSelfAttentionLayer(hidden_dim, heads, dropout) for _ in range(layers)]
        )

    def forward(self, X, mask=None):
        '''
        :param X: (batch_size, seq_len, hidden_dim)
        :param mask: (batch_size, seq_len, seq_len)
        :return: (seq_len, batch_size, hidden_dim)
        '''
        for l in self.encode_layers:
            X = l(X, mask)
        return X

if __name__ == '__main__':
    x = torch.randn(2, 10, 128)
    net = Encoder(128, 8, 4)
    print(net(x).size())



