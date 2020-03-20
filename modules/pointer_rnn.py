import torch
from torch import nn
import torch.nn.functional as F
from modules.mhsa import Encoder
from torch.autograd import Variable
from modules.rnn import LSTMCell
from torch.nn import init


class PointerRNN(nn.Module):
    def __init__(self, users, user_dim, items, item_dim, heads=1, layers=1, rnn_cell=LSTMCell):
        super(PointerRNN, self).__init__()
        self.embed_user = nn.Embedding(users, user_dim)
        self.embed_item = nn.Embedding(items, item_dim)
        # self.linear_input = nn.Sequential(
        #     nn.Linear(user_dim + item_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim)
        # )
        self.hidden_dim = user_dim + item_dim
        hidden_dim = self.hidden_dim
        self.encoder = Encoder(hidden_dim, heads, layers)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.rnn_cell = rnn_cell(hidden_dim, hidden_dim)
        if isinstance(self.rnn_cell, nn.LSTMCell):
            self.W_D1 = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
            self.W_D3 = nn.Linear(3 * hidden_dim, hidden_dim, bias=False)
        else:
            self.W_D1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.W_D3 = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)

        self.W_D2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias1 = nn.Parameter(torch.Tensor(hidden_dim))
        self.V_D1 = nn.Linear(hidden_dim, 1, bias=False)


        self.W_D4 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias2 = nn.Parameter(torch.Tensor(hidden_dim))
        self.V_D2 = nn.Linear(hidden_dim, 1, bias=False)
        self._init_parameters()

    def _init_parameters(self):
        init.zeros_(self.bias1)
        init.zeros_(self.bias2)

    def _rnn_init_state(self, batch_size, device):
        if isinstance(self.rnn_cell, nn.LSTMCell):
            hidden_size = self.hidden_dim * 2
        else:
            hidden_size = self.hidden_dim
        return torch.zeros(batch_size, self.hidden_dim).to(device), torch.zeros(batch_size, hidden_size).to(device)

    def rnn_forward(self, input, hidden, target_embed=None):
        """
        :param enc_output: (batch_size, hidden_dim)
        :param target_embed: (seq_len, batch_size, hidden_dim)
        :return: (seq_len or 1, batch_size, hidden_dim)
        """
        # batch_size, hidden_dim = input.size()
        hidden = self.rnn_cell(input, hidden)
        hs = [hidden]

        if target_embed is not None:
            for i in range(1, target_embed.size(0)):
                hidden = self.rnn_cell(target_embed[i], hidden)
                hs.append(hidden)
        hs = torch.stack(hs, 0)
        return hs

    def glimpse_attention(self, dec_output, enc_output):
        """
        :param dec_output: (seq_len_dec, batch_size, hidden_dim)
        :param enc_output: (seq_len_enc, batch_size, hidden_dim)
        :return: (seq_len2, batch_size, hidden_dim * 2)
        """
        dec_output_ = self.W_D1(dec_output).unsqueeze_(1)
        enc_output_ = self.W_D2(enc_output).unsqueeze_(0)
        e = dec_output_ + enc_output_ + self.bias1  # (seq_len2, seq_len1, batch_size, hidden_dim)
        e = torch.tanh(e)
        e = self.V_D1(e).squeeze_(-1)  # (seq_len2, seq_len1, batch_size)
        e = torch.softmax(e, 1)
        out = torch.einsum('ijb,jbh->ibh', [e, enc_output])
        out = torch.cat((dec_output, out), -1)
        return out

    def pointer_attention(self, dec_output, enc_output, mask=None, output_prob=False):
        """
        :param dec_output: (seq_len_dec, batch_size, hidden_dim * 2)
        :param enc_output: (seq_len_enc, batch_size, hidden_dim)
        :param mask: (batch_size, seq_len_dec, seq_len_enc)
        :return: (batch_size, seq_len_dec, seq_len_enc)
        """
        dec_output_ = self.W_D3(dec_output).unsqueeze_(1)
        enc_output_ = self.W_D4(enc_output).unsqueeze_(0)
        e = dec_output_ + enc_output_ + self.bias2
        e = torch.tanh(e)
        e = self.V_D2(e).squeeze_(-1)
        e = e.permute(2, 0, 1)
        if mask is not None:
            e = e.masked_fill_(mask, -float('inf'))
        if output_prob:
            e = torch.softmax(e, -1)
        return e

    def encode(self, users, items, mask_enc):
        candidates_embed = self.embed_item(items)
        users_embed = self.embed_user(users).unsqueeze(1).repeat(1, candidates_embed.size(1), 1)
        candidates_embed = torch.cat([candidates_embed, users_embed], 2)
        candidates_embed = self.encoder(candidates_embed, mask_enc)  # (n_candidates, batch_size, hidden_dim)
        return candidates_embed

    def forward(self, users, candidates, targets_idx, mask_enc=None, mask_dec=None, output_prob=False):
        '''
        :param users: (batch_size,)
        :param candidates: (batch_size, n_candidates)
        :param targets_idx: (batch_size, k)
        :return: (batch_size, k, n_candidates)
        '''
        # targets = torch.gather(candidates, 1, targets_idx)
        candidates_embed = self.encode(users, candidates, mask_enc)
        enc_output = candidates_embed
        # enc_output = self.ff(enc_output)
        enc_output = enc_output.permute(1, 0, 2)
        targets_idx_ = targets_idx.unsqueeze(-1).repeat(1, 1, self.hidden_dim)
        targets_embed = torch.gather(candidates_embed, 1, targets_idx_)
        targets_embed = targets_embed.permute(1, 0, 2)

        input, hidden = self._rnn_init_state(users.size(0), users.device)
        dec_output = self.rnn_forward(input, hidden, target_embed=targets_embed)

        dec_output_glimpse = self.glimpse_attention(dec_output, enc_output)

        output = self.pointer_attention(dec_output_glimpse, enc_output, mask_dec, output_prob)
        output = output.contiguous()
        return output

    def sample(self, users, candidates, depth=10, mask_enc=None):
        batch_size, n_candidates = candidates.size()
        candidates_embed = self.encode(users, candidates, mask_enc)
        enc_output = candidates_embed
        enc_output = enc_output.permute(1, 0, 2)
        input, hidden = self._rnn_init_state(users.size(0), users.device)
        if mask_enc is not None:
            mask_enc_ = torch.zeros(batch_size, n_candidates, dtype=torch.bool).to(users.device)
        mask_dec = torch.zeros(batch_size, n_candidates, dtype=torch.bool).to(users.device)
        output_card_idx = []
        output_logits = []
        for i in range(depth):
            if mask_enc is not None:
                mask_ = mask_enc_ | mask_dec
            else:
                mask_ = mask_dec
            mask_ = mask_.unsqueeze(1)

            hidden = self.rnn_forward(input, hidden)
            dec_output_glimpse = self.glimpse_attention(hidden, enc_output)

            logits = self.pointer_attention(dec_output_glimpse, enc_output, mask=None, output_prob=False).contiguous()
            max_logit = logits.max().detach()
            min_logit = logits.min().detach()
            logits = torch.min(logits, (max_logit + 1 + mask_.float() * (min_logit - 10000 - max_logit)))
            # logits = torch.min(logits, logits + mask_.float() * (logits - 10000))
            prob = torch.softmax(logits, -1).squeeze(1)
            sample_idx = torch.multinomial(prob, 1)
            input = torch.gather(candidates_embed, 1, sample_idx.unsqueeze(-1).repeat(1, 1, self.hidden_dim))
            # mask_dec = mask_dec.index_put((indices, sample_idx.squeeze()), fill_val)
            # mask_dec = mask_dec.scatter_(1, sample_idx.squeeze(), True)
            mask_dec.scatter_(1, sample_idx, True)
            if mask_enc is not None:
                mask_enc_ = mask_enc.gather(1, sample_idx.view(batch_size, 1, 1)
                                            .repeat(1, 1, n_candidates)).squeeze_(1)
            output_card_idx.append(sample_idx)
            output_logits.append(logits)
            input = input.squeeze(1)
            hidden = hidden.squeeze(0)
        output_card_idx = torch.cat(output_card_idx, 1)
        output_logits = torch.cat(output_logits, 1)

        return output_card_idx, output_logits


    def get_logprob(self, enc_output, input, hidden, mask_dec=None):
        dec_output = self.rnn_forward(input, hidden)
        dec_output_glimpse = self.glimpse_attention(dec_output, enc_output)
        output = self.pointer_attention(dec_output_glimpse, enc_output, mask_dec, False)
        return output - torch.logsumexp(output, -1, keepdim=True), dec_output


    def beam_search(self, users, candidates, beam=2, depth=10, mask_enc=None):
        '''
        :param users: (batch_size,)
        :param candidates: (batch_size, n_candidates)
        :param beam:
        :param depth:
        :param mask: (batch_size, n_candidates, n_candidates)
        :return: (batch_size, beam, depth)
        '''
        batch_size, n_candidates = candidates.size()
        candidates_embed = self.encode(users, candidates, mask_enc)
        enc_output = candidates_embed
        enc_output = enc_output.permute(1, 0, 2)
        device = candidates_embed.device
        accumulate_logprob = torch.zeros((batch_size * beam, 1)).to(device)
        accumulate_logprob.fill_(-float('inf'))
        accumulate_logprob.index_fill_(0, torch.arange(0, beam * batch_size, beam).to(device), 0.0)
        input, hidden = self._rnn_init_state(batch_size * beam, device)
        enc_output = enc_output.unsqueeze(2).repeat(1, 1, beam, 1).view(n_candidates, -1, self.hidden_dim)
        mask_dec_ = torch.zeros(batch_size * beam, n_candidates).bool().to(candidates_embed.device)
        if mask_enc:
            mask_enc_ = torch.zeros(batch_size * beam, n_candidates, dtype=torch.bool).to(device)
        pos_index = (torch.arange(batch_size) * beam).view(-1, 1).to(device)
        predecessors_list = []
        idx_list = []
        for i in range(depth):
            if mask_enc:
                mask_ = mask_enc_ | mask_dec_
            else:
                mask_ = mask_dec_
            logprob, hidden = self.get_logprob(enc_output, input, hidden, mask_dec=mask_.unsqueeze(1))
            hidden = hidden.squeeze(0)
            logprob = logprob.squeeze_(1)
            accumulate_logprob = accumulate_logprob.repeat(1, n_candidates)
            accumulate_logprob += logprob
            accumulate_logprob = accumulate_logprob.view(batch_size, -1)
            top_logprob, top_idx = accumulate_logprob.topk(beam, 1)
            input_idx = (top_idx % n_candidates).view(batch_size * beam, 1)
            input = candidates_embed.gather(1, input_idx.view(batch_size, -1)
                                                .unsqueeze(-1).repeat(1, 1, self.hidden_dim)).view(-1, self.hidden_dim)
            accumulate_logprob = top_logprob.view(batch_size * beam, 1)

            predecessors = (top_idx / n_candidates + pos_index.expand_as(top_idx)).view(-1, 1)
            mask_dec_ = mask_dec_.index_select(0, predecessors.squeeze())
            mask_dec_.scatter_(1, input_idx, True)
            # mask_dec_.index_put_((indices, input_idx.squeeze()), fill_val)
            if mask_enc is not None:
                mask_enc_ = mask_enc.gather(1, input_idx.view(batch_size, beam, 1)
                                            .repeat(1, 1, n_candidates)).view(-1, n_candidates)
            hidden = hidden.index_select(0, predecessors.squeeze())
            predecessors_list.append(predecessors.squeeze())
            idx_list.append(input_idx.squeeze())
        output = self.backtrack(predecessors_list, idx_list)
        output = output.view(-1, batch_size, beam)
        output = output.permute(1, 2, 0)
        output = candidates.gather(1, output.reshape(batch_size, -1)).view(batch_size, beam, -1)
        return output

    def backtrack(self, predecessors_list, idx_list):
        output = []
        output.append(idx_list.pop())
        predecessors = predecessors_list.pop()
        while len(idx_list) > 0:
            idx = idx_list.pop()
            idx = idx.index_select(0, predecessors)
            output.append(idx)
            t_predecessors = predecessors_list.pop()
            predecessors = t_predecessors.index_select(0, predecessors)
        return torch.stack(output, 0)


















if __name__ == '__main__':
    net = PointerRNN(100, 24, 100, 24)
    # enc_final_output = torch.randn(3, 24)
    # target_embed = torch.randn(10, 3, 24)
    # print(net.rnn_forward(enc_final_output, target_embed).size())
    #
    # enc_output = torch.randn(20, 3, 24)
    # dec_output = torch.randn(10, 3, 48)
    #
    # print(net.pointer_attention(dec_output, enc_output).size())
    # net.eval()
    user = torch.LongTensor([1, 2, 3])
    candidates = torch.arange(60).view(3, 20)
    targets = torch.LongTensor([[0, 1, 2, 4], [3, 1, 4, 4], [3, 1, 4, 4]])
    # # mask_enc = torch.randn(3, 5, 5) > 0
    output = net.beam_search(user, candidates, depth=10, beam=5, mask_enc=None)
    # print(output.size())
    print(net.sample(user, candidates, mask_enc=torch.randn(3, 20, 20) > 0)[0].size())
    # print(net.sample(user, candidates))