import torch
from torch import nn


class LSTMCell(nn.LSTMCell):
    def forward(self, input, hx=None):
        if hx is not None:
            hx = (hx[:, :self.hidden_size], hx[:, :self.hidden_size])
        hx, cx = super().forward(input, hx)
        return torch.cat((hx, cx), 1)
