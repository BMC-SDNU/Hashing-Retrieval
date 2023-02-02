import math
import torch
import torch.nn as nn

from collections import OrderedDict

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        pos_encoder = PositionalEncoding(d_model)
    Refer:
        https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MLP(nn.Module):
    def __init__(self, hidden_dim=[1000, 2048, 512], act=nn.Tanh()):
        super(MLP, self).__init__()
        self.input_dim = hidden_dim[0]
        self.hidden_dim = hidden_dim

        orderedDict = OrderedDict()
        for i in range(len(hidden_dim) - 1):
            index = i + 1
            orderedDict['linear' + str(index)] = nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1])
            orderedDict['bn' + str(index)] = nn.BatchNorm1d(self.hidden_dim[i + 1])
            orderedDict['act' + str(index)] = act

        self.mlp = nn.Sequential(orderedDict)
        # self._initialize()

    def _initialize(self):
        nn.init.xavier_normal_(self.mlp.linear1.weight.data)
        nn.init.xavier_normal_(self.mlp.linear2.weight.data)

    def forward(self, x):
        return self.mlp(x)

