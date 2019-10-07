import numpy as np
from torch import nn
import torch.nn.functional as f


# a simple RNN model using LSTM

from models.auto_encoder import AutoEncoder


class SimpleRNN(nn.Module):

    def __init__(self, **kwargs):
        super(SimpleRNN, self).__init__()

        # define parameters
        self.input_size = kwargs['in_size']
        self.rnn_hidden_size = kwargs['rnn_h_size']
        self.output_size = kwargs['out_size']
        self.num_layers = kwargs.get('num_layers', 1)

        # rnn layer
        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True)

        # output layer
        self.out = nn.Linear(in_features=self.rnn_hidden_size, out_features=self.output_size)

        # initialize weights
        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x):
        y, _ = self.rnn(x)
        y_last = y[:, -1, :]
        y_last = self.out(y_last)
        return y_last


# an RNN Regression model using LSTM

class RNNRegr(nn.Module):

    def __init__(self, **kwargs):
        super(RNNRegr, self).__init__()

        # define parameters
        self.input_size = kwargs['in_size']
        self.rnn_hidden_size = kwargs['rnn_h_size']
        self.reg_hidden_sizes = kwargs['reg_h_sizes']
        self.output_size = kwargs['out_size']
        self.num_layers = kwargs.get('num_layers', 1)
        self.p_dropout = kwargs.get('p_dropout', 0.0)

        # rnn layer
        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.rnn_hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True)

        # regression layer
        self.reg = nn.ModuleList()
        for k in range(len(self.reg_hidden_sizes) - 1):
            self.reg.append(nn.Linear(self.reg_hidden_sizes[k], self.reg_hidden_sizes[k + 1]))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.p_dropout)

        # output layer
        self.out = nn.Linear(in_features=self.reg_hidden_sizes[-1], out_features=self.output_size)

    def forward(self, x):

        y, _ = self.rnn(x)
        y_t = y[:, -1, :]

        for layer in self.reg:
            y_t = self.relu(layer(y_t))
            y_t = self.dropout(y_t)

        return self.out(y_t)


# an RNN Regression model using LSTM with AutoEncoder

class AutoEncoderRNNRegr(nn.Module):

    def __init__(self, **kwargs):
        super(AutoEncoderRNNRegr, self).__init__()

        # define parameters
        self.input_size = kwargs['in_size']
        self.rnn_input_size = kwargs['rnn_in_size']
        self.rnn_hidden_size = kwargs['rnn_h_size']
        self.reg_hidden_sizes = kwargs['reg_h_sizes']
        self.output_size = kwargs['out_size']
        self.num_layers = kwargs.get('num_layers', 1)
        self.p_dropout = kwargs.get('p_dropout', 0.0)

        # auto_encoder layer
        self.ae = AutoEncoder(**kwargs)
        if kwargs.get('ae_pretrain_weight'):
            self.ae.load_state_dict(kwargs['ae_pretrain_weight'])

        if kwargs.get('if_trainable'):
            for p in self.ae.parameters():
                p.requires_grad = kwargs['if_trainable']
        else:
            self.ae.weight.requires_grad = False

        # rnn layer
        self.rnn = nn.LSTM(input_size=self.rnn_input_size,
                           hidden_size=self.rnn_hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True)

        # regression layer
        self.reg = nn.ModuleList()
        for k in range(len(self.reg_hidden_sizes) - 1):
            self.reg.append(nn.Linear(self.reg_hidden_sizes[k], self.reg_hidden_sizes[k + 1]))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.p_dropout)

        # output layer
        self.out = nn.Linear(in_features=self.reg_hidden_sizes[-1], out_features=self.output_size)

    def forward(self, x):

        n_samples, seq_len, _ = x.shape
        en_x = x.view(n_samples * seq_len, -1)
        en_x, _ = self.ae(en_x)
        en_x = en_x.view(n_samples, seq_len, -1)  # (batch_size, seq_len, h_size)

        y, _ = self.rnn(en_x)
        y_t = y[:, -1, :]

        for layer in self.reg:
            y_t = self.relu(layer(y_t))
            y_t = self.dropout(y_t)

        return self.out(y_t)
