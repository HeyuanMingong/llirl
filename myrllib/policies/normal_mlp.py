import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from collections import OrderedDict
from myrllib.policies.policy import Policy, weight_init

class NormalMLPPolicy(Policy):
    def __init__(self, input_size, output_size, hidden_sizes=(),
                 nonlinearity=F.relu, init_std=1.0, min_std=1e-6, max_std=1e6):
        super(NormalMLPPolicy, self).__init__(input_size=input_size, output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.max_log_std = math.log(max_std)
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.mu = nn.Linear(layer_sizes[-1], output_size)

        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))
        self.apply(weight_init)

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output, weight=params['layer{0}.weight'.format(i)],
                    bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        mu = F.linear(output, weight=params['mu.weight'], bias=params['mu.bias'])
        #mu = torch.tanh(mu)
        #scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std, max=self.max_log_std))

        return Normal(loc=mu, scale=scale)


class NormalLSTMPolicy(Policy):
    def __init__(self, input_size, output_size, nonlinearity=F.relu, bs=16,
            init_std=1.0, min_std=1e-6, max_std=1e6, device='cpu'):
        super(NormalLSTMPolicy, self).__init__(input_size=input_size, output_size=output_size)
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.max_log_std = math.log(max_std)

        self.embed_size = 200
        self.hidden_size = 200
        self.device = device
        self.bs = bs
        self.embed = nn.Linear(input_size, self.embed_size).to(device)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size).to(device)
        self.mu = nn.Linear(self.hidden_size, output_size).to(device)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))
        self.apply(weight_init)
        self.init_hidden()

    def init_hidden(self):
        ### (num_layers*num_directions, batch_size, hidden_size)
        h0 = torch.from_numpy(np.zeros((1, self.bs, 
            self.hidden_size))).float().to(self.device)
        c0 = torch.from_numpy(np.zeros((1, self.bs, 
            self.hidden_size))).float().to(self.device)
        self.hidden = (h0, c0)

    def forward(self, inputs):
        self.init_hidden()
        ### input: (seq_len, batch_size, dim)
        ### embedding: (seq_len, batch_size, embed_size)
        embedding = F.relu(self.embed(inputs))

        ### lstm_out: (seq_len, batch_size, hidden_size)
        lstm_out, (ht, ct) = self.lstm(embedding, self.hidden)

        ### out: (batch_size, output_size)
        mu = self.mu(F.relu(lstm_out[-1]))
        scale = torch.exp(torch.clamp(self.sigma, min=self.min_log_std, max=self.max_log_std))

        return Normal(loc=mu, scale=scale)
