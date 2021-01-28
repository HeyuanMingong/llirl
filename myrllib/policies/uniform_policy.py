import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform

from collections import OrderedDict
from myrllib.policies.policy import Policy, weight_init

class UniformPolicy(Policy):
    def __init__(self, input_size, output_size, low=None, high=None):
        super(UniformPolicy, self).__init__(
            input_size=input_size, output_size=output_size)
        if low is None:
            print('Please provide the action space...')
        self.low = torch.FloatTensor(low) 
        self.high = torch.FloatTensor(high) 

    def forward(self, input, params=None):
        batch_size = input.size(0)
        low = torch.stack([self.low for _ in range(batch_size)],dim=0)
        high = torch.stack([self.high for _ in range(batch_size)],dim=0)

        return Uniform(low, high)
