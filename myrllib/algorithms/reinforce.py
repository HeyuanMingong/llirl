"""
The vanilla policy gradient method, REINFORCE 

[1] Yan Duan, et al., "Benchmarking Deep Reinforcement Learning for 
    Continuous Control", ICML 2016.
[2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
    2018 (http://incompleteideas.net/book/the-book-2nd.html).
"""

import torch
import numpy as np
from torch.nn.utils.convert_parameters import (vector_to_parameters, parameters_to_vector)
from torch.distributions.kl import kl_divergence
from torch.optim import Adam, SGD

from myrllib.utils.torch_utils import (weighted_mean, detach_distribution, weighted_normalize)
from myrllib.utils.optimization import conjugate_gradient
from myrllib.baselines.baseline import LinearFeatureBaseline

def norm_01(weights, epsilon=1e-10):
    return (weights-weights.min()+epsilon)/(weights.max()-weights.min()+epsilon)
def norm_sum1(weights, epsilon=1e-10):
    weights += epsilon
    return weights/weights.sum()*weights.size(0)

class REINFORCE(object):
    def __init__(self, policy, lr=1e-2, opt='sgd', baseline=None, tau=1.0, device='cpu'):
        self.tau = tau
        self.policy = policy 
        self.lr = lr
        if opt=='adam': self.opt = Adam(policy.parameters(), lr=lr)
        else: self.opt = SGD(policy.parameters(), lr=lr)

        if baseline == 'linear': self.baseline = LinearFeatureBaseline(policy.input_size)
        else: self.baseline = None 
        self.to(device)


    def step(self, episodes, clip=False, recurrent=False, seq_len=5):
        if self.baseline is None:
            returns = episodes.returns
        else:
            self.baseline.fit(episodes)
            values = self.baseline(episodes)
            advantages = episodes.gae(values, tau=self.tau)
            returns = weighted_normalize(advantages, weights=episodes.mask)

        if recurrent:
            ### (time_horizon, batch_size, state_dim)
            obs = episodes.observations
            log_probs = []
            for idx in range(obs.size(0)):
                if idx < seq_len: obs_seq = obs[:idx+1]
                else: obs_seq = obs[-seq_len+idx+1:idx+1]

                pi = self.policy(obs_seq)
                log_prob = pi.log_prob(episodes.actions[idx])
                log_probs.append(log_prob)
            log_probs = torch.stack(log_probs, axis=0)
        else:
            pi = self.policy(episodes.observations)
            log_probs = pi.log_prob(episodes.actions)

        if log_probs.dim() > 2: log_probs = torch.sum(log_probs, dim=2)

        loss = -weighted_mean(log_probs * returns, weights=episodes.mask)

        self.opt.zero_grad()
        loss.backward()
        if clip: torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()


    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        if self.baseline is not None: self.baseline.to(device, **kwargs)
        self.device = device



