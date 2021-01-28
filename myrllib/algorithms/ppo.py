import torch
import numpy as np
from myrllib.utils.torch_utils import (weighted_mean, detach_distribution, weighted_normalize)
from myrllib.baselines.baseline import LinearFeatureBaseline 
from torch.optim import Adam, SGD
from torch.distributions import Normal


class PPO(object):
    def __init__(self, policy, epochs=5, clip=0.2, opt='adam', lr=1e-3,
            baseline=None, tau=1.0, device='cpu'):
        self.policy = policy
        self.baseline = baseline
        self.tau = tau 
        self.epochs = epochs
        self.clip = clip
        self.lr = lr
        
        self.opt = Adam(self.policy.parameters(), lr=lr) if opt=='adam' else \
                SGD(self.policy.parameters(), lr=lr)
        self.baseline = LinearFeatureBaseline(policy.input_size) \
                if baseline=='linear' else None

        self.to(device)


    def step(self, episodes, recurrent=False, seq_len=5, clip=False):
        if self.baseline is None:
            advantages = episodes.returns
        else:
            self.baseline.fit(episodes)
            values = self.baseline(episodes)
            advantages = episodes.gae(values, tau=self.tau)
            advantages = weighted_normalize(advantages, weights=episodes.mask)

        if recurrent:
            ### (time_horizon, batch_size, state_dim)
            obs = episodes.observations
            locs, scales = [], []
            for idx in range(obs.size(0)):
                if idx < seq_len: obs_seq = obs[:idx+1]
                else: obs_seq = obs[-seq_len+idx+1:idx+1]

                pi_ = self.policy(obs_seq)
                locs.append(pi_.loc); scales.append(pi_.scale)
            loc = torch.stack(locs, axis=0)
            scale = torch.stack(scales, axis=0)
            pi = Normal(loc=loc, scale=scale)
        else:
            pi = self.policy(episodes.observations)
        old_pi = detach_distribution(pi)
        
        for epoch in range(self.epochs):
            if recurrent:
                ### (time_horizon, batch_size, state_dim)
                obs = episodes.observations
                locs, scales = [], []
                for idx in range(obs.size(0)):
                    if idx < seq_len: obs_seq = obs[:idx+1]
                    else: obs_seq = obs[-seq_len+idx+1:idx+1]

                    pi_ = self.policy(obs_seq)
                    locs.append(pi_.loc)
                    scales.append(pi_.scale)

                loc = torch.stack(locs, axis=0)
                scale = torch.stack(scales, axis=0)
                pi = Normal(loc=loc, scale=scale)
            else:
                pi = self.policy(episodes.observations)

            log_ratio = pi.log_prob(episodes.actions) - old_pi.log_prob(episodes.actions)
            if log_ratio.dim() > 2: log_ratio = torch.sum(log_ratio, dim=2)
            ratio = torch.exp(log_ratio)

            loss = advantages * ratio
            clipped_ratio = torch.clamp(ratio, 1.0-self.clip, 1.0+self.clip)
            clipped_loss = advantages * clipped_ratio

            total_loss = - weighted_mean(torch.min(loss, clipped_loss), weights=episodes.mask)

            self.opt.zero_grad()
            total_loss.backward()
            if clip: torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()


    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        if self.baseline is not None: self.baseline.to(device, **kwargs)
        self.device = device
