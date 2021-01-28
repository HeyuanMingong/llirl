"""
Implement the trust region policy optimization algorithm

[1] John Schulman, et al., "Trust Region Policy Optimization", ICML, 2015.
[2] Yan Duan, et al., "Benchmarking Deep Reinforcement Learning 
    for Continuous Control", ICML, 2016.
"""

import torch
import numpy as np
from torch.nn.utils.convert_parameters import (vector_to_parameters, parameters_to_vector)
from torch.distributions.kl import kl_divergence
from torch.optim import Adam, SGD

from myrllib.utils.torch_utils import (weighted_mean, detach_distribution, weighted_normalize)
from myrllib.utils.optimization import conjugate_gradient
from myrllib.baselines.baseline import LinearFeatureBaseline 


class TRPO(object):
    def __init__(self, policy, baseline=None, tau=1.0, device='cpu', 
            max_kl=1e-3, cg_iters=10, cg_damping=1e-2, 
            ls_max_steps=10, ls_backtrack_ratio=0.5):
        self.policy = policy
        if baseline == 'linear': self.baseline = LinearFeatureBaseline(policy.input_size)
        else: self.baseline = None 
        self.tau = tau 
        self.max_kl = max_kl; self.cg_iters = cg_iters 
        self.cg_damping = cg_damping; self.ls_max_steps = ls_max_steps 
        self.ls_backtrack_ratio = ls_backtrack_ratio 
        self.to(device)

    def kl_divergence(self, episodes, old_pi=None):
        pi = self.policy(episodes.observations)
        if old_pi is None: old_pi = detach_distribution(pi)

        mask = episodes.mask 
        if episodes.actions.dim() > 2: mask = mask.unsqueeze(2)

        kl = weighted_mean(kl_divergence(pi, old_pi), weights=mask)
        return kl 

    def hessian_vector_product(self, episodes, damping=1e-2):
        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            grad2s_copy = []
            for item in grad2s:
                item = item.contiguous()
                grad2s_copy.append(item)
            grad2s = tuple(grad2s_copy)
            flat_grad2_kl = parameters_to_vector(grad2s)
            return flat_grad2_kl + damping * vector 
        return _product 

    def surrogate_loss(self, episodes, old_pi=None):
        with torch.set_grad_enabled(old_pi is None):
            if self.baseline is None:
                advantages = episodes.returns 
            else:
                self.baseline.fit(episodes)
                values = self.baseline(episodes)
                advantages = episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages, weights=episodes.mask)

            pi = self.policy(episodes.observations)
            if old_pi is None: old_pi = detach_distribution(pi)
            
            log_ratio = pi.log_prob(episodes.actions) - old_pi.log_prob(episodes.actions)
            if log_ratio.dim() > 2: log_ratio = torch.sum(log_ratio, dim=2)
            ratio = torch.exp(log_ratio)
            loss = -weighted_mean(ratio * advantages, weights=episodes.mask)

            mask = episodes.mask 
            if episodes.actions.dim() > 2: mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), weights=mask)

        return loss, kl, pi 


    def step(self, episodes):
        max_kl = self.max_kl; cg_iters = self.cg_iters 
        cg_damping = self.cg_damping; ls_max_steps = self.ls_max_steps 
        ls_backtrack_ratio = self.ls_backtrack_ratio 

        old_loss, _, old_pi = self.surrogate_loss(episodes)
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        # compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes, damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads, cg_iters=cg_iters)

        # computer the Lagrange multiplier
        shs = 0.5*torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)
        step = stepdir / lagrange_multiplier 

        # save the old parameters 
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size*step, self.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes, old_pi=old_pi)
            improve = loss - old_loss 

            if (improve.item() < 0.0) and (kl.item() < max_kl): break 
            step_size *= ls_backtrack_ratio 
        else:
            vector_to_parameters(old_params, self.policy.parameters())
    

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        if self.baseline is not None: self.baseline.to(device, **kwargs)
        self.device = device



