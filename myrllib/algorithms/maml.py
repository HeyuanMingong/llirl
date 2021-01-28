import torch
from torch.optim import Adam, SGD
from torch.nn.utils.convert_parameters import (vector_to_parameters, parameters_to_vector)
from torch.distributions.kl import kl_divergence

from myrllib.utils.torch_utils import (weighted_mean, detach_distribution, weighted_normalize)
from myrllib.utils.optimization import conjugate_gradient
from myrllib.baselines.baseline import LinearFeatureBaseline

class MetaLearner(object):
    def __init__(self, sampler, policy, baseline=None, gamma=0.95, lr=0.02, 
                 fast_lr=0.02, tau=1.0, device='cpu', opt='adam'):
        self.sampler = sampler
        self.policy = policy
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.lr = lr
        self.tau = tau

        self.opt = Adam(self.policy.parameters(), lr=lr) if opt=='adam' else \
                SGD(self.policy.parameters(), lr=lr)
        self.baseline = LinearFeatureBaseline(self.policy.input_size) \
                if baseline=='linear' else None

        self.to(device)

    def adapt(self, episodes, first_order=False):
        if self.baseline is None:
            advantages = episodes.returns
        else:
            self.baseline.fit(episodes)
            values = self.baseline(episodes)
            advantages = episodes.gae(values, tau=self.tau)
            advantages = weighted_normalize(advantages, weights=episodes.mask)

        pi = self.policy(episodes.observations)
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2: log_probs = torch.sum(log_probs, dim=2)
        loss = -weighted_mean(log_probs * advantages, weights=episodes.mask)

        # Get the new parameters after a one-step gradient update
        params = self.policy.update_params(loss, step_size=self.fast_lr, first_order=first_order)
        return params

    def sample(self, tasks, first_order=False):
        """Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """
        episodes = []
        for task in tasks:
            self.sampler.reset_task(task)
            train_episodes = self.sampler.sample(self.policy, gamma=self.gamma, device=self.device)

            params = self.adapt(train_episodes, first_order=first_order)

            valid_episodes = self.sampler.sample(self.policy, params=params,
                gamma=self.gamma, device=self.device)
            episodes.append((train_episodes, valid_episodes))
        return episodes

    def step(self, episodes):
        losses = []
        for (train_episodes, valid_episodes) in episodes:
            if self.baseline is None:
                advantages = valid_episodes.returns
            else:
                self.baseline.fit(valid_episodes)
                values = self.baseline(valid_episodes)
                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages, weights=valid_episodes.mask)

            params = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)

            log_probs = pi.log_prob(valid_episodes.actions)
            if log_probs.dim() > 2: log_probs = torch.sum(log_probs, dim=2)

            loss = -weighted_mean(log_probs * advantages, weights=valid_episodes.mask)
            losses.append(loss)

        total_loss = torch.mean(torch.stack(losses, dim=0))
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()


    def step_ppo(self, episodes, epochs=5, ppo_clip=0.2, clip=False):
        advantages, old_pis = [], []
        for (train_episodes, valid_episodes) in episodes:
            if self.baseline is None:
                advantage = valid_episodes.returns
            else:
                self.baseline.fit(valid_episodes)
                values = self.baseline(valid_episodes)
                advantage = valid_episodes.gae(values, tau=self.tau)
                advantage = weighted_normalize(advantage, weights=valid_episodes.mask)
            advantages.append(advantage)

            params = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)
            old_pis.append(detach_distribution(pi))

        for epoch in range(epochs):
            losses, clipped_losses, masks = [], [], []
            for idx, (train_episodes, valid_episodes) in enumerate(episodes):
                params = self.adapt(train_episodes)
                pi = self.policy(valid_episodes.observations, params=params)

                log_ratio = pi.log_prob(valid_episodes.actions) - \
                        old_pis[idx].log_prob(valid_episodes.actions)
                if log_ratio.dim()>2: log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)
                loss = advantages[idx] * ratio

                clipped_ratio = torch.clamp(ratio, 1.0-ppo_clip, 1.0+ppo_clip)
                clipped_loss = advantages[idx] * clipped_ratio

                losses.append(loss); clipped_losses.append(clipped_loss)
                masks.append(valid_episodes.mask)
            
            losses = torch.cat(losses, dim=0)
            clipped_losses = torch.cat(clipped_losses, dim=0)
            masks = torch.cat(masks, dim=0)

            total_loss = - weighted_mean(torch.min(losses, clipped_losses), weights=masks)

            self.opt.zero_grad()
            total_loss.backward()
            if clip:torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()


    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        if self.baseline is not None: self.baseline.to(device, **kwargs)
        self.device = device


    def kl_divergence(self, episodes, old_pis=None):
        kls = []
        if old_pis is None: old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None: old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2: mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def hessian_vector_product(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""
        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    def surrogate_loss(self, episodes, old_pis=None):
        losses, kls, pis = [], [], []
        if old_pis is None: old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt(train_episodes)
            with torch.set_grad_enabled(old_pi is None):
                pi = self.policy(valid_episodes.observations, params=params)
                pis.append(detach_distribution(pi))

                if old_pi is None: old_pi = detach_distribution(pi)

                if self.baseline is None:
                    advantages = valid_episodes.returns
                else:
                    self.baseline.fit(valid_episodes)
                    values = self.baseline(valid_episodes)
                    advantages = valid_episodes.gae(values, tau=self.tau)
                    advantages = weighted_normalize(advantages, weights=valid_episodes.mask)

                log_ratio = (pi.log_prob(valid_episodes.actions) \
                        - old_pi.log_prob(valid_episodes.actions))
                if log_ratio.dim() > 2: log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss = -weighted_mean(ratio * advantages, weights=valid_episodes.mask)
                losses.append(loss)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2: mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(pi, old_pi), weights=mask)
                kls.append(kl)

        return (torch.mean(torch.stack(losses, dim=0)),
                torch.mean(torch.stack(kls, dim=0)), pis)

    def step_trpo(self, episodes, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        old_loss, _, old_pis = self.surrogate_loss(episodes)
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes, damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads, cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step, self.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes, old_pis=old_pis)
            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < max_kl): break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())

   
    
