import torch
from torch import nn
import torch.nn.functional as F
from myrllib.policies.policy import weight_init
from collections import OrderedDict
import numpy as np


### construct the model to parameterize the environment
class EnvModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(), nonlinearity=F.relu):
        super(EnvModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1
        
        layer_sizes = (input_size, ) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        self.mu = nn.Linear(layer_sizes[-1], output_size)
        self.apply(weight_init)
    
    def forward(self, input, params=None):
        if params is None: params = OrderedDict(self.named_parameters())
        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output, weight=params['layer{0}.weight'.format(i)],
                    bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        mu = F.linear(output, weight=params['mu.weight'], bias=params['mu.bias'])
        return mu


### construct the input and output data for the environment model
def construct_env_io(episodes, env_type='reward', H=4):
    def construct_single_episode(episode):
        s_traj, a_traj, r_traj, ns_traj, id_traj = \
                episode.s_traj, episode.a_traj, episode.r_traj, episode.ns_traj, episode.id_traj

        batch_size, time_length = s_traj.shape[0], s_traj.shape[1]
        states, actions, rewards, nstates = [], [], [], []
        for idx in range(batch_size):
            s_, a_, r_, ns_, id_ = s_traj[idx], a_traj[idx], r_traj[idx], ns_traj[idx], id_traj[idx]
            for start in range(time_length-H+1):
                if id_[start+H-1] == np.inf: break
                states.append(s_[start:start+H, :].reshape(1, -1))
                actions.append(a_[start:start+H, :].reshape(1, -1))
                rewards.append(r_[start:start+H].reshape(1, -1))
                nstates.append(ns_[start:start+H, :].reshape(1, -1))

        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        nstates = np.concatenate(nstates, axis=0)
        return states, actions, rewards, nstates

    states_list, actions_list, rewards_list, nstates_list = [], [], [], []
    for episode in episodes:
        states, actions, rewards, nstates = construct_single_episode(episode)
        states_list.append(states)
        actions_list.append(actions)
        rewards_list.append(rewards)
        nstates_list.append(nstates)

    states = np.concatenate(states_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    rewards = np.concatenate(rewards_list, axis=0)
    nstates = np.concatenate(nstates_list, axis=0)

    inputs = np.concatenate((states,actions), axis=1)
    if env_type == 'state-transition': outputs = nstates
    elif env_type == 'both': outputs = np.concatenate((rewards, nstates), axis=1)
    else: outputs = rewards

    return inputs, outputs






