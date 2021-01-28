#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This the code for the paper:
[1] Zhi Wang, Chunlin Chen, and Daoyi Dong, "Lifelong Incremental Reinforcement Learning with 
Online Bayesian Inference", IEEE Trasactions on Neural Networks and Learning Systems, 2021.
https://github.com/HeyuanMingong/llinrl.git

This file is the implementation of the policy learning part of the proposed LLIRL algorithm
"""

### common lib
import sys
import gym
import numpy as np
import argparse 
import torch
from tqdm import tqdm
import os
import time 
from torch.optim import Adam, SGD 
import scipy.io as sio
import copy 
from collections import OrderedDict
import torch.nn.functional as F
import pickle
import shutil
import random

### personal lib
from myrllib.episodes.episode import BatchEpisodes 
from myrllib.samplers.sampler import BatchSampler 
from myrllib.policies import NormalMLPPolicy, UniformPolicy  
from myrllib.baselines.baseline import LinearFeatureBaseline
from myrllib.algorithms.reinforce import REINFORCE 
from myrllib.algorithms.trpo import TRPO 
from myrllib.algorithms.ppo import PPO
from myrllib.mixture.inference import CRP


start_time = time.time()
######################## Arguments ############################################
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, 
        help='number of rollouts/learning episodes in one policy iteration')
parser.add_argument('--hidden_size', type=int, default=200,
        help='hidden size of the policy network')
parser.add_argument('--num_layers', type=int, default=2,
        help='number of hidden layers of the policy network')
parser.add_argument('--num_iter', type=int, default=100,
        help='number of policy iterations')
parser.add_argument('--lr', type=float, default=1e-3,
        help='learning rate, if REINFORCE algorithm is used')
parser.add_argument('--output', type=str, default='output/navi_v1',
        help='output folder for saving the experimental results')
parser.add_argument('--model_path', type=str, default='saves/navi_v1',
        help='the folder for saving and loading the pretrained model')
parser.add_argument('--algorithm', type=str, default='reinforce',
        help='reinforce or trpo, the base algorithm for policy gradient')
parser.add_argument('--opt', type=str, default='sgd',
        help='sgd or adam, if using the reinforce algorithm')
parser.add_argument('--env', type=str, default='Navigation2D-v1')
parser.add_argument('--baseline', type=str, default=None,
        help='linear or None, baseline for policy gradient step')
parser.add_argument('--num_periods', type=int, default=50)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
print(args)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed); torch.manual_seed(args.seed); random.seed(args.seed)
np.set_printoptions(precision=3)

######################## Small functions ######################################
### build a learner given a policy network
def generate_learner(policy):
    if args.algorithm == 'trpo':
        learner = TRPO(policy, baseline=args.baseline, device=device)
    elif args.algorithm == 'ppo':
        learner = PPO(policy, baseline=args.baseline, lr=args.lr, opt=args.opt, device=device)
    else:
        learner = REINFORCE(policy, baseline=args.baseline, lr=args.lr, opt=args.opt, device=device)
    return learner 

### train a policy using a learner
def inner_train(policy, learner):
    rews = np.zeros(args.num_iter)
    for idx in tqdm(range(args.num_iter)):
        episodes = sampler.sample(policy, device=device)
        rews[idx] = episodes.evaluate()
        learner.step(episodes, clip=True)
    return rews


######################## Main Functions #######################################
### build a sampler given an environment
env = gym.make(args.env).unwrapped
sampler = BatchSampler(args.env, args.batch_size, num_workers=args.batch_size, seed=args.seed) 
state_dim = int(np.prod(sampler.envs.observation_space.shape))
action_dim = int(np.prod(sampler.envs.action_space.shape))
print('state dim: %d; action dim: %d'%(state_dim,action_dim))

### get the task ids that are computed using env_clustering.py
task_info = np.load(os.path.join(args.model_path, 'task_info.npy'))
tasks = task_info[:, :-1]
task_ids = task_info[:, -1]

print('====== Lifelong Incremental Reinforcement Learning (LLIRL) =======')
if not os.path.exists(args.output): os.makedirs(args.output)

### at the initial time period, nominal model
print('The nominal task:', tasks[0]) 
sampler.reset_task(tasks[0])

### generate the nominal policy model at the first time period
policy_init = NormalMLPPolicy(state_dim, action_dim, 
        hidden_sizes=(args.hidden_size,) * args.num_layers)
learner_init = generate_learner(policy_init)

### training the nominal policy model 
print('Train the nominal model...')
rews_init = inner_train(policy_init, learner_init)

### initialize the Dirichlet mixture model
policies = [policy_init]; learners = [learner_init]
num_policies = 1

### record the performance 
rews_llirl = np.zeros((args.num_periods, args.num_iter))
rews_llirl[0] = rews_init

### in the following time periods, dynamic environments
for period in range(1, args.num_periods):
    print('\n----------- Time period %d------'%(period+1))
    task = tasks[period]
    print('The task information:', task) 
    sampler.reset_task(task)

    task_id = int(task_ids[period])
    if task_id == num_policies + 1:
        print('Generate a new policy...')
        policy = NormalMLPPolicy(state_dim, action_dim, 
                hidden_sizes=(args.hidden_size,) * args.num_layers)

        ### randomly choose an existing policy model to initialize the new one
        index = np.random.choice(num_policies)
        policy.load_state_dict(policies[index].state_dict())

        ### train the policy in the current environment
        learner = generate_learner(policy)
        rews = inner_train(policy, learner)
        policies.append(policy)
        learners.append(learner)
        num_policies += 1

    elif task_id <= num_policies:
        print('Choosing the policy %d'%task_id)
        policy = policies[task_id-1]; learner = learners[task_id-1]

        rews = inner_train(policy, learner)
        policies[task_id-1] = policy; learners[task_id-1] = learner

    else:
        assert('Error on task id!!!')
    rews_llirl[period] = rews

    print('Average return: %.2f'%rews.mean())
    np.save(os.path.join(args.output, 'rews_llirl.npy'), rews_llirl)






print('Running time: %.2f min'%((time.time()-start_time)/60.0))


