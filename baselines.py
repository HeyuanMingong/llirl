#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This the code for the paper:
[1] Zhi Wang, Chunlin Chen, and Daoyi Dong, "Lifelong Incremental Reinforcement Learning with 
Online Bayesian Inference", IEEE Transactions on Neural Networks and Learning Systems, 2021.
https://github.com/HeyuanMingong/llinrl.git

This file is the implementation of the baseline approaches investigated in the paper, including:
    CA: Continuous Adapt a single policy model during lifelong learning
    Robust: leverages domain randomization to train a robust policy for all environments
    Adaptive: leverages domain randomization to train an LSTM policy for all environments
    MAML: trains a meta-policy using Model-Agnostic Meta-Learning
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
from myrllib.policies import NormalMLPPolicy, NormalLSTMPolicy  
from myrllib.baselines.baseline import LinearFeatureBaseline
from myrllib.algorithms.reinforce import REINFORCE 
from myrllib.algorithms.trpo import TRPO 
from myrllib.algorithms.ppo import PPO
from myrllib.algorithms.maml import MetaLearner
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
parser.add_argument('--robust', action='store_true', default=False)
parser.add_argument('--adaptive', action='store_true', default=False)
parser.add_argument('--maml', action='store_true', default=False)
parser.add_argument('--ca', action='store_true', default=False)
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

######################## Main Functions #######################################
### build a sampler given an environment
env = gym.make(args.env).unwrapped
state_dim = int(np.prod(env.observation_space.shape))
action_dim = int(np.prod(env.action_space.shape))
print('state dim: %d; action dim: %d'%(state_dim,action_dim))

if not os.path.exists(args.model_path): os.makedirs(args.model_path)
if not os.path.exists(args.output): os.makedirs(args.output)

### get the task information
tasks = np.load(os.path.join(args.model_path, 'task_info.npy'))[:, :-1]

###############################################################################
if args.robust:
    print('\n======== Robust baseline =========')
    policy_robust = NormalMLPPolicy(state_dim, action_dim, 
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    learner_robust = generate_learner(policy_robust)
    sampler = BatchSampler(args.env, args.batch_size, num_workers=args.batch_size, seed=args.seed) 

    print('Training the robust policy...')
    rews_pre_robust = np.zeros(args.num_iter)
    for idx in tqdm(range(args.num_iter)):
        _ = sampler.domain_randomization()
        episodes = sampler.sample(policy_robust, device=device)
        rews_pre_robust[idx] = episodes.evaluate()
        learner_robust.step(episodes, clip=True)


    print('Test in execution time...')
    policy = NormalMLPPolicy(state_dim, action_dim, 
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    learner = generate_learner(policy)

    rews_robust = np.zeros((args.num_periods, args.num_iter))
    for period in range(args.num_periods):
        print('\nTask %d'%(period+1), tasks[period])
        policy.load_state_dict(policy_robust.state_dict())
        sampler.reset_task(tasks[period])

        for idx in tqdm(range(args.num_iter)):
            episodes = sampler.sample(policy, device=device)
            rews_robust[period, idx] = episodes.evaluate()
            learner.step(episodes, clip=True)

        print('Average return: %.2f'%np.mean(rews_robust[period]))
        name = os.path.join(args.output, 'rews_robust.npy')
        np.save(name, rews_robust)


###############################################################################
if args.adaptive:
    print('\n======== Adaptive baseline =========')
    policy_adapt = NormalLSTMPolicy(state_dim, action_dim, bs=args.batch_size, device=device)
    learner_adapt = generate_learner(policy_adapt)
    sampler = BatchSampler(args.env, args.batch_size, num_workers=args.batch_size, seed=args.seed) 

    print('Training the adaptive policy...')
    rews_pre_adapt = np.zeros(args.num_iter)
    for idx in tqdm(range(args.num_iter)):
        _ = sampler.domain_randomization()
        episodes = sampler.sample(policy_adapt, recurrent=True, device=device)
        rews_pre_adapt[idx] = episodes.evaluate()
        learner_adapt.step(episodes, recurrent=True, clip=True)


    print('Test in execution time...')
    policy = NormalLSTMPolicy(state_dim, action_dim, bs=args.batch_size, device=device)
    learner = generate_learner(policy)
    rews_adapt = np.zeros((args.num_periods, args.num_iter))

    for period in range(args.num_periods):
        print('\nTask %d'%(period+1), tasks[period])
        policy.load_state_dict(policy_adapt.state_dict())
        sampler.reset_task(tasks[period])

        for idx in tqdm(range(args.num_iter)):
            episodes = sampler.sample(policy, recurrent=True, device=device)
            rews_adapt[period, idx] = episodes.evaluate()
            learner.step(episodes, recurrent=True, clip=True)

        print('Average return: %.2f'%np.mean(rews_adapt[period]))
        np.save(os.path.join(args.output, 'rews_adapt.npy'), rews_adapt)


################################################################################
if args.maml:
    print('\n======== MAML baseline =========')
    policy_maml = NormalMLPPolicy(state_dim, action_dim,  
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    sampler = BatchSampler(args.env, args.batch_size, num_workers=args.batch_size, seed=args.seed) 

    print('Training the maml policy...')
    learner_maml = MetaLearner(sampler, policy_maml, fast_lr=args.lr, opt=args.opt,
            lr=args.lr, baseline=args.baseline, device=device)

    rews_pre_maml = np.zeros((2, args.num_iter))
    for idx in tqdm(range(args.num_iter)):
        random_tasks = sampler.sample_task(num_tasks=args.batch_size)
        episodes = learner_maml.sample(random_tasks)
        learner_maml.step_ppo(episodes, clip=True)

        for (train_episodes, valid_episodes) in episodes:
            rews_pre_maml[0, idx] += train_episodes.evaluate()
            rews_pre_maml[1, idx] += valid_episodes.evaluate()
        rews_pre_maml[:, idx] /= len(episodes)
        print('iter %d'%idx, rews_pre_maml[:, idx])


    print('Test in execution time...')
    policy = NormalMLPPolicy(state_dim, action_dim,  
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    learner = generate_learner(policy)
    
    rews_maml = np.zeros((args.num_periods, args.num_iter))
    for period in range(args.num_periods):
        print('\nTask %d'%(period+1), tasks[period])

        policy.load_state_dict(policy_maml.state_dict())
        sampler.reset_task(tasks[period])

        for idx in tqdm(range(args.num_iter)):
            episodes = sampler.sample(policy, device=device)
            rews_maml[period, idx] = episodes.evaluate()
            learner.step(episodes, clip=True)

        print('Average return: %.2f'%np.mean(rews_maml[period]))
        np.save(os.path.join(args.output, 'rews_maml.npy'), rews_maml)


################################################################################
if args.ca:
    print('\n======== CA baseline =========')
    policy = NormalMLPPolicy(state_dim, action_dim, 
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    learner = generate_learner(policy)
    sampler = BatchSampler(args.env, args.batch_size, num_workers=args.batch_size, seed=args.seed)

    rews_ca = np.zeros((args.num_periods, args.num_iter))
    for period in range(args.num_periods):
        print('\nTask %d'%(period+1), tasks[period])
        sampler.reset_task(tasks[period])

        for idx in tqdm(range(args.num_iter)):
            episodes = sampler.sample(policy, device=device)
            rews_ca[period, idx] = episodes.evaluate()
            learner.step(episodes, clip=True)

        print('Average return: %.2f'%np.mean(rews_ca[period]))
        np.save(os.path.join(args.output, 'rews_ca.npy'), rews_ca)











print('Running time: %.2f minutes'%((time.time()-start_time)/60.0))


