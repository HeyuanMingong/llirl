#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This the code for the paper:
[1] Zhi Wang, Chunlin Chen, and Daoyi Dong, "Lifelong Incremental Reinforcement Learning with 
Online Bayesian Inference", IEEE Transactions on Neural Networks and Learning Systems, 2021.
https://github.com/HeyuanMingong/llinrl.git

This file is for clustering the environment models in a latent space with online
Bayesian inference. The prior distribution on the mixture of environment models
is instantiated as the Chinese Restaurant Process.
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
from myrllib.mixture.env_train import env_update, env_nominal_train
from myrllib.mixture.env_model import EnvModel, construct_env_io
from myrllib.mixture.inference import CRP, compute_likelihood


start_time = time.time()
######################## Arguments ############################################
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, 
        help='number of rollouts/learning episodes in one policy iteration')
parser.add_argument('--model_path', type=str, default='saves/navi_v1',
        help='the folder for saving and loading the pretrained model')
parser.add_argument('--env', type=str, default='Navigation2D-v1')
parser.add_argument('--env_num_layers', type=int, default=2, 
        help='the number of hidden layers of the environment model')
parser.add_argument('--env_hidden_size', type=int, default=200,
        help='the size of the hidden layer of the environment model')
parser.add_argument('--H', type=int, default=4,
        help='using the consecutive H transitions to construct the data')
parser.add_argument('--num_periods', type=int, default=50, 
        help='number of the environment changes')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--et_length', type=int, default=1, 
        help='length of episodic transitions for collecting samples')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
print(args)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
np.set_printoptions(precision=3)
np.random.seed(args.seed); torch.manual_seed(args.seed); random.seed(args.seed)

######################## Small functions ######################################
def softmax_normalize(array, temperature=1.0):
    array = np.array(array).reshape(-1)
    array -= array.mean()
    array_exp = np.exp(array * temperature)
    array_exp /= array_exp.sum()
    return array_exp

### generate legal tasks for the type II and III navigation tasks
def puddle_filtering(start=[0.0, 0.0], radius=[0.1, 0.15, 0.2]):
    tasks_ori = np.random.uniform(-0.5, 0.5, size=(1000, 6))
    tasks = []
    for idx, row in enumerate(tasks_ori):
        centers = row.reshape(-1, 2)
        legal = True
        for jdx, center in enumerate(centers):
            dist = np.sqrt((center[0] - start[0])**2 + (center[1] - start[1])**2)
            if dist <= radius[jdx]:
                legal = False
        if legal:
            tasks.append(row.reshape(1,-1))
        if len(tasks) == args.num_periods:
            break
    return np.concatenate(tasks, axis=0)


######################## Hyperparameters #####################################
### tune these hyperparameters to get different clustering results
SIGMA = 0.25; TAU1 = 1.0; TAU2 = 1.0; EM_STEPS = 1


######################## Main Functions #######################################
""" ENV_TYPE: the type of parameterizing the environment
Select a model to parameterize the environment
reward: using reward function, default
state-transition: using state transition function
both: using the concatenation of both functions
"""
### get the task parameters for each experimental domain
if args.env == 'Navigation2D-v1':
    ### the task is to reaching a dynamic goal by a point agent
    tasks = np.random.uniform(-0.5, 0.5, size=(1000, 2))
    ENV_TYPE = 'reward'

elif args.env == 'Navigation2D-v2':
    tasks = puddle_filtering(start=[-0.5, -0.5])
    ENV_TYPE = 'state-transition'

elif args.env == 'Navigation2D-v3':
    puddles = puddle_filtering(start=[0, 0])
    goals = np.random.uniform(-0.5, 0.5, size=(args.num_periods, 2))
    tasks = np.concatenate((puddles, goals), axis=1)
    ENV_TYPE = 'both'

elif args.env == 'HalfCheetahVel-v1':
    tasks = np.random.uniform(0.0, 2.0, size=(args.num_periods, 1))
    ENV_TYPE = 'reward'

elif args.env == 'HopperVel-v1':
    tasks = np.random.uniform(0.0, 1.0, size=(args.num_periods, 1))
    ENV_TYPE = 'reward'

elif args.env == 'AntVel-v1':
    tasks = np.random.uniform(0.0, 0.5, size=(args.num_periods, 1))
    ENV_TYPE = 'reward'


### build a sampler given an environment
sampler = BatchSampler(args.env, args.batch_size, num_workers=args.batch_size, seed=args.seed) 
env = gym.make(args.env).unwrapped
state_dim = int(np.prod(sampler.envs.observation_space.shape))
action_dim = int(np.prod(sampler.envs.action_space.shape))
print('state dim: %d; action dim: %d'%(state_dim, action_dim))

### generate a uniform policy to collect samples for environment parameterization
action_l, action_h = sampler.envs.action_space.low, sampler.envs.action_space.high
policy_uni = UniformPolicy(state_dim, action_dim, low=action_l, high=action_h)


################ in the initial time period, nominal model ####################
if not os.path.exists(args.model_path): os.makedirs(args.model_path)
task = tasks[0]
print('The nominal task: ', task) 
sampler.reset_task(task)

### generate a batch of episodes to collect samples for environment parameterization
def collect_episodes():
    episodes = []
    for _ in range(args.et_length): 
        episodes.append(sampler.sample(policy_uni, device=device))
    return episodes

### constrict the input-output pairs w.r.t. the environment model accroding to the type
episodes = collect_episodes()
inputs, outputs = construct_env_io(episodes, env_type=ENV_TYPE, H=args.H)

### construct the MLP of the environment model
env_model = EnvModel(inputs.shape[1], outputs.shape[1], 
        hidden_sizes=(args.env_hidden_size,) * args.env_num_layers).to(device=device)

### train the environment model for the nominal task
print('Training the nominal environment model...')
env_model, tloss = env_nominal_train(env_model, inputs, outputs, device=device)

### initialize the Dirichlet  mixture of environment models
env_models = [env_model]

### initilize the CRP prior distribution
crp = CRP(zeta=1.0)


'''
We train a universal model for the initialization of new environment models.
In principle, the new environment models can be initialized in any way, 
e.g., randomly initialization.

'''
epi_list = []
for idx in range(20):
    task = tasks[idx]; sampler.reset_task(task)
    episodes = sampler.sample(policy_uni, device=device)
    epi_list.append(episodes)
inputs, outputs = construct_env_io(epi_list, env_type=ENV_TYPE, H=args.H)
env_model_init = EnvModel(inputs.shape[1], outputs.shape[1], 
        hidden_sizes=(args.env_hidden_size,) * args.env_num_layers).to(device=device)
print('Training the universal model for initialization of new environment models...')
env_model_init, tloss = env_nominal_train(env_model_init, inputs, outputs, device=device)


########## in the following time periods, dynamic environments ################ 
tasks = tasks[:args.num_periods]

### the id of the sequential tasks, the id of the first task is 1
task_ids = np.zeros((tasks.shape[0], 1)); task_ids[0] = 1

for period in range(1, args.num_periods):
    print('\n----------- Time period %d--------------'%period)
    L = crp._L; prior = crp._prior
    
    task = tasks[period]
    print('Taks information', task) 
    sampler.reset_task(task)

    episodes = collect_episodes()
    inputs, outputs = construct_env_io(episodes, env_type=ENV_TYPE, H=args.H)

    ### create a potentially new environment model
    env_model_new = EnvModel(inputs.shape[1], outputs.shape[1], 
            hidden_sizes=(args.env_hidden_size,) * args.env_num_layers).to(device=device)
    env_model_new.load_state_dict(env_model_init.state_dict())

    ### predictive likelihood of the collected samples, including the empty new model
    llls = np.zeros(L+1)
    for idx in range(L):
        llls[idx] = compute_likelihood(env_models[idx], inputs, outputs, sigma=SIGMA)
    llls[-1] = compute_likelihood(env_model_new, inputs, outputs, sigma=SIGMA)
    llls = softmax_normalize(llls, temperature=TAU1)
    print('Predictive likelihood: ', llls)
    prior = softmax_normalize(prior, temperature=TAU2)
    print('Prior distribution: ', prior)

    posterior = llls * prior[:llls.shape[0]]
    posterior = softmax_normalize(posterior)
    print('Posterior over environment models: ', posterior)
    l_post = np.argmax(posterior) + 1
    print('Posterior selection: %d'%l_post)
    if l_post == L+1:
        print('Add a new cluster...')
        env_models.append(env_model_new)

    ### update the CRP prior distribution using the posterior selection
    crp.update(l_post)

    def Estep(env_models, inputs, outputs):
        ### the Expectation-step, compute the posterior of environment-to-cluster assignment
        llls = np.zeros(len(env_models))
        for idx in range(len(env_models)):
            llls[idx] = compute_likelihood(env_models[idx], inputs, outputs, sigma=SIGMA)
        llls = softmax_normalize(llls, temperature=100)
        posterior = llls * prior[:llls.shape[0]]
        posterior = softmax_normalize(posterior, temperature=100)
        return llls, posterior
    
    def Mstep(env_models, inputs, outputs, posterior):
        ### the Maximization-step, update parameters of the mixture model
        for idx in range(len(env_models)):
            env_model = env_models[idx]
            env_model, _ = env_update(env_model, inputs, outputs, 
                    posterior=posterior[idx], device=device)
            env_models[idx] = env_model
        return env_models

    print('********** EM Algorithm **********')
    for _ in range(EM_STEPS):
        llls, posterior = Estep(env_models, inputs, outputs)
        print('Predictive likelihood: ', llls)
        print('Posterior: ', posterior)
        env_models = Mstep(env_models, inputs, outputs, posterior)
    print('********** EM Algorithm **********')

    ### compute the final assignment of the current environment
    updated_llls, posterior = Estep(env_models, inputs, outputs)
    l_star = np.argmax(updated_llls) + 1
    task_ids[period] = l_star
    print('Updated likelihood: ', updated_llls)
    print('Choosing the cluster %d'%l_star)

task_info = np.concatenate((tasks, task_ids), axis=1)
np.save(os.path.join(args.model_path, 'task_info.npy'), task_info)





print('Running time: %.2f mimutes.'%((time.time()-start_time)/60.0))


