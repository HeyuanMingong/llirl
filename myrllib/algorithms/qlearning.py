#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 10:54:03 2018

@author: baobao
"""

import numpy as np
import copy 

class QLearning(object):
    def __init__(self, env, gamma=0.99, lr=0.1, Q=None, R=None, T=None, 
            Q_reuse=None):
        super(QLearning, self).__init__()
        self.gamma = gamma; self.lr = lr 
        self.env = env 
        self.Q_reuse = Q_reuse 
        if Q is None:
            self.Q = np.zeros((env.observation_space.n, env.action_space.n), 
                    dtype=np.float32)
        else:
            self.Q = copy.deepcopy(Q)
        if R is None:
            self.R = np.zeros((env.observation_space.n, env.action_space.n), 
                    dtype=np.float32)
        else:
            self.R = copy.deepcopy(R)
        if T is None:
            self.T = np.zeros((env.observation_space.n, env.action_space.n), 
                    dtype=np.int32)
        else:
            self.T = copy.deepcopy(T)

    def greedy(self, s, Q):
        candi_actions = []; Qmax = -np.inf
        for a in range(self.env.action_space.n):
            if Q[s, a] > Qmax:
                candi_actions = []; candi_actions.append(a)
                Qmax = Q[s, a]
            elif Q[s, a] == Qmax:
                candi_actions.append(a)
        return np.random.choice(candi_actions)

    def softmax(self, s, Q, tau=1.0):
        p = np.exp(Q[s]*tau)/np.sum(np.exp(Q[s]*tau))
        return np.random.choice(self.env.action_space.n, p=p)

    def pi(self, s, epsilon=1.0, tau=1.0, strategy='epsilon'):
        if strategy == 'softmax':
            return self.softmax(s, self.Q, tau=tau)
        else:
            if np.random.random() < epsilon:
                ### random action
                return self.env.action_space.sample()
            else:
                ### epsilon greedy
                return self.greedy(s, self.Q)

    def pi_prq(self, s, epsilon=1.0, reuse_p=0.0, strategy='epsilon', tau=1.0):
        if np.random.random() < reuse_p:
            ### reuse old policy 
            if strategy == 'softmax':
                return self.softmax(s, self.Q_reuse, tau=tau)
            else:
                return self.greedy(s, self.Q_reuse)
        else:
            return self.pi(s, epsilon=epsilon, tau=tau, strategy=strategy)

    def step(self, s, a, r, s_next):
        Qmax_s_next = np.max(self.Q[s_next])
        self.Q[s, a] += self.lr * (r + self.gamma * Qmax_s_next - self.Q[s, a])
        self.R[s, a] = r; self.T[s, a] = s_next 

    def virtual_step(self, s, a, r, s_next):
        self.R[s, a] = r; self.T[s, a] = s_next 

    def drift_detection(self, R_old):
        drift_env = []; drift_env_2d = []
        for i in range(R_old.shape[0]):
            for j in range(R_old.shape[1]):
                if self.R[i, j] != 0 and R_old[i, j] != self.R[i, j]:
                    drift_env.append((i, j))
                    i1,i2 = self.env.to_2d_state(i)
                    drift_env_2d.append((i1,i2,j))
        return drift_env, drift_env_2d 


    def prioritized_sweeping(self, drift_env, m=3, lr=1e-2, max_iters=100):
        def sweep_pair_nei(s, a, m=0):
            i, j = self.env.to_2d_state(s)
            for idx in range(i-m, i+m+1):
                if idx < 0 or idx > self.env.height - 1:
                    continue 
                for jdx in range(j-m, j+m+1):
                    if jdx < 0 or jdx > self.env.width - 1:
                        continue 
                    if self.env._env[idx, jdx]:
                        continue 

                    s_1d = self.env.to_1d_state((idx, jdx))
                    for action in range(self.env.action_space.n):
                        s_next = self.T[s_1d, action]
                        Qmax_s_next = np.max(self.Q[s_next])
                        self.Q[s_1d, action] += lr * (self.R[s_1d, action] + 
                                self.gamma * Qmax_s_next - self.Q[s_1d, action])

        for _ in range(max_iters):
            for (s,a) in drift_env:
                #sweep_pair_nei(s, a, m=0)
                sweep_pair_nei(s, a, m=m)
                




    
