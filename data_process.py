#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from myrllib.utils.myplot import simple_plot
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.manifold import TSNE


###############################################################################
### DOMAIN can be 'navi_v1', 'navi_v2', 'navi_v3', 'hopper', 'cheetah', 'ant'
np.random.seed(950418)
DOMAIN = 'navi_v1'
p_output = 'output/%s'%DOMAIN
p_model = 'saves/%s'%DOMAIN


###############################################################################
### visualize the clustering results in domains: navi_v1, hopper, cheetah, ant
def task_clustering():
    assert DOMAIN in ['navi_v1', 'hopper', 'cheetah', 'ant']
    
    info = np.load(os.path.join(p_model, 'task_info.npy'))
    tasks = info[:, :-1]
    
    if DOMAIN in ['hopper', 'cheetah', 'ant']:
        tasks = np.concatenate([tasks, np.random.uniform(0.4, 0.6, size=tasks.shape)], axis=1)
        plt.figure(figsize=(6,2),dpi=200)
    else:
        plt.figure(figsize=(3,3),dpi=200)
    info = np.concatenate((tasks, info[:, -1].reshape(-1,1)), axis=1)
    
    clusters = max(info[:, -1])
    points = np.zeros(int(clusters))
    print('num of clusters:', int(clusters))
    
    for idx in range(info.shape[0]):
        item = info[idx][-3:]        
        points[int(item[-1])-1] += 1
        
        S = 50
        if item[-1] == 1:
            plt.scatter(item[0], item[1], marker='x', color='k', s=S)
        elif item[-1] == 2:
            plt.scatter(item[0], item[1], marker='+', color='r', s=S)
        elif item[-1] == 3:
            plt.scatter(item[0], item[1], marker='s', color='b', s=S)
        elif item[-1] == 4:
            plt.scatter(item[0], item[1], marker='^', color='g', s=S)
        elif item[-1] == 5:
            plt.scatter(item[0], item[1], marker='*', color='m', s=S)
        elif item[-1] == 6:
            plt.scatter(item[0], item[1], marker='o', color='c', s=S)
    
    if DOMAIN in ['navi_v1']:
        plt.grid(axis='x', ls='--')
        plt.grid(axis='y', ls='--')
        tick_real = [-0.5, -0.25, 0, 0.25, 0.5]
        tick_show = [-0.5, '', '', '', 0.5]
        tick_show_y = ['', '', '', '', 0.5]
        plt.yticks(tick_real, tick_show_y, fontsize=12)
        plt.xticks(tick_real, tick_show, fontsize=12)
        plt.axis([-0.5, 0.5, -0.5, 0.5])
        
    elif DOMAIN in ['hopper', 'cheetah', 'ant']:
        plt.xlabel('Goal velocity', fontsize=16)
        plt.grid(axis='x', ls='--')
        plt.yticks([])
    
        if DOMAIN == 'hopper':
            plt.xticks([0, 0.25, 0.5, 0.75, 1], fontsize=12)
            plt.axis([0, 1, 0.3, 0.7])
    
        elif DOMAIN == 'cheetah':
            plt.xticks([0, 0.5, 1, 1.5, 2], fontsize=12)
            plt.axis([0, 2, 0.3, 0.7])
    
        elif DOMAIN == 'ant':
            plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=12)
            plt.axis([0, 0.5, 0.3, 0.7])
    
    print(points)
    
    return info
info = task_clustering()



###############################################################################
def perforamnce_comparison():
    rews_llirl = np.load(os.path.join(p_output, 'rews_llirl.npy'))
    rews_ca = np.load(os.path.join(p_output, 'rews_ca.npy'))
    rews_robust = np.load(os.path.join(p_output, 'rews_robust.npy'))
    rews_adapt = np.load(os.path.join(p_output, 'rews_adapt.npy'))
    rews_maml = np.load(os.path.join(p_output, 'rews_maml.npy'))
    
    if DOMAIN in ['navi_v1', 'navi_v2', 'navi_v3']:
        cutoff = 100
    elif DOMAIN in ['hopper']:
        cutoff = 200
    elif DOMAIN in ['cheetah', 'ant']:
        cutoff = 500

    rews_llirl = rews_llirl[:, :cutoff].mean(axis=0) 
    rews_ca = rews_ca[:, :cutoff].mean(axis=0)
    rews_robust = rews_robust[:, :cutoff].mean(axis=0)
    rews_adapt = rews_adapt[:, :cutoff].mean(axis=0)
    rews_maml = rews_maml[:, :cutoff].mean(axis=0)
    
    print('Return CA:', rews_ca.mean())
    print('Return Robust:', rews_robust.mean())
    print('Return Adaptive:', rews_adapt.mean())
    print('Return Maml:', rews_maml.mean())
    print('Return LIRL:', rews_llirl.mean())
    
    xx = np.arange(rews_llirl.shape[0])
    plt.figure()
    plt.plot(xx, rews_ca, xx, rews_robust, xx, rews_adapt, xx, rews_maml, xx, rews_llirl)
    plt.legend(['CA', 'Robust', 'Adaptive', 'MAML', 'LLIRL'])
    plt.xlabel('Learning episode', fontsize=16)
    plt.ylabel('Return', fontsize=16)
    
    return (rews_ca, rews_robust, rews_adapt, rews_maml, rews_llirl)
data = perforamnce_comparison()
    





























































