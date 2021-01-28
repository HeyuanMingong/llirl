#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:51:34 2018

@author: qiutian
"""
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def st3d(Z, rstride=1, cstride=1, xlabel='Time', ylabel='Space', zlabel='Output', 
         xticks=None, yticks=None, index=None, sub_fig=None):
    # x - row - rstride
    # y - column - cstride
    Z = np.array(Z)
    len_y, len_x = Z.shape
    X = np.arange(0, len_x, 1)
    Y = np.arange(0, len_y, 1)
    X, Y = np.meshgrid(X, Y)
    
    if sub_fig is None:
        fig = plt.figure() if index is None else plt.figure(index)
        ax = Axes3D(fig)
    else:
        ax = sub_fig
    
    ax.plot_surface(X, Y, Z, rstride=rstride, cstride=cstride, 
                    cmap=plt.get_cmap('rainbow'), alpha=1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
    
    if yticks is not None:
        plt.yticks(np.arange(0,len_y+1,len_y//(len(yticks)-1)), yticks)
        ax.set_ylim((0, len_y-1))
    if xticks is not None:
        plt.xticks(np.arange(0,len_x+1,len_x//(len(xticks)-1)), xticks)
        ax.set_xlim((0, len_x-1))
    plt.show()
    
def multi_st3d(fig_list, index=None):
    if index is None:
        fig = plt.figure(figsize=(10,4))
    else:
        plt.close(index)
        fig = plt.figure(index,figsize=(10,4))
        
    num_fig = len(fig_list)
    for idx in range(num_fig):
        ax = fig.add_subplot(1, num_fig, idx+1, projection='3d')
        st3d(fig_list[idx], sub_fig=ax)
    plt.show()
    

def simple_plot(vector_list, index=None, title=None, xlabel=None, ylabel=None, 
                legend=None, logy=False):
    if index is None:
        plt.figure()
    else:
        plt.figure(index)   
    x = range(len(vector_list[0]))
    for vector in vector_list:
        if logy:
            plt.semilogy(x, vector)
        else:
            plt.plot(x, vector)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(legend)
    plt.show()
    

def multi_plot(fig_list, index=None, title=None, xlabel=None, ylabel=None, legend=None):
    if index is None:
        plt.figure(figsize=(12,3))
    else:
        plt.close(index)
        plt.figure(index,figsize=(12,3))
        
    num_fig = len(fig_list)
    for idx in range(num_fig):
        plt.subplot(1, num_fig, idx+1)
        vector_list = fig_list[idx]
        x = range(len(vector_list[0]))
        for vi, vector in enumerate(vector_list):
            plt.plot(x, vector)
            
        if title is not None:
            plt.title(title[idx])
        if xlabel is not None:
            plt.xlabel(xlabel[idx])
        if ylabel is not None:
            plt.ylabel(ylabel[idx])
        if legend is not None:
            plt.legend(legend[idx])
            
    plt.show()  

    
    
    
    
    
    
    
    
    
    

    