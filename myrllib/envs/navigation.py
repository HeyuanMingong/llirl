#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:22:59 2018

@author: qiutian
"""
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class Navigation2DEnvV1(gym.Env):
    def __init__(self):
        super(Navigation2DEnvV1, self).__init__()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)

        self._goal = np.array([0.25,0.25], dtype=np.float32)
        self._state = np.zeros(2, dtype=np.float32)

    def reset_task(self, task):
        self._goal = np.array(task).reshape(-1)

    def sample_task(self, num_tasks=1):
        tasks = np.random.uniform(-0.5, 0.5, size=(num_tasks, 2))
        return tasks

    def reset(self, env=True):
        self._state = np.zeros(2, dtype=np.float32)
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        #assert self.action_space.contains(action)
        self._state = self._state + action

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward_dist = - np.sqrt(x ** 2 + y ** 2)
        reward_ctrl = - 0.01 * np.square(action).sum()
        reward = reward_dist + reward_ctrl

        done = ((np.abs(x) < 0.01) and (np.abs(y) < 0.01))
        return self._state, reward, done, {}


class Navigation2DEnvV2(gym.Env):
    def __init__(self, task={}):
        super(Navigation2DEnvV2, self).__init__()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)

        self._goal = np.array([0.5, 0.5], dtype=np.float32)
        self._state = np.array([-0.5, -0.5], dtype=np.float32)
        self._radius = np.array([0.1, 0.15, 0.2], dtype=np.float32)
        self._center = np.zeros((3,2), dtype=np.float32)

    def sample_task(self, num_tasks=1):
        tasks = np.random.uniform(-0.5, 0.5, size=(num_tasks, 6))
        return tasks

    def reset_task(self, task):
        self._center = np.array(task).reshape(self._center.shape)

    def reset(self):
        self._state = np.array([-0.5,-0.5], dtype=np.float32)
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        # assert self.action_space.contains(action)
        temp_state = self._state + action
        if self.check_puddle(temp_state):
            self._state = temp_state
            reward_puddle = 0
        else:
            reward_puddle = 0
        
        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward_dist = -np.sqrt(x ** 2 + y ** 2)
        reward_ctrl = - 0.01 * np.square(action).sum()
        reward = reward_dist + reward_ctrl + reward_puddle
        done = ((np.abs(x) < 0.01) and (np.abs(y) < 0.01))

        return self._state, reward, done, {}

    def check_puddle(self, pos):
        x = pos[0]; y = pos[1]
        for idx, point in enumerate(self._center):
            dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)
            if dist <= self._radius[idx]:
                return False
        else:
            return True


class Navigation2DEnvV3(gym.Env):
    def __init__(self, task={}):
        super(Navigation2DEnvV3, self).__init__()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)

        self._goal = np.array([0.5,0.5], dtype=np.float32)
        self._state = np.zeros(2, dtype=np.float32)
        self._radius = np.array([0.1, 0.15, 0.2], dtype=np.float32)
        self._center = np.zeros((3,2), dtype=np.float32)

    def sample_task(self, num_tasks=1):
        tasks = np.random.uniform(-0.5, 0.5, size=(num_tasks, 8))
        return tasks

    def reset_task(self, task):
        task = task.reshape(-1, self._center.shape[1])
        self._center = task[:-1]
        self._goal = task[-1]

    def reset(self, env=True):
        self._state = np.zeros(2, dtype=np.float32)
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        # assert self.action_space.contains(action)
        temp_state = self._state + action
        navigable = self.check_puddle(temp_state)
        if navigable:
            self._state = temp_state 
            reward_puddle = 0.0
        else:
            reward_puddle = - 0.1

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward_dist = - np.sqrt(x ** 2 + y ** 2)
        reward_ctrl = - 0.01 * np.square(action).sum()
        reward = reward_dist + reward_puddle + reward_ctrl 
        done = ((np.abs(x) < 0.01) and (np.abs(y) < 0.01))

        return self._state, reward, done, {}

    def check_puddle(self, pos):
        x = pos[0]; y = pos[1]
        for idx, point in enumerate(self._center):
            dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)
            if dist <= self._radius[idx]:
                return False
        else:
            return True



