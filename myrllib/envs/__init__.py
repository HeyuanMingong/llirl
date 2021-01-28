#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:52:14 2018

@author: qiutian
"""
from gym.envs.registration import register

# 2D Navigation
# ----------------------------------------

register(
    'Navigation2D-v1',
    entry_point='myrllib.envs.navigation:Navigation2DEnvV1',
    max_episode_steps=100
)

register(
        'Navigation2D-v2',
        entry_point='myrllib.envs.navigation:Navigation2DEnvV2',
        max_episode_steps=100 
        )

register(
        'Navigation2D-v3',
        entry_point='myrllib.envs.navigation:Navigation2DEnvV3',
        max_episode_steps=100 
        )

register(
        'HalfCheetahVel-v1',
        entry_point = 'myrllib.envs.mujoco.half_cheetah:HalfCheetahVelEnv',
        #entry_point = 'myrllib.envs.utils:mujoco_wrapper',
        #kwargs = {'entry_point': 'myrllib.envs.mujoco.half_cheetah:HalfCheetahVelEnv'},
        max_episode_steps = 100 
        )

register(
        'ReacherDyna-v1',
        entry_point = 'myrllib.envs.utils:mujoco_wrapper',
        kwargs = {'entry_point': 'myrllib.envs.mujoco.reacher:ReacherDynaEnvV1'},
        max_episode_steps = 100 
        )

register(
        'ReacherDyna-v2',
        entry_point = 'myrllib.envs.utils:mujoco_wrapper',
        kwargs = {'entry_point': 'myrllib.envs.mujoco.reacher:ReacherDynaEnvV2'},
        max_episode_steps = 100 
        )

register(
        'ReacherDyna-v3',
        entry_point = 'myrllib.envs.utils:mujoco_wrapper',
        kwargs = {'entry_point': 'myrllib.envs.mujoco.reacher:ReacherDynaEnvV3'},
        max_episode_steps = 100 
        )

register(
        'HopperVel-v1',
        entry_point = 'myrllib.envs.mujoco.hopper:HopperVelEnv',
        #entry_point = 'myrllib.envs.utils:mujoco_wrapper',
        #kwargs = {'entry_point': 'myrllib.envs.mujoco.hopper:HopperVelEnv'},
        max_episode_steps = 100 
        )

register(
        'AntVel-v1',
        entry_point = 'myrllib.envs.mujoco.ant:AntVelEnv',
        #entry_point = 'myrllib.envs.utils:mujoco_wrapper',
        #kwargs = {'entry_point': 'myrllib.envs.mujoco.ant:AntVelEnv'},
        max_episode_steps = 100 
        )

register(
        'SwimmerVel-v1',
        entry_point = 'myrllib.envs.mujoco.swimmer:SwimmerVelEnv',
        #entry_point = 'myrllib.envs.utils:mujoco_wrapper',
        #kwargs = {'entry_point': 'myrllib.envs.mujoco.swimmer:SwimmerVelEnv'},
        max_episode_steps = 100 
        )

register(
        'HumanoidVel-v1',
        entry_point = 'myrllib.envs.mujoco.humanoid:HumanoidVelEnv',
        max_episode_steps = 100 
        )

