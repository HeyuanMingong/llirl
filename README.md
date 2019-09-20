# LifeLong Incremental Reinforcement Learning (LLIRL)

This repo contains code accompaning the paper: [Zhi Wang, Chunlin Chen, and Daoyi Dong, "Lifelong Incremental reinforcement learning with online Bayesian inference", submitted.]()
It contains code for running the incremental learning domain tasks, including 2D navigation, Hopper, HalfCheetah, and Ant domains.

### Dependencies
This code requires the following:
* python 3.\*
* pytorch 0.4+
* gym
* MuJoCo license

### Data
* For the 2D navigation domain, data is generated from `myrllib/envs/navigation.py`
* For the Hopper/HalfCheetah/Ant Mujoco domains, the modified Mujoco enviornments are in `myrllib/envs/mujoco/*`

### Usage 
* For example, to run the code in the 2D Navigation domain with type I dynamic environment, just run the bash script `navigation_v1.sh`, also see the usage instructions in the python scripts, `env_clustering.py`, and `policy_training.py`.
* When getting the results in .mat files, plot the results using `data_process.py`. For example, the results for `navigation_v3.sh` and `ant.sh` are as follow:

navigation_v3 | ant
------------ | -------------
![experimental results for navigation_v3 domain](https://github.com/HeyuanMingong/llirl/blob/master/exp/navigation_v3.png) | ![experimental results for ant domain](https://github.com/HeyuanMingong/llirl/blob/master/exp/ant.png)

Also, the results for other demo scripts are shown in `exp/*`

### Contact 
For safety reasons, the source code is coming soon.

To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/HeyuanMingong/llirl/issues), or email to njuwangzhi@gmail.com.
 


