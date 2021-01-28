# LifeLong Incremental Reinforcement Learning (LLIRL)

This repo contains code accompaning the manuscript: [Zhi Wang, Chunlin Chen, and Daoyi Dong, "Lifelong incremental reinforcement learning with online Bayesian inference", *IEEE Transactions on Neural Networks and Learning Systems*, 2021.](https://arxiv.org/abs/2007.14196)
It contains code for running the incremental learning domain tasks, including 2D navigation, Hopper, HalfCheetah, and Ant domains.

### Dependencies
This code requires the following:
* python 3.5+
* pytorch 1.0+
* gym
* MuJoCo license

### Data
* For the 2D navigation domains, data is generated from `myrllib/envs/navigation.py`
* For the Hopper/HalfCheetah/Ant Mujoco domains, the modified Mujoco enviornments are in `myrllib/envs/mujoco/*`

### Usage 
* For example, to run the code in the navi_v1 domain where the dynamic environment is contructed by changing the goal points, just run the bash script `navi_v1_llirl.sh` for LLIRL, and run the bash script `navi_v1_baselines.sh` for the baseline approaches including CA, Robust, Adaptive, and MAML. Also see the usage instructions in the python scripts `env_clustering.py`, `policy_training.py`, and `baselines.py`.
* The task information is saved in `saves/*/task_info.npy`. For visualization of the clustering results, plot the task information using the 'task_clustering' function in `data_process.py`.
* The performance of all methods is recorded in `output/*/*.npy` files. For visualization of performance comparison, plot the learning curves using the 'perforamnce_comparison' function in `data_process.py`.

### Example
* The demo experimental results for the navi_vi domain are showed as
task clustering | performance comparison
------------ | -------------
![task clustering for navi_v1 domain](https://github.com/HeyuanMingong/llirl/blob/master/demo/navi_v1_clustering.png) | ![performance comparison for navi_v1 domain](https://github.com/HeyuanMingong/llirl/blob/master/demo/navi_v1_performance.png)

### Contact 
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/HeyuanMingong/llirl/issues), or email to zhiwang@nju.edu.cn.
 


