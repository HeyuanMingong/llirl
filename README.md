# Incremental Reinforcement Learning in Continuous Spaces

This repo contains code accompaning the paper: [Zhi Wang, Han-Xiong Li, and Chunlin Chen, "Incremental reinforcement learning in continuous spaces via policy relaxation and importance weighting for dynamic environments", submitted to IEEE Transactions on Neural Networks and Learning Systems, 2018.](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385)
It contains code for running the incremental learning domain tasks, including 2D navigation, Reacher, Swimmer, Hopper, and HalfCheetah domains.

### Dependencies
This code requires the following:
* python 3.\*
* pytorch 0.4+
* gym
* MuJoCo license

### Data
* For the 2D navigation domain, data is generated from `myrllib/envs/navigation.py`
* For the Reacher/Swimmer/Hopper/HalfCheetah Mujoco domains, the modified Mujoco enviornments are in `myrllib/envs/mujoco/*`

### Usage 
* For example, to run the code in the 2D Navigation domain with type I dynamic environment, just run the bash script `./demo_navigation_v1.sh`, also see the usage instructions in the script and `main.py`
* When getting the results in .mat files, plot the results using `demo_data_process.py`. For example, the results for `./demo_navigation_v1.sh` is as follow:
![experimental results for navigation domain](https://github.com/HeyuanMingong/irl_cs/blob/master/exp/demo_navigation_v1.png)

Also, the results for other demo scripts are shown in `exp/*`

### Contact 
For safety reasons, the source code is coming soon.

To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/HeyuanMingong/irl_cs/issues), or email to njuwangzhi@gmail.com.
 


