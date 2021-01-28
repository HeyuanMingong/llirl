import numpy as np
from gym import utils
#from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco import HopperEnv 

class HopperVelEnv(HopperEnv):
    def __init__(self):
        self._goal_vel = 0.5
        super(HopperVelEnv, self).__init__()

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        #alive_bonus = 1.0
        #reward = (posafter - posbefore) / self.dt
        #reward += alive_bonus
        #reward -= 1e-3 * np.square(a).sum()
        
        alive_bonus = 1.0
        forward_vel = (posafter - posbefore) / self.dt 
        forward_reward = - 4.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = - 0.01 * np.square(a).sum()
        #reward = forward_reward + ctrl_cost + alive_bonus
        reward = forward_reward + alive_bonus

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        
        return ob, reward, done, dict(velocity=forward_vel)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ]).astype(np.float32).flatten()

    def reset_task(self, task):
        self._goal_vel = task[0]

    def sample_task(self, num_tasks=1):
        tasks = np.random.uniform(0, 1.0, size=(num_tasks,2))
        return tasks


