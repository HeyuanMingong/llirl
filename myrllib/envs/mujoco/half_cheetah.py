import numpy as np
from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_

class HalfCheetahEnv(HalfCheetahEnv_):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    def viewer_setup(self):
        camera_id = self.model.camera_name2id('track')
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        # Hide the overlay
        self.viewer._hide_overlay = True

    def render(self, mode='human'):
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self._get_viewer().read_pixels(width, height, depth=False)
            return data
        elif mode == 'human':
            self._get_viewer().render()

class HalfCheetahVelEnv(HalfCheetahEnv):
    def __init__(self):
        self._goal_vel = 1.0
        super(HalfCheetahVelEnv, self).__init__()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = 0.05 * np.sum(np.square(action))

        observation = self._get_obs()
        # reward = forward_reward - ctrl_cost
        reward = forward_reward
        done = False
        infos = dict(reward_forward=forward_reward, reward_ctrl=-ctrl_cost, velocity=forward_vel)
        return (observation, reward, done, infos)

    def sample_task(self, num_tasks=1):
        tasks = self.np_random.uniform(0.0, 2.0, size=(num_tasks, 2))
        return tasks

    def reset_task(self, task):
        self._goal_vel = task[0]


