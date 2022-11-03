import numpy as np
from PIL import Image

from utils import *
from cppenv import env as cpp_env


class ManyUavEnv:

    TARGET_R = 100
    COLLISION_R = 30

    def __init__(self, uav_cnt, seed):
        self._cpp_env = cpp_env.ManyUavEnv(uav_cnt, seed)
        self._viewer = None

    def reset(self):
        self._cpp_env.reset()
        obs = self._cpp_env.getObservations()
        return np.array(obs)

    def step(self, actions):
        self._cpp_env.step(cpp_env.ControlInfo(actions))
        obs = np.array(self._cpp_env.getObservations())
        rewards = np.array(self._cpp_env.getRewards())
        done = self._cpp_env.isDone()
        return obs, rewards, done, {}

    def render(self, mode='human'):
        image = Image.new(mode='RGB', size=(800, 800), color='white')
        transform = np.array([
            [800 / 2000, 0],
            [0, 800 / 2000]
        ])

        target_pos = self._cpp_env.getTarget()
        draw_target_area(image, (target_pos.x, target_pos.y), transform, ManyUavEnv.TARGET_R)

        uavs = self._cpp_env.getUavs()
        for u in uavs:
            draw_uav(image, [u.x, u.y], transform)

        obs = self._cpp_env.getObstacles()
        collision = self._cpp_env.getCollision()
        for i, o in enumerate(obs):
            draw_obstacle(image, [o.x, o.y], transform, collision[i], ManyUavEnv.COLLISION_R)

        image = np.asarray(image)
        if mode == 'rgb_array':
            return image
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self._viewer is None:
                self._viewer = rendering.SimpleImageViewer()
            self._viewer.imshow(image)
            return self._viewer.isopen

    def close(self):
        if self._viewer:
            self._viewer.close()


class EnvWrapper:

    def __init__(self, n_agents):
        self._env = ManyUavEnv(n_agents, 123)
        self._n_agents = n_agents
        self._global_center = False
        
    def set_global_center(self, value):
        self._global_center = value

    def reset(self):
        s = self._env.reset()

        result = {}
        for i in range(self._n_agents):
            result[f'uav_{i}'] = np.copy(s[i])
            if not self._global_center:
                result[f'uav_{i}'][-1] = 0.0
                result[f'uav_{i}'][-2] = 0.0
        return result

    def step(self, actions):
        act = []
        for i in range(self._n_agents):
            act.append(actions[f'uav_{i}'] * np.array([np.pi / 4, 1.0]))
        s, r, done, info = self._env.step(act)

        result_s = {}
        result_r = {}
        result_d = {}
        for i in range(self._n_agents):
            result_s[f'uav_{i}'] = np.copy(s[i])
            result_r[f'uav_{i}'] = r[i]
            result_d[f'uav_{i}'] = done
            
            if not self._global_center:
                result_s[f'uav_{i}'][-1] = 0.0
                result_s[f'uav_{i}'][-2] = 0.0
        result_d['__all__'] = all(result_d.values())
        return result_s, result_r, result_d, {}

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def getCollisionWithUAV(self):
        return self._env._cpp_env.getCollisionWithUav()

    def getSuccCnt(self):
        return self._env._cpp_env.getSuccCnt()

    def getRadius(self):
        return self._env._cpp_env.getRadius()

    def getStatus(self):
        return self._env._cpp_env.getStatus()
    
    def get_ta(self):
        return self._env._cpp_env.getTa()

    def get_die(self):
        return self._env._cpp_env.getDie()

    def get_dtc_avg(self):
        return self._env._cpp_env.getRadius()



def main():
    n_agents = 10
    env = ManyUavEnv(n_agents, 123)
    env.reset()
    done = False
    while not done:
        s, r, done, info = env.step(np.random.uniform(-1, 1, (n_agents, 2)))
        env.render()
    env.close()


if __name__ == '__main__':
    main()
