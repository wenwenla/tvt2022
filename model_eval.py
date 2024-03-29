import itertools
import os
import pickle

import torch
import numpy as np
from uav_2d_ma_fast import EnvWrapper

OBS_DIM = 50


class Actor(torch.nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc0 = torch.nn.Linear(obs_dim, 128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        a = torch.tanh(self.fc2(x))
        return a


class Policy:

    def __init__(self):
        self._actor = Actor(OBS_DIM, 2)

    def choose_action(self, obs):
        o = torch.tensor(obs).float()
        with torch.no_grad():
            a = self._actor(o)
            a = a.detach().cpu().numpy()
        return a

    def load(self, fn):
        data = torch.load(fn)
        self._actor.load_state_dict(data[0])


def real_done(done):
    for v in done.values():
        if not v:
            return False
    return True


data_collection = []


def is_die_state(st):
    return abs(st[-4] - 1) < 1e-9


class EvalAgent:

    def __init__(self, n_agents):
        self._agent = Policy()
        self._n_agents = n_agents
        self._env = EnvWrapper(n_agents)

    def eval_one_ep(self):
        enum_seq = [f'uav_{i}' for i in range(self._n_agents)]
        states = self._env.reset()
        done = {n: False for n in enum_seq}
        total_rew = {n: 0 for n in enum_seq}

        dtc_seq = []
        time_steps = 0
        while not real_done(done):
            actions = {}

            for s in states.values():
                if not is_die_state(s):
                    data_collection.append(s)

            for seq in enum_seq:
                action = self._agent.choose_action(states[seq])
                actions[seq] = action

            time_steps += 1
            next_states, rewards, done, info = self._env.step(actions)
            time_steps += 1
            # self._env.render()
            for seq in enum_seq:
                total_rew[seq] += rewards[seq]
            states = next_states
            dtc_seq.append(self._env.get_dtc_avg())
        ta = self._env.get_ta()
        die = self._env.get_die()
        # print('=' * 10)
        # print(f'CNT: {self._env.get_ta()}')
        # print(f'Die: {self._env.get_die()}')
        # print(f'Dtc: {dtc_seq}')
        # print(f'R: {total_rew}')
        # print(f'RS: {self._env.get_reward_split()}')
        return total_rew, ta, die, dtc_seq

    def load(self, fn):
        self._agent.load(fn)

    def setGlobal(self, value):
        self._env.set_global_center(value)


def get_models(folder):
    files = os.path.join(folder, 'models')
    check_points = []

    for rt, fd, pth in os.walk(files):
        for p in pth:
            check_points.append(os.path.join(rt, p))
        break
    return check_points


def get_bst_path(n_agents, folder):
    agent = EvalAgent(n_agents)
    models = get_models(folder)
    bst_r = -1e9
    result = None
    for m in models:
        agent.load(m)
        r = agent.eval_n(2)
        if r > bst_r:
            bst_r = r
            result = m
    print(f'Bst: {bst_r} Pth: {result}')
    return result


def evaluate():
    agent = EvalAgent(16)
    agent.setGlobal(True)
    agent.load('./0226/gy2/td3_16/models/38201.pkl')

    ta, die, dtc = [], [], []
    for i in range(100):
        print(f'--{i}--')
        _r, _ta, _die, _dtc = agent.eval_one_ep()
        ta.append(_ta)
        die.append(_die)
        dtc.append(_dtc)
        print(_r)
    print(f'MeanTA: {np.mean(ta)}±{np.std(ta)}')
    print(f'MeanDie: {np.mean(die)}±{np.std(die)}')
    with open('new_version/l32_data.pkl', 'wb') as f:
        pickle.dump((ta, die, dtc), f)


def collect_data():
    agent = EvalAgent(32)
    agent.setGlobal(True)
    agent.load('./new_version/27601_g_32.pkl')
    for i in range(1000):
        agent.eval_one_ep()
        print(f'EP: {i} DL: {len(data_collection)}')

        if len(data_collection) > 500000:
            break

    with open('new_version/sl_samples_32.pkl', 'wb') as f:
        pickle.dump(data_collection, f)


if __name__ == '__main__':
    evaluate()
    # collect_data()