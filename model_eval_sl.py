import itertools
import os
import pickle

import torch
import numpy as np
from uav_2d_ma_fast import EnvWrapper

from sl_train import PredictModel


N_Agents = 32
SL_MODEL_PATH = './new_version/sl_models_32/model_199.pkl'
POLICY_PATH = './new_version/27601_g_32.pkl'
OUTPUT_PATH = './new_version/gl32_data_sl.pkl'


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
        self._actor = Actor(50, 2)

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
    return abs(st[46] - 1) < 1e-9


class EvalAgent:

    def __init__(self, n_agents):
        self._agent = Policy()
        self._n_agents = n_agents
        self._env = EnvWrapper(n_agents)
        self._predict_model = PredictModel()
        self._predict_model.load_state_dict(torch.load(SL_MODEL_PATH))
        # self._env.set_global_center(True)

    def eval_one_ep(self):
        enum_seq = [f'uav_{i}' for i in range(self._n_agents)]
        states = self._env.reset()
        done = {n: False for n in enum_seq}
        total_rew = {n: 0 for n in enum_seq}

        dtc_seq = []

        while not real_done(done):
            actions = {}

            for s in states.values():
                if not is_die_state(s):
                    data_collection.append(s)

            for seq in enum_seq:
                with torch.no_grad():
                    states[seq][-2:] = self._predict_model(torch.tensor(states[seq][:-2]).float()).numpy()
                action = self._agent.choose_action(states[seq])
                actions[seq] = action
            next_states, rewards, done, info = self._env.step(actions)
            # self._env.render()
            for seq in enum_seq:
                total_rew[seq] += rewards[seq]
            states = next_states
            dtc_seq.append(self._env.get_dtc_avg())
        ta = self._env.get_ta()
        die = self._env.get_die()
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


def main():
    agent = EvalAgent(N_Agents)
    agent.setGlobal(False)
    agent.load(POLICY_PATH)

    ta, die, dtc = [], [], []
    for i in range(100):
        print(f'--{i}--')
        _r, _ta, _die, _dtc = agent.eval_one_ep()
        ta.append(_ta)
        die.append(_die)
        dtc.append(_dtc)
        print(_r)
        print(f'R={_r}, TA={_ta}, D={_die}, DTC={_dtc}')

    with open(f'{OUTPUT_PATH}', 'wb') as f:
        pickle.dump((ta, die, dtc), f)


if __name__ == '__main__':
    main()
