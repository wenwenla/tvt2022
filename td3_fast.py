import argparse
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from uav_2d_ma_fast import EnvWrapper
from cppenv.env import get_version

parser = argparse.ArgumentParser(description='Input n_agents and main folder')
parser.add_argument('--agents', type=int)
parser.add_argument('--folder', type=str)
parser.add_argument('--global_', type=str)

args = parser.parse_args()

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
TAU = 0.05
STD = 0.1
TARGET_STD = 0.1
DELAY = 2
GAMMA = 0.95
BATCH_SIZE = 128
START_UPD_SAMPLES = 2000

N_AGENTS = args.agents
MAIN_FOLDER = args.folder


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


class Critic(torch.nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc0 = torch.nn.Linear(obs_dim + action_dim, 128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        q = self.fc2(x)
        return q


class ReplayBuffer:

    def __init__(self, cap, state_dim, action_dim):
        self._states = np.zeros((cap, state_dim))
        self._actions = np.zeros((cap, action_dim))
        self._rewards = np.zeros((cap,))
        self._next_states = np.zeros((cap, state_dim))
        self._index = 0
        self._cap = cap
        self._is_full = False
        self._rnd = np.random.RandomState(19971023)

    def add(self, states, actions, rewards, next_states):
        self._states[self._index] = states
        self._actions[self._index] = actions
        self._rewards[self._index] = rewards
        self._next_states[self._index] = next_states

        self._index += 1
        if self._index == self._cap:
            self._is_full = True
            self._index = 0

    def sample(self, n):
        indices = self._rnd.randint(0, self._cap if self._is_full else self._index, (n,))
        s = self._states[indices]
        a = self._actions[indices]
        r = self._rewards[indices]
        s_ = self._next_states[indices]
        return s, a, r, s_

    def n_samples(self):
        return self._cap if self._is_full else self._index


class TD3Agent:

    def __init__(self, obs_dim, act_dim):
        self._actor = Actor(obs_dim, act_dim)
        self._critic = [Critic(obs_dim, act_dim) for _ in range(2)]
        self._target_actor = Actor(obs_dim, act_dim)
        self._target_critic = [Critic(obs_dim, act_dim) for _ in range(2)]

        self._target_actor.load_state_dict(self._actor.state_dict())
        for i in range(2):
            self._target_critic[i].load_state_dict(self._critic[i].state_dict())

        self._actor_opt = torch.optim.Adam(self._actor.parameters(), lr=ACTOR_LR)
        self._critic_opt = [
            torch.optim.Adam(self._critic[i].parameters(), lr=CRITIC_LR) for i in range(2)
        ]

        self._act_dim = act_dim
        self._obs_dim = obs_dim
        self._sw = SummaryWriter(f'./{MAIN_FOLDER}/logs')
        self._step = 0

    def soft_upd(self):
        with torch.no_grad():
            for t, s in zip(self._target_actor.parameters(), self._actor.parameters()):
                t.copy_((1 - TAU) * t.data + TAU * s.data)
            for t, s in zip(self._target_critic[0].parameters(), self._critic[0].parameters()):
                t.copy_((1 - TAU) * t.data + TAU * s.data)
            for t, s in zip(self._target_critic[1].parameters(), self._critic[1].parameters()):
                t.copy_((1 - TAU) * t.data + TAU * s.data)

    def query_target_action(self, obs):
        o = torch.tensor(obs).float()
        with torch.no_grad():
            a = self._target_actor(o)
            a = a.detach().cpu().numpy()
        target_noise = np.random.normal(0, TARGET_STD, a.shape)
        return a + target_noise

    def choose_action(self, obs):
        o = torch.tensor(obs).float()
        with torch.no_grad():
            a = self._actor(o)
            a = a.detach().cpu().numpy()
        return a

    def choose_action_with_exploration(self, obs):
        noise = np.random.normal(0, STD, (self._act_dim,))
        a = self.choose_action(obs)
        a += noise
        return np.clip(a, -1, 1)

    def update(self, s, a, r, s_, a_):
        self._step += 1
        s_tensor = torch.tensor(s).float()
        a_tensor = torch.tensor(a).float()
        r_tensor = torch.tensor(r).float().view(-1, 1)
        next_s_tensor = torch.tensor(s_).float()
        next_a_tensor = torch.tensor(a_).float()

        if len(a_tensor.shape) == 1:
            a_tensor = a_tensor.view(-1, 1)
        if len(next_a_tensor.shape) == 1:
            next_a_tensor = next_a_tensor.view(-1, 1)

        self._actor_opt.zero_grad()
        self._critic_opt[0].zero_grad()
        self._critic_opt[1].zero_grad()

        # update critic
        next_sa_tensor = torch.cat([next_s_tensor, next_a_tensor], dim=1)
        with torch.no_grad():
            m = torch.min(self._target_critic[0](next_sa_tensor), self._target_critic[1](next_sa_tensor))
            target_q = r_tensor + GAMMA * m
        now_sa_tensor = torch.cat([s_tensor, a_tensor], dim=1)
        q_loss_log = [0, 0]
        for i in range(2):
            now_q = self._critic[i](now_sa_tensor)
            q_loss_fn = torch.nn.MSELoss()
            q_loss = q_loss_fn(now_q, target_q)
            self._critic_opt[i].zero_grad()
            q_loss.backward()
            self._critic_opt[i].step()
            q_loss_log[i] = q_loss.detach().cpu().item()

        # update actor
        a_loss_log = 0
        if self._step % DELAY == 0:
            new_a_tensor = self._actor(s_tensor)
            new_sa_tensor = torch.cat([s_tensor, new_a_tensor], dim=1)
            q = -self._critic[0](new_sa_tensor).mean()
            self._actor_opt.zero_grad()
            q.backward()
            self._actor_opt.step()
            a_loss_log = q.detach().cpu().item()
            self.soft_upd()

        if self._step % 500 == 0:
            self._sw.add_scalar('loss/critic_0', q_loss_log[0], self._step)
            self._sw.add_scalar('loss/critic_1', q_loss_log[1], self._step)
            self._sw.add_scalar('loss/actor', a_loss_log, self._step)

    def policy_state_dict(self):
        return self._actor.state_dict()

    def value_state_dict(self):
        return [self._critic[i].state_dict() for i in range(2)]


def real_done(done):
    for v in done.values():
        if not v:
            return False
    return True


class TD3Trainer:

    def __init__(self, n_agents):
        self._n_agents = n_agents
        self._obs_dim = 50
        self._action_dim = 2

        self._agent = TD3Agent(self._obs_dim, self._action_dim)
        self._replay_buffer = ReplayBuffer(1000000, self._obs_dim, self._action_dim)
        self._env = EnvWrapper(n_agents)
        self._now_ep = 0
        self._sw = SummaryWriter(f'./{MAIN_FOLDER}/logs/trainer')
        self._step = 0
        
    def _sample_global(self):
        if self._now_ep < 10000:
            self._env.set_global_center(True)
        elif self._now_ep > 25000:
            self._env.set_global_center(False)
        else:
            magic = np.random.uniform()
            p = -(self._now_ep - 25000) / 15000
            self._env.set_global_center(magic < p)
        
    def train_one_episode(self):
        
        if args.global_ == 'GLOBAL':
            self._env.set_global_center(True)
        elif args.global_ == 'LOCAL':
            self._env.set_global_center(False)
        elif args.global_ == 'ANNEAL':
            self._sample_global()
        else:
            assert False
        
        self._now_ep += 1

        enum_seq = [f'uav_{i}' for i in range(self._n_agents)]

        states = self._env.reset()
        done = {n: False for n in enum_seq}
        total_rew = {n: 0 for n in enum_seq}

        while not real_done(done):
            actions = {}
            in_states = []
            for seq in enum_seq:
                in_states.append(states[seq])
            out_actions = self._agent.choose_action_with_exploration(in_states)
            for i, seq in enumerate(enum_seq):
                actions[seq] = out_actions[i]

            die = self._env.getStatus()
            choices = []
            for i in range(self._n_agents):
                if not die[i]:
                    choices.append(i)
            
            if args.global_ == 'GLOBAL':
                self._env.set_global_center(True)
            elif args.global_ == 'LOCAL':
                self._env.set_global_center(False)
            elif args.global_ == 'ANNEAL':
                self._sample_global()
            
            next_states, rewards, done, info = self._env.step(actions)
            self._step += 1

            buffer_index = np.random.choice(choices)   # np.random.randint(0, self._n_agents)
            buffer_name = enum_seq[buffer_index]
            self._replay_buffer.add(states[buffer_name], actions[buffer_name], rewards[buffer_name],
                                    next_states[buffer_name])
            if self._step % 50 == 0 and self._replay_buffer.n_samples() > START_UPD_SAMPLES:
                for _ in range(20):
                    s, a, r, s_ = self._replay_buffer.sample(BATCH_SIZE)
                    a_ = self._agent.query_target_action(s_)
                    self._agent.update(s, a, r, s_, a_)

            for seq in enum_seq:
                total_rew[seq] += rewards[seq]
            states = next_states
            if self._now_ep % 200 == 0:
                self._sw.add_scalar(f'train_rew/0', total_rew['uav_0'], self._now_ep)
        return total_rew

    def test_one_episode(self):
        self._env.set_global_center(False)
        enum_seq = [f'uav_{i}' for i in range(self._n_agents)]

        states = self._env.reset()
        done = {n: False for n in enum_seq}
        total_rew = {n: 0 for n in enum_seq}

        while not real_done(done):
            actions = {}
            in_states = []
            for seq in enum_seq:
                in_states.append(states[seq])
            out_actions = self._agent.choose_action(in_states)
            for i, seq in enumerate(enum_seq):
                actions[seq] = out_actions[i]

            next_states, rewards, done, info = self._env.step(actions)

            for seq in enum_seq:
                total_rew[seq] += rewards[seq]
            states = next_states
        for i, seq in enumerate(enum_seq):
            self._sw.add_scalar(f'test_rew/{i}', total_rew[seq], self._now_ep)
        return total_rew

    def save(self):
        path = f'./{MAIN_FOLDER}/models'
        if not os.path.exists(path):
            os.makedirs(path)
        save_pth = path + '/' + f'{self._now_ep}.pkl'
        torch.save([self._agent.policy_state_dict(), *self._agent.value_state_dict()], save_pth)


def main():
    print(f'START AGENTS: {N_AGENTS} FOLDER: {MAIN_FOLDER}')
    torch.set_num_threads(1)
    trainer = TD3Trainer(N_AGENTS)

    for i in range(40001):
        print(f'{i} -> version: {get_version()}')
        r = trainer.train_one_episode()
        if i % 200 == 0:
            trainer.save()
            trainer.test_one_episode()


if __name__ == '__main__':
    main()
