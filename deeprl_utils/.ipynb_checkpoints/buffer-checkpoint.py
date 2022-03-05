import numpy as np


class ReplayBuffer:

    def __init__(self, cap, state_shape, action_dim):
        self._states = np.zeros((cap, *state_shape), dtype=np.float32)
        self._actions = np.zeros((cap, action_dim))
        self._rewards = np.zeros((cap, ))
        self._states_next = np.zeros((cap, *state_shape), dtype=np.float32)
        self._done = np.zeros((cap, ), dtype=np.bool)
        self._last_index = 0
        self._full = False
        self._cap = cap
        self._random_state = np.random.RandomState(19971023)

    def add(self, s, a, r, s_, done):
        self._states[self._last_index] = s
        self._actions[self._last_index] = a
        self._rewards[self._last_index] = r
        self._states_next[self._last_index] = s_
        self._done[self._last_index] = done
        self._last_index += 1
        if self._last_index == self._cap:
            self._full = True
            self._last_index = 0

    def sample(self, n):
        if not self._full:
            indices = self._random_state.randint(0, self._last_index, (n, ))
        else:
            indices = self._random_state.randint(0, self._cap, (n, ))
        return self._states[indices], self._actions[indices], self._rewards[indices], self._states_next[indices], self._done[indices]
