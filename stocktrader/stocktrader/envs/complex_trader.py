import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ComplexTrader(gym.Env):
    def __init__(self, states, initial_balance=1e6):
        super(ComplexTrader, self).__init__()

        self.states = states

        self.initial_balance = np.array([initial_balance])
        self.balance = np.array([initial_balance])

        self.previous_networth = self.initial_balance
        self.networth = self.initial_balance

        self.initial_shares = np.zeros(self.states.n_stocks)
        self.shares = np.zeros(self.states.n_stocks)

        self.time_index = 0

        self.action_space = spaces.Box(low=-1, high=1, shape=(states.n_stocks,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.states.shape, dtype=np.float32)

        self.state = np.concatenate([self.states.get_state(0).flatten(), self.balance, self.shares])

    def reset(self):
        self.balance = self.initial_balance
        self.networth = self.initial_balance
        self.shares = self.initial_shares

        self.state = np.concatenate([self.states.get_state(0).flatten(), self.balance, self.shares])
        self.time_index = 0

        return self.state

    def step(self, action):
        self.take_action(action)
        self.set_networth()

        reward = np.log(self.networth / self.previous_networth)

        self.time_index += 1
        self.update_state()

        done = self.time_index >= len(self.states.df) - 1
        info = {}

        return self.state, reward, done, info

    def update_state(self):
        self.state = np.concatenate([self.states.get_state(self.time_index).flatten(), self.balance, self.shares])

    def take_action(self, action):
        sell_price = self.states.get_sell_price(self.time_index)
        buy_price = self.states.get_buy_price(self.time_index)

        for i in range(len(action)):
            if action[i] < 0:  # sell
                self.shares[i] *= (1 + action[i].item())
                self.balance += sell_price[i] * self.shares[i] * (-action[i].item())

        for i in range(len(action)):
            if action[i] >= 0:  # buy
                self.shares[i] += self.balance * action[i].item() / buy_price[i]
                self.balance -= self.balance * action[i].item()

    def set_networth(self):
        self.previous_networth = self.networth
        self.networth = self.balance + np.sum(self.shares * self.states.get_sell_price(self.time_index))

    def render(self, mode='human', close=False):
        print(f'Time Index: {self.time_index}')
        print(f'Shares: {self.shares}')
        print(f'Balance: {self.balance}')
        print(f'Networth: {self.networth}')