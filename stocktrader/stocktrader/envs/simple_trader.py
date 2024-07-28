import gymnasium as gym
from gymnasium import spaces

import numpy as np


class SimpleTrader(gym.Env):
    def __init__(self, states, initial_balance=1e6, percent=0.05):
        super(SimpleTrader, self).__init__()

        self.states = states

        self.initial_balance = np.copy(np.array([initial_balance]))
        self.balance = np.array([initial_balance])

        self.previous_networth = self.initial_balance
        self.networth = self.initial_balance

        self.initial_shares = np.zeros(self.states.n_stocks)
        self.shares = np.zeros(self.states.n_stocks)

        self.percent = percent

        self.time_index = 0

        # 0 for buy, 1 for sell, 2 for hold
        self.action_space = spaces.MultiDiscrete([3] * states.n_stocks)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.states.get_state(0).size + 2 * self.states.n_stocks,), dtype=np.float32)

        self.state = np.concatenate([self.states.get_state(0), self.balance, self.shares])


    def reset(self):
        self.balance = np.copy(self.initial_balance)
        self.networth = np.copy(self.initial_balance)
        self.shares = np.copy(self.initial_shares)
        self.state = np.concatenate([self.states.get_state(0), self.balance, self.shares])
        self.time_index = 0
        return self.state

    def step(self, action):
        self.take_action(action)
        self.set_networth()

        epsilon = 1e-8  # small value to avoid division by zero
        reward = np.log((self.networth + epsilon) / (self.previous_networth + epsilon))

        self.time_index += 1
        self.update_state()

        done = self.time_index >= len(self.states.df) - 1
        info = {}

        return self.state, reward, done, info

    def update_state(self):
        self.state = np.concatenate([self.states.get_state(self.time_index), self.balance, self.shares])

    def take_action(self, action):
        sell_price = self.states.get_sell_price(self.time_index)
        buy_price = self.states.get_buy_price(self.time_index)

        buy_actions = (1 - np.clip(action, 0, 1)) * self.percent

        for i in range(len(action)):
            if action[i] == 0:  # buy
                # Buy 5% of balance in shares
                self.shares[i] += self.balance * buy_actions[i] / buy_price[i]

        self.balance *= (1 - np.sum(buy_actions))

        for i in range(len(action)):
          if action[i] == 1:  # sell
            # Sell 5% of shares held
            shares_sold = self.shares[i] * self.percent
            self.shares[i] -= self.shares[i] * self.percent
            self.balance += shares_sold * sell_price[i]

    def set_networth(self):
        self.previous_networth = self.networth
        self.networth = self.balance + np.sum(self.shares * self.states.get_sell_price(self.time_index))

    def render(self, mode='human', close=False):
        print(f'Time Index: {self.time_index}')
        print(f'Shares: {self.shares}')
        print(f'Balance: {self.balance}')
        print(f'Networth: {self.networth}')