import gymnasium as gym
from gymnasium import spaces

import numpy as np

class SimpleTrader(gym.Env):
    def __init__(self, states, initial_balance=1e6):
        super(SimpleTrader, self).__init__()

        self.states = states

        self.initial_balance = np.array([initial_balance])
        self.balance = np.array([initial_balance])

        self.previous_networth = self.initial_balance
        self.networth = self.initial_balance

        self.initial_shares = np.zeros(self.states.n_stocks)
        self.shares = np.zeros(self.states.n_stocks)

        self.time_index = 0

        # 0 for hold, 1 for buy，2 for sell
        self.action_space = spaces.MultiDiscrete([3] * states.n_stocks)
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(states.n_stocks,), dtype=np.float32)

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
          if action[i] == 1: # buy
            # Buy 5% of balance in shares
            total_possible = self.balance / buy_price[i]
            shares_bought = total_possible * 0.05
            cost = shares_bought * buy_price[i] 
            self.shares[i] += shares_bought
            self.balance -= self.balance * 0.05

          elif action[i] == 2:  # sell
            # Sell 5% of shares held
            shares_sold = self.shares[i] * 0.05
            self.shares[i] -= shares_sold
            self.balance += sell_price[i] * self.shares[i] * 0.05

          else:
            pass

    def set_networth(self):
        self.previous_networth = self.networth
        self.networth = self.balance + np.sum(self.shares * self.states.get_sell_price(self.time_index))

    def render(self, mode='human', close=False):
        print(f'Time Index: {self.time_index}')
        print(f'Shares: {self.shares}')
        print(f'Balance: {self.balance}')
        print(f'Networth: {self.networth}')