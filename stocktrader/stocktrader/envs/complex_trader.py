import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ComplexTrader(gym.Env):
    def __init__(self, states, initial_balance=1e6):
        super(ComplexTrader, self).__init__()

        self.states = states
        self.initial_balance = np.copy(np.array([initial_balance]))
        self.balance = np.array([initial_balance])

        self.previous_networth = self.initial_balance
        self.networth = self.initial_balance

        self.initial_shares = np.zeros(self.states.n_stocks)
        self.shares = np.zeros(self.states.n_stocks)

        self.time_index = 0

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.states.n_stocks,), dtype=np.float32)
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
        balance_change, share_change = self.take_action(action)
        self.set_networth()

        epsilon = 1e-8  # small value to avoid division by zero
        reward = np.log(self.networth + epsilon) - np.log(self.previous_networth + epsilon)

        self.time_index += 1
        self.update_state()

        done = self.time_index >= len(self.states.df) - 1
        info = {'balance_change' : balance_change, 'share_change' : share_change}

        return self.state, reward, done, info

    def update_state(self):
        self.state = np.concatenate([self.states.get_state(self.time_index), self.balance, self.shares])

    def take_action(self, action):
        sell_price = self.states.get_sell_price(self.time_index)
        buy_price = self.states.get_buy_price(self.time_index)

        balance_change = np.zeros(self.states.n_stocks)
        share_change = np.zeros(self.states.n_stocks)

        for i in range(len(action)):
            action_value = action[i].item()
            if action_value < 0:  # sell
                sell_amount = min(self.shares[i], -action_value * self.shares[i]).item()
                share_change[i] = -sell_amount
                balance_change[i] = sell_amount * sell_price[i]
                self.balance += balance_change[i]

        for i in range(len(action)):
            action_value = action[i].item()
            if action_value >= 0:  # buy
                buy_amount = (min(self.balance, action_value * self.balance) / buy_price[i]).item()
                share_change[i] = buy_amount
                balance_change[i] = -buy_amount * buy_price[i]
                self.balance += balance_change[i]

        self.shares += share_change

        return balance_change, share_change

    def set_networth(self):
        self.previous_networth = self.networth
        self.networth = self.balance + np.sum(self.shares * self.states.get_sell_price(self.time_index))

    def render(self, mode='human', close=False):
        print(f'Time Index: {self.time_index}')
        print(f'Shares: {self.shares}')
        print(f'Balance: {self.balance}')
        print(f'Networth: {self.networth}')

# class ComplexTrader(gym.Env):
#     def __init__(self, states, initial_balance=1e6):
#         super(ComplexTrader, self).__init__()

#         self.states = states
#         self.initial_balance = np.copy(np.array([initial_balance]))
#         self.balance = np.array([initial_balance])
#         self.previous_networth = self.initial_balance
#         self.networth = self.initial_balance
#         self.initial_shares = np.zeros(self.states.n_stocks)
#         self.shares = np.zeros(self.states.n_stocks)
#         self.time_index = 0

#         self.action_space = spaces.Box(low=-1, high=1, shape=(self.states.n_stocks,), dtype=np.float32)
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.states.get_state(0).size + 2 * self.states.n_stocks,), dtype=np.float32)

#         self.state = np.concatenate([self.states.get_state(0), self.balance, self.shares])

#     def reset(self):
#         self.balance = np.copy(self.initial_balance)
#         self.networth = np.copy(self.initial_balance)
#         self.shares = np.copy(self.initial_shares)
#         self.state = np.concatenate([self.states.get_state(0), self.balance, self.shares])
#         self.time_index = 0
#         return self.state

#     def step(self, action):
#         self.take_action(action)
#         self.set_networth()

#         epsilon = 1e-8  # small value to avoid division by zero
#         reward = np.log(self.networth + epsilon) - np.log(self.previous_networth + epsilon)

#         self.time_index += 1
#         self.update_state()

#         done = self.time_index >= len(self.states.df) - 1
#         info = {}

#         return self.state, reward, done, info

#     def update_state(self):
#         self.state = np.concatenate([self.states.get_state(self.time_index), self.balance, self.shares])

#     def take_action(self, action):
#         sell_price = self.states.get_sell_price(self.time_index)
#         buy_price = self.states.get_buy_price(self.time_index)

#         for i in range(len(action)):
#             action_value = action[i].item()
#             if action_value < 0:  # sell
#                 sell_amount = min(self.shares[i], -action_value * self.shares[i])
#                 self.shares[i] -= sell_amount
#                 self.balance += sell_amount * sell_price[i]

#         for i in range(len(action)):
#             action_value = action[i].item()
#             if action_value >= 0:  # buy
#                 buy_amount = min(self.balance, action_value * self.balance) / buy_price[i]
#                 self.shares[i] += buy_amount
#                 self.balance -= buy_amount * buy_price[i]

#     def get_transaction(self, action):
#         sell_price = self.states.get_sell_price(self.time_index)
#         buy_price = self.states.get_buy_price(self.time_index)

#         balance_change = []

#         for i in range(len(action)):
#             action_value = action[i].item()
#             if action_value < 0:  # sell
#                 sell_amount = min(self.shares[i], -action_value * self.shares[i])
#                 balance_change.append(sell_amount * sell_price[i])

#         for i in range(len(action)):
#             action_value = action[i].item()
#             if action_value >= 0:  # buy
#                 buy_amount = min(self.balance, action_value * self.balance) / buy_price[i]
#                 balance_change.append(-buy_amount * buy_price[i])

#         return balance_change

#     def set_networth(self):
#         self.previous_networth = self.networth
#         self.networth = self.balance + np.sum(self.shares * self.states.get_sell_price(self.time_index))

#     def render(self, mode='human', close=False):
#         print(f'Time Index: {self.time_index}')
#         print(f'Shares: {self.shares}')
#         print(f'Balance: {self.balance}')
#         print(f'Networth: {self.networth}')
