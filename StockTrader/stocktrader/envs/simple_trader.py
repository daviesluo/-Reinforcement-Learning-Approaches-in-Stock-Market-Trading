import gymnasium as gym
from gymnasium import spaces

import numpy as np

class SimpleTrader(gym.Env):
    def __init__(self, initial_cash=1000, initial_stocks=10, stock_price=10):
        super(StockTradingEnv, self).__init__()
        
        # Initial conditions
        self.initial_cash = initial_cash
        self.initial_stocks = initial_stocks
        self.stock_price = stock_price
        
        # State: [cash, stocks]
        self.state = np.array([self.initial_cash, self.initial_stocks], dtype=np.float32)
        
        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [cash, stocks]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)
    
    def step(self, action):
        cash, stocks = self.state
        
        # Implement logic for each action
        if action == 1:  # Buy
            if cash >= self.stock_price:
                cash -= self.stock_price
                stocks += 1
            else:
                # Not enough cash to buy, handle invalid action
                pass  # Or give a penalty if desired
        elif action == 2:  # Sell
            if stocks > 0:
                cash += self.stock_price
                stocks -= 1
            else:
                # No stocks to sell, handle invalid action
                pass  # Or give a penalty if desired
        
        # Update the state
        self.state = np.array([cash, stocks], dtype=np.float32)
        
        # Calculate reward (example: net worth)
        reward = cash + stocks * self.stock_price
        
        # Example termination condition (optional)
        done = False  # For simplicity, let's assume this episode never ends
        
        info = {}
        
        return self.state, reward, done, info
    
    def reset(self):
        # Reset the state of the environment to the initial state
        self.state = np.array([self.initial_cash, self.initial_stocks], dtype=np.float32)
        return self.state
    
    def render(self, mode='human'):
        cash, stocks = self.state
        print(f"Cash: {cash}, Stocks: {stocks}")
    
    def close(self):
        pass