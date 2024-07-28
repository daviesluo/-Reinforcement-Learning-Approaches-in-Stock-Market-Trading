# Define Policy Network
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from . import training_plot

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Training process of Policy Network
def train_policy_gradient(env, policy_network, opt, n_ep, eps, plot_episodes=None, print_n=50, figsize = (14,10), no_action_penalty=None):

    if plot_episodes is not None:
        cumulative_rewards = []

    for episode in range(n_ep):
        state = env.reset()
        done = False
        total_reward = 0
        cumulative_reward = 0

        cumulative_rewards_ep = []

        while not done:
            state = np.array(state, dtype=np.float32)
            state = (state - state.mean()) / (state.std() + 1e-8)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            alpha, beta = policy_network(state_tensor)

            # Apply exploratory level to alpha and beta to increase variance
            prob_rand_action = eps * (n_ep - episode) / n_ep
            # alpha += exploration_factor
            # beta += exploration_factor

            action_dist = torch.distributions.Beta(alpha, beta)
            u = random.random()
            if u < prob_rand_action:
                action = torch.rand(2) 
                action_np = 2 * action.cpu().numpy() - 1
            else:
                action = action_dist.sample()
                action_np = action.detach().cpu().numpy().flatten()  # sample from Beta
                action_np = 2 * action_np - 1  # scale from [0, 1] to [-1, 1]

            # Apply no-action penalty
            if no_action_penalty is not None:
              if np.all(np.abs(action_np) < 0.1):
                  reward -= no_action_penalty

            next_state, reward, done, info = env.step(action_np)

            opt.zero_grad()
            reward_tensor = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
            log_prob = action_dist.log_prob(action).sum()
            loss = -log_prob * reward_tensor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=1.0)
            opt.step()

            state = next_state
            total_reward += reward
            cumulative_reward += reward

            if plot_episodes is not None: # TODO: logic for running on whole dataset???
                if episode in plot_episodes:
                    cumulative_rewards_ep.append(cumulative_reward.copy())

        if plot_episodes is not None:
            if episode in plot_episodes:
                cumulative_rewards.append(np.concatenate(cumulative_rewards_ep))

        if episode % print_n == 0:
            print(f"Episode {episode + 1}: Total Reward: {total_reward}, Balance: {env.balance}, Net Worth: {env.networth}, Shares (AMD: {env.shares[0]}, Nvidia: {env.shares[1]})")


    # get final action distribution for plotting (run once after training)  
    if plot_episodes is not None:
        alphas = [] # alpha parameters
        betas = [] # beta parameters

        amd_prices = [] # amd prices
        nvda_prices = [] # nvda prices

        networths = [] # networth over time
        transactions = [] # transactions over time
        actions = [] # actions taken

        dates = [] # dates

        state = env.reset()
        done = False
        while not done:
            amd_prices.append(env.states.df.iloc[env.time_index]['AMD_Close']) #amd
            nvda_prices.append(env.states.df.iloc[env.time_index]['NVDA_Close']) #nvda
            dates.append(env.states.df.iloc[env.time_index]['DateTime']) #dates

            state = np.array(state, dtype=np.float32)
            state = (state - state.mean()) / (state.std() + 1e-8)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            alpha, beta = policy_network(state_tensor)

            alphas.append(alpha.detach().cpu())
            betas.append(beta.detach().cpu())

            action_dist = torch.distributions.Beta(alpha, beta)
            action = action_dist.sample().detach().cpu()
            action_np = 2 * action.detach().numpy().flatten() - 1  # scale from [0, 1] to [-1, 1]
            next_state, reward, done, info = env.step(action_np)
            state = next_state

            networths.append(env.networth)
            actions.append(action_np.copy())
            transactions.append(info['balance_change'])

        action_params = [torch.stack(alphas).squeeze(1).T,torch.stack(betas).squeeze(1).T]
        stock_data = [amd_prices, nvda_prices]
        transactions = np.stack(transactions)
        networths = np.array(networths)
        actions = np.stack(actions)

        return training_plot(dates, cumulative_rewards, stock_data, networths, transactions, actions, action_params, plot_episodes, figsize=figsize)
    else:
        return None