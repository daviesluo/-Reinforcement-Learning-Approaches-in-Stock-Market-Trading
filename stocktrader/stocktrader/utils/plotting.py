import numpy as np
import matplotlib.pyplot as plt

def training_plot(dates, cumulative_rewards, stock_data, networths, transactions, actions, action_params, plot_episodes, figsize = (14,10)):
    fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize=figsize, sharex=False)

    # Stock prices
    ax = axes[0][0]

    amd, nvda = stock_data
    ax.plot(dates, amd, label='AMD')
    ax.plot(dates, nvda, label='Nvidia')
    ax.set_ylabel('Stock Price')
    ax.set_title('Stock Price over Time')
    ax.set_xlabel('Date')
    ax.legend()

    # Cumulative Rewards per year
    ax = axes[0][1]
    for i in range(len(plot_episodes)):
        ax.plot(dates, cumulative_rewards[i], label = f'Episode {plot_episodes[i]}')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward per Year')
    ax.legend()

    # Plot action distribution of the last episode
    ax = axes[1][0]

    alphas, betas = action_params
    amd_alphas = alphas[0]
    amd_betas = betas[0]
    nvda_alphas = alphas[1]
    nvda_betas = betas[1]

    amd_mean, amd_sd = get_mean_sd(amd_alphas, amd_betas)
    nvda_mean, nvda_sd = get_mean_sd(nvda_alphas, nvda_betas)

    ax.plot(dates, amd_mean, 'r-', label='AMD Mean Action')
    ax.plot(dates, nvda_mean, 'b-', label='Nvidia Mean Action')

    ax.plot(dates, actions[:, 0], 'r--', label='AMD Action')
    ax.plot(dates, actions[:, 1], 'b--', label='Nvidia Action')

    ax.fill_between(dates, amd_mean - 2 * amd_sd, amd_mean + 2 * amd_sd, color='red', alpha=0.2)
    ax.fill_between(dates, nvda_mean - 2 * nvda_sd, nvda_mean + 2 * nvda_sd, color='blue', alpha=0.2)

    ax.set_ylabel('Action')
    ax.set_title('Action Distribution of the Last Episode')
    ax.set_xlabel('Year')
    ax.legend()

    # Plot transactions of the last episode
    ax = axes[1][1]

    ax.plot(dates, transactions[:, 0], 'r-', label='AMD Transaction')
    ax.plot(dates, transactions[:, 1], 'b-', label='Nvidia Transaction')

    ax.set_ylabel('Balance Change')
    ax.set_title('Transactions of the Last Episode')
    ax.set_xlabel('Year')
    ax.legend()

    # Plot change in networth of the last episode
    ax = axes[1][2]

    ax.plot(dates, networths, label='Networth')

    ax.set_ylabel('Networth')
    ax.set_title('Change in Networth of the Last Episode')
    ax.set_xlabel('Year')
    ax.legend()

    plt.tight_layout()
    return fig


def get_mean_sd(alphas, betas):
    means = 2 * (alphas / (alphas + betas)) - 1
    sds = 2 * np.sqrt(alphas * betas / ((alphas + betas) ** 2 * (alphas + betas + 1)))

    return means, sds
