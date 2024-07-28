import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=3):
        super(PolicyNetwork, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(hidden_size, 2 * output_size))  # output alpha and beta parameters
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        alpha_beta = F.softplus(x) + 1e-3  # ensure alpha and beta are positive and non-zero
        return alpha_beta[:, :x.size(1)//2], alpha_beta[:, x.size(1)//2:]  # split into alpha and beta
