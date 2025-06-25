from rlmep.agents.base import Agent

import numpy as np
import torch
from rich.progress import Progress
import gymnasium as gym

from torch.distributions import Normal, HalfNormal

class PolicyNetwork(torch.nn.Module):

    def __init__(self, obs_size, n_actions, lr=1e-3, hidden_size=64):
        super(PolicyNetwork, self).__init__()

        self.shared_layers = torch.nn.Sequential(
            torch.nn.Linear(obs_size, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.SiLU()
        )

        self.mean_net = torch.nn.Linear(hidden_size, n_actions)
        self.stddev_net = torch.nn.Linear(hidden_size, n_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.eps = 1e-6

    def forward(self, x):
        x = self.shared_layers(x)
        means = self.mean_net(x) ** 2
        stddevs = self.stddev_net(x) ** 2
        return means, stddevs

    def sample_action(self, x):
        means, stddevs = self.forward(x)
        dist = Normal(means, stddevs)
        action = dist.sample()
        logp = dist.log_prob(action).sum(dim=-1)
        return action, logp

class Trajectory:

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def append(self, state, action, reward, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def __len__(self):
        return len(self.states)

class ReinforceAgent(Agent):

    def __init__(self, policy: torch.nn.Module, env: gym.Env):
        self.policy = policy
        self.env = env

    def get_action(self, state):
        action, log_probs = self.policy.sample_action(torch.tensor(state, dtype=torch.float32))
        return action

    def play_episode(self):

        state, _ = self.env.reset()
        terminal = False
        truncated = False

        trajectory = Trajectory()
        while not terminal and not truncated: 
            
            # Get the action and the corresponding log_prob from the policy. 
            # Remember to convert the state to a torch tensor. 
            action, log_probs = self.policy.sample_action(torch.tensor(state, dtype=torch.float32))

            # Take a step in the environment with the action:
            next_state, reward, terminal, truncated, info = self.env.step(action)

            # Append the step to the trajectory:
            trajectory.append(state, action, reward, log_probs)

            # Update the state:
            state = next_state

        return trajectory
    
    def update(self, trajectory, gamma):

        T = len(trajectory)

        # Calculate G_t for each time step
        G = []
        for t in range(T):
            Gt = np.sum([trajectory.rewards[k] * gamma**(k-t) for k in range(t, T)])
            G.append(Gt)
            
        G = torch.tensor(np.array(G))

        # Calculate the gradients of the log-prob
        log_probs_tensor = torch.stack(trajectory.log_probs)
        loss = -log_probs_tensor * G

        # Update the policy
        self.policy.optimizer.zero_grad()
        loss.sum().backward()
        self.policy.optimizer.step()

    def train(self, episodes=100, gamma=0.99):

        returns = np.zeros(episodes)

        for episode in Progress(range(episodes)):

            # Play an episode:
            trajectory = self.play_episode()

            # Update the policy:
            self.update(trajectory, gamma)

            # Print the total reward of the episode:
            returns[episode] = np.sum(trajectory.rewards)

        return returns