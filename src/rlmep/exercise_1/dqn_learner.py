from copy import deepcopy
import numpy as np
import torch
from rich.progress import track

import gymnasium as gym

from typing import Callable
from rlmep.exercise_1 import ExperienceReplay

class DQN:
    def __init__(
        self,
        main_network: torch.nn.Module,
        target_network: torch.nn.Module,
        replay: ExperienceReplay,
        gamma: float = 0.90,
        epsilon: Callable = lambda x: 0.1,
        train_interval: int = 10,
        copy_interval: int = 100,
    ):
        """
        Initializes the DQN learner with the environment, main network, target network, and replay buffer

        Parameters
        ----------
        main_network : torch.nn.Module
            The main neural network used for action-value function approximation.
        target_network : torch.nn.Module
            The target neural network used for stabilizing training.
        replay : ExperienceReplay
            The experience replay buffer used to store and sample experiences.
        """
        self.main_network = main_network
        self.target_network = target_network
        self.copy_weights()
        self.replay = replay

        self.gamma = gamma
        self.epsilon = epsilon
        self.train_interval = train_interval
        self.copy_interval = copy_interval

    def get_action(self, env: gym.Env, state: torch.Tensor, episode: int, apply_epsilon: bool = True) -> int:
        """
        Chooses an action based on the current state and the epsilon-greedy policy.

        If a random number is less than epsilon, a random action is selected.
        Otherwise, the action with the highest Q-value is selected.        
        """

        if np.random.rand() < self.epsilon(episode) and apply_epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                Q = self.main_network(state_tensor)
                action = torch.argmax(Q).item()
        return action
    
    def rollout(
        self,
        env: gym.Env,
        episode: int,
        train: bool = True,
        apply_epsilon: bool = True,
    ):
        state, _ = env.reset()
        terminal = False
        truncated = False
        return_value = 0

        while not terminal and not truncated:
            # Choose action
            action = self.get_action(
                env=env,
                state=state,
                episode=episode,
                apply_epsilon=apply_epsilon,
            )

            # Take action by calling env.step
            next_state, reward, terminal, truncated, info = env.step(action)
            return_value += reward

            if train:
                # Add to replay buffer
                self.replay.add(state, action, reward, next_state, terminal)

                # Train the network.
                if self.step_count % self.train_interval == 0 and self.step_count > 200:
                    self.train_network(gamma=self.gamma)

                # Copy weights from the main network to the target network
                if self.step_count % self.copy_interval == 0:
                    self.copy_weights()


                self.step_count += 1

            # Update the state.
            state = next_state

        return return_value
    
    def train_network(self, gamma: float):
        # Get a batch of data from the replay buffer
        states, actions, rewards, new_states, terminal = self.replay.get_batch()
        batch_size = len(states)

        # Calculate the Q-values for the current state
        Q = self.main_network(states)

        # Calculate the Q-values for the next state
        with torch.no_grad():  # Do not track gradients for the target network
            Q_next = self.target_network(new_states).detach()
            Q_next_max = torch.max(Q_next, dim=1).values

        # Calculate the TD-target - Remember to use the terminal states such that
        # the target for terminal states is just the reward.
        # torch.logical_not is useful here.
        td_target = rewards + torch.logical_not(terminal) * gamma * Q_next_max

        # Calculate the loss
        Q = Q[torch.arange(batch_size), actions]
        loss = torch.nn.functional.mse_loss(Q, td_target)

        # Backpropagate the loss
        loss.backward()
        self.main_network.optimizer.step()
        self.main_network.optimizer.zero_grad()

    def learn(
        self,
        env: gym.Env,
        num_episodes=1000,
    ):
        # This wraps out environment to record statistics about the episodes.
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=num_episodes)

        # Reset the replay buffer
        self.replay.clear()
        self.step_count = 0

        for episode in track(
            range(num_episodes), description="Training DQN", transient=True
        ):
            self.rollout(env=env, episode=episode, train=True)

        return env.return_queue, env.length_queue

    def test(self, env: gym.Env, num_episodes=100, apply_epsilon=False):
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=num_episodes)

        for episode in track(
            range(num_episodes), description="Testing DQN", transient=True
        ):
            self.rollout(
                env=env, episode=episode, train=False, apply_epsilon=apply_epsilon
            )

        return np.array(env.return_queue).flatten(), np.array(
            env.length_queue
        ).flatten()

    def copy_weights(self):
        self.target_network.load_state_dict(deepcopy(self.main_network.state_dict()))