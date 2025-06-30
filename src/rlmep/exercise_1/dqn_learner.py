from copy import deepcopy
import numpy as np
import torch
from rich.progress import track

import gymnasium as gym

class DQN_learner:

    def __init__(self, env, main_network, target_network, replay):
        self.env = env 
        self.main_network = main_network
        self.target_network = target_network
        self.copy_weights()
        self.replay = replay

    def learn(self, num_episodes=1000, gamma=0.90, train_interval=10, copy_interval=100, 
              epsilon=lambda x: 0.1):

        env = gym.wrappers.RecordEpisodeStatistics(self.env, buffer_length=num_episodes)

        step_count = 0
        for episode in track(range(num_episodes), description="Training DQN", transient=True):

            state, _ = env.reset()
            terminal = False
            truncated = False

            while not terminal and not truncated:

                # Choose action
                # If random number is less than epsilon, then select a random action
                # Else select according to the Q-values. When selecting according to Q-values,
                if np.random.rand() < epsilon(episode):
                    action = env.action_space.sample()
                else:
                    state_tensor = torch.tensor(np.array([state], dtype=np.float32))
                    with torch.no_grad():
                        Q = self.main_network(state_tensor).detach().numpy()
                        action = np.argmax(Q)

                # Take action by calling env.step
                next_state, reward, terminal, truncated, info = env.step(action)
                step_count += 1

                # Add to replay buffer
                self.replay.add(state, action, reward, next_state, terminal)

                # Train the network. 
                if step_count % train_interval == 0 and step_count > 200:
                    self.train_network(gamma=gamma)

                # Copy weights from the main network to the target network
                if step_count % copy_interval == 0:
                    self.copy_weights()        

                # Update the state. 
                state = next_state

        return env.return_queue, env.length_queue

    def train_network(self, gamma=0.90):
        # Get a batch of data from the replay buffer
        states, actions, rewards, new_states, terminal = self.replay.get_batch()
        batch_size = len(states)

        # Calculate the Q-values for the current state
        Q = self.main_network(states)

        # Calculate the Q-values for the next state
        with torch.no_grad():
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

    def test(self, N=100):

        env = gym.wrappers.RecordEpisodeStatistics(self.env, buffer_length=N)

        for episode in track(range(N), description="Testing DQN", transient=True):

            state, _ = env.reset()
            terminal = False
            truncated = False

            while not terminal and not truncated:

                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    Q = self.main_network(state_tensor)
                    action = torch.argmax(Q).item()

                # Take action by calling env.step
                next_state, reward, terminal, truncated, info = env.step(action)

                # Update the state. 
                state = next_state

        return np.array(env.return_queue).flatten(), np.array(env.length_queue).flatten()

    def copy_weights(self):
        self.target_network.load_state_dict(deepcopy(self.main_network.state_dict()))