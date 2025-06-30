import torch
import numpy as np
from copy import deepcopy

class ExperienceReplay: 
    """
    Simple experience replay buffer for storing relevant properties for a DQN search.
    """
    def __init__(self, observation_dim, size=100, batch_size=16):
        self.batch_size = batch_size
        self.obs_dim = observation_dim
        self.size = size
        self.clear()

    def add(self, state, action, reward, new_state, terminal):
        self.states[self.index] = torch.tensor(np.array(state))
        self.actions[self.index] = torch.tensor(action)
        self.rewards[self.index] = torch.tensor(reward)
        self.new_states[self.index] = torch.tensor(np.array(new_state))
        self.terminal[self.index] = torch.tensor(terminal)

        self.index += 1
        if self.index == self.size:
            self.index = 0

        if self.current_size < self.size:
            self.current_size += 1

    def get_batch(self):
        """
        Returns        
        states, actions, rewards, next_states, terminal
        """
        indices = np.random.randint(0, self.current_size, size=self.batch_size)
        return self.states[indices], self.actions[indices], self.rewards[indices], self.new_states[indices], self.terminal[indices]
    
    def clear(self):
        self.states = torch.tensor(np.zeros((self.size, self.obs_dim)), dtype=torch.float32) # YOUR CODE HERE
        self.new_states = torch.tensor(np.zeros((self.size, self.obs_dim)), dtype=torch.float32) # YOUR CODE HERE
        self.rewards = torch.tensor(np.zeros(self.size), dtype=torch.float32)
        self.actions = torch.tensor(np.zeros(self.size), dtype=torch.int32)
        self.terminal = torch.tensor(np.zeros(self.size), dtype=torch.bool)
        self.current_size = 0
        self.index = 0

def copy_weights(source_network, target_network):
    """
    """
    source_network.load_state_dict(deepcopy(target_network.state_dict()))