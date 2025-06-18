from .base import Agent
import numpy as np


class QLearningAgent(Agent):
    def __init__(
        self,
        state_shape: tuple[int, int],
        action_shape: int,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1,
        Q_init: float = 0.0
    ):
        """ 
        """
        self.q_table = np.ones((state_shape[0], state_shape[1], action_shape)) * Q_init
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def get_action(self, state):
        """
        Get the action to take in the given state.
        """
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            action = np.random.randint(low=0, high=self.q_table.shape[2])
        elif (self.q_table[state[0], state[1], :] == self.q_table[state[0], state[1], 0]).all():
            action = np.random.randint(low=0, high=self.q_table.shape[2])
        else:
            action = np.argmax(self.q_table[state[0], state[1]])

        return action
    
    def update(self, state, action, reward, next_state, terminal):
        """
        Update the Q-table based on the action taken and the reward received.
        """
        if not terminal:
            # best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
            td_target = reward + self.discount_factor * self.q_table[next_state[0], next_state[1]].max()
            td_error = td_target - self.q_table[state[0], state[1], action]
        else:
            td_target = reward
            td_error = td_target - self.q_table[state[0], state[1], action]

        self.q_table[state[0], state[1], action] += self.learning_rate * td_error
