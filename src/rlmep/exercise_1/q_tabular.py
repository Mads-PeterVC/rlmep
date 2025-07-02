import numpy as np
import gymnasium as gym
from rich.progress import track

def Q_tabular():

    env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)
    Q_table = np.zeros((16, 4), dtype=np.float64)
    # Q_table = np.random.uniform(low=-1, high=1, size=(16, 4))

    # Hyperparameters
    epsilon = 0.5
    learning_rate = 1
    gamma = 0.9
    num_episodes = 250


    for episode in track(range(num_episodes)):
        state, _ = env.reset()
        terminal = False
        truncated = False

        while not terminal and not truncated:
            # Choose action
            # If random number is less than epsilon, then select a random action
            # Else select according to the Q-values. When selecting according to Q-values,
            # if all actions have the same Q-value, then select a random action.
            # This is done as argmax will always select the first index in the case of a tie.
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 4)
            else:
                if (Q_table[state, :] == Q_table[state, 0]).all():
                    action = np.random.randint(0, 4)
                else:
                    action = np.argmax(Q_table[state])

            # Take action by calling env.step
            next_state, reward, terminal, truncated, info = env.step(action)

            # Calculate Q-target
            Q_target = (
                reward + gamma * np.max(Q_table[next_state, :]) - Q_table[state, action]
            )
            # if not terminal:
            #     Q_target = (
            #         reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action]
            #     )
            # else:
            #     Q_target = reward - Q_table[state, action]

            # Update Q-table
            Q_table[state, action] += learning_rate * Q_target

            state = next_state