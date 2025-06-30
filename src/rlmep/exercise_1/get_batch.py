import numpy as np
import torch

def get_batch(Q_table, size=16):
    """
    Helper function to generate batches from the Q_table learned with Q-learning.
    """
    states = np.random.randint(0, 16, size=size)
    actions = np.random.randint(0, 4, size=size)
    targets = Q_table[states, actions]

    # Convert to tensors: 
    states = torch.tensor(states, dtype=torch.float32).reshape(-1, 1)
    actions = torch.tensor(actions, dtype=torch.int64)
    targets = torch.tensor(targets, dtype=torch.float32)    

    return states, actions, targets