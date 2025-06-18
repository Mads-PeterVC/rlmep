from .base import Agent

class RandomAgent(Agent):

    def __init__(self, action_space):
        """
        Parameters
        ----------
        action_space : gym.spaces.Space
            The action space of the environment.
        """
        self.action_space = action_space

    def get_action(self, state):
        """
        Get a random action from the action space.

        Parameters
        ----------
        state : Any
            The current state of the environment.

        Returns
        -------
        int
            A random action from the action space.
        """
        return self.action_space.sample()
    
    @classmethod
    def from_env(cls, env):
        """
        Create a RandomAgent from the environment.

        Parameters
        ----------
        env : gym.Env
            The environment to create the agent from.

        Returns
        -------
        RandomAgent
            A RandomAgent instance.
        """
        return cls(env.action_space)
