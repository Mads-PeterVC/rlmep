from abc import ABC, abstractmethod

class Agent(ABC):

    @abstractmethod
    def get_action(self, state):
        """
        Get the action to take in the given state.
        """
        ...