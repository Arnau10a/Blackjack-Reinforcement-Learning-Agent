#***** PLEASE, DO NOT MODIFY ********
from abc import ABC, abstractmethod

from blackjack import BlackjackEnv


class AbstractAgent(ABC):
    """
    Abstract base class for agents in the homework. Provides two fields: env for the environment and number_of_episodes.
    """

    def __init__(self, env: BlackjackEnv, number_of_episodes: int):
        """
        Initializes the agent.
        :param env: The environment in which the agent plays Blackjack.
        :param number_of_episodes: Number of episodes to train on.
        """
        self.env = env
        self.number_of_episodes = number_of_episodes
        super().__init__()

    @abstractmethod
    def train(self):
        """
        This method should train the agent by repeatedly playing the game.
        :return: None.
        """
        pass
