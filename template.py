from abc import ABC, abstractmethod


class Agent(ABC):
    """
    The abstract base class for all RL algorithms.

    """

    def __init__(self, env):
        """
        Signature for initialization

        :param env: gym env object
        """
        self.env = env

    @abstractmethod
    def train(self):
        """
        Signature for training

        :return: Optimal policy
        """

        pass

    @abstractmethod
    def evaluate(self, pi):
        """
        Signature for evaluation

        :param pi: input policy for evaluation
        :return: Value function
        """

        pass

