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


class Policy(ABC):
    """
    A base policy class for all policies
    """

    @abstractmethod
    def __getitem__(self, s):
        pass


class ValFunc(ABC):
    """
    A base value function class for all value functions
    """

    def __init__(self, val=None):

        if not val:
            self.val = {}

        else:
            self.val = val.copy()

    def init_state(self, s, value):

        if s not in self.val.keys():
            self.val[s] = value

    def __repr__(self):

        return repr(self.val)

    @abstractmethod
    def __getitem__(self, s):
        pass

    @abstractmethod
    def __setitem__(self, i, v):
        pass
