from template import Agent
import numpy as np


class DpBase(Agent):
    """
    Base class for dp algorithms, assume environment is given, in other words transition probability
    and reward distribution is given

    """

    def __init__(self, env, threshold=10e-5, init_val=None, gamma=0.5, value_func='V'):

        super().__init__(env)
        self.threshold = threshold
        self.gamma = gamma
        self.init_val = init_val
        self.value_func = value_func

        self._Q = {}
        self._V = {}
        self._action_map = {}

    def train(self):
        pass

    def evaluate(self, pi):
        pass

    def get_q(self, a, s):

        try:
            self._Q[s][a]

        except KeyError:

            raise KeyError(f'state action pair {(s, a)} not available!')

    def get_v(self, s):

        try:
            self._V[s]

        except KeyError:

            raise KeyError(f'state {s} not available !')

    def _init_states(self, s, a=None):
        pass


class ValueIter(DpBase):

    def train(self):
        pass


class PolicyIter(DpBase):

    def train(self):
        pass
