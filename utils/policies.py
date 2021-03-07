from template import Policy
import numpy as np
from utils.value_functions import ActionValFunc, StateValFunc


class GreedyPolicy(Policy):
    """

    Greedy Policy class, given a value function Q or V, return a probability distribution

    \pi(a | s) s.t probability of argmax_a val_func(a) = 1, otherwise = 0, for all a, s


    """

    def __init__(self, val, env=None, gamma=None):
        """
        init GreedyPolicy \pi_g

        :param val: the value function
        :param env:
        :param gamma:
        """
        self._val = None
        self.env = env
        self.gamma = gamma
        self.val = val
        self._A = {}

    @property
    def val(self):

        return self._val

    @val.setter
    def val(self, vf):

        if not isinstance(vf, (ActionValFunc, StateValFunc)):
            raise ValueError(f'value function {vf} is not valid, available value functions are Q, V')

        if isinstance(vf, StateValFunc) and (self.env is None or self.gamma is None):
            raise ValueError(f'please make sure you have a env object when val_func is V')

        self._val = vf

    def _get_policy(self, s):

        if isinstance(self.val, ActionValFunc):
            val = self.val[s]

        else:
            p = self.env.env.P
            val = [0] * self.env.action_space.n

            for a in p[s].keys():
                dynamics = p[s][a]

                for prob, s_prime, r, _ in dynamics:
                    val[a] = val[a] + prob * (r + self.gamma * self.val[s_prime])

        actions = np.arange(len(val))
        best_action = np.random.choice(actions[val == np.max(val)])
        self._A[s] = np.where(actions == best_action, 1.0, 0.0)

    def __getitem__(self, s):

        if s not in self._A.keys():
            self._get_policy(s)

        return self._A[s]
