from template import ValFunc
import numpy as np
from typing import Iterable


class ActionValFunc(ValFunc):

    def __getitem__(self, s):

        return self.val[s]

    def __setitem__(self, i, v):

        if not isinstance(v, Iterable):
            print('Try to set value of state s with non-iterable data type')

        else:
            self.val[i] = v

    def get_max(self, s):

        actions = np.arange(len(self.val[s]))
        max_val = np.max(self.val[s])
        max_a = actions[self.val[s] == max_val]

        return max_val, max_a


class StateValFunc(ValFunc):

    def __getitem__(self, s):

        return self.val[s]

    def __setitem__(self, i, v):

        self.val[i] = v

