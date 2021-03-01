from abc import ABC, abstractmethod

class Agent(ABC):

    @abstractmethod
    def train(self):
        """

        :return:
        """

        pass

    @abstractmethod
    def evaluate(self, pi):
        """

        :return:
        """

        pass

