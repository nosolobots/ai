import numpy as np
from abc import ABC, abstractmethod
from TicTacToeEnv import TicTacToeEnv
from collections import defaultdict
import MinimaxAlgorithm
import pickle

class TicTacToeAgent(ABC):
    def __init__(self, env):
        self._env = env

    @abstractmethod
    def action(self, state=None):
        pass


class TicTacToeRandomAgent(TicTacToeAgent):
    def action(self, state=None):
        if state == None:
            return np.random.randint(0, self._env.action_space.n)

        valid = [i for i, s in enumerate(state) if s == 0]
        return np.random.choice(valid)


class TicTacToeMinimaxAgent(TicTacToeAgent):
    def action(self, state):
        # if first move
        free = 0
        for i in state:
            free += i
        if not free:
            # move to a corner
            return np.random.choice([0, 2, 6, 8])
        tree = MinimaxAlgorithm.Tree(state)
        return tree.get_best_action()


class TicTacToeHumanAgent(TicTacToeAgent):
    def action(self, state=None):
        return int(input("? "))


class TicTacToeQLAgent(TicTacToeAgent):
    def __init__(self, env, Q=None, Q_file=None):
        super().__init__(env)
        self._Q = Q
        self._policy = None

        # load Q values
        if Q_file:
            with open(Q_file, 'rb') as f:
                self._Q = pickle.load(f)

        # set policy
        if self._Q:
            self.set_policy()

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, Q):
        self._Q = Q
        self.set_policy()


    def set_policy(self):
        self._policy = dict((s, np.argmax(a)) for s, a in self._Q.items())


    def action(self, state):
        if state not in self._policy:
            valid = [i for i, s in enumerate(state) if s == 0]
            return np.random.choice(valid)

        return self._policy[state]

