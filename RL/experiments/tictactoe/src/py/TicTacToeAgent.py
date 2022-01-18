import numpy as np
from abc import ABC, abstractmethod
from TicTacToeEnv import TicTacToeEnv
from collections import defaultdict
import MinimaxAlgorithm

class TicTacToeAgent(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def action(self, state=None):
        pass


class TicTacToeRandomAgent(TicTacToeAgent):
    def action(self, state=None):
        if state == None:
            return np.random.randint(0, self.env.action_space.n)

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
    def __init__(self, env, Q):
        super().__init__(env)
        self.set_policy(Q)

    def set_policy(self, Q):
        self.policy = dict((s, np.argmax(a)) for s, a in Q.items())

    def action(self, state):
        if state not in self.policy:
            valid = [i for i, s in enumerate(state) if s == 0]
            return np.random.choice(valid)

        return self.policy[state]

    @staticmethod
    def train_X(env, opponent, num_episodes=100000000, alpha=0.02, eps_decay=0.9999965,
                gamma=1.0, log=False, render=False):
        EPS_START = 1.0
        EPS_MIN = 0.05

        # stats
        results = np.zeros(num_episodes, dtype=int)

        # ---------------------------------------------------------------------
        # QL Algorithm
        # ---------------------------------------------------------------------

        # Initialize Q(s)
        nA = env.action_space.n
        #Q = defaultdict(lambda: np.zeros(nA))
        Q = defaultdict(lambda: np.random.uniform(low=-1.0,high=1.0,size=nA)/1e-3)

        epsilon = EPS_START

        # for each episode
        for episode in range(num_episodes):
            if render and episode%25000 == 0:
                print(f"episode: {episode}... {len(Q)} states")

            # update epsilon with epsilon-decay
            epsilon = max(epsilon * eps_decay, EPS_MIN)

            # start episode
            state = env.reset()

            # process episode
            while True:
                # player 1 - get action
                if state in Q:
                    probs = TicTacToeQLAgent.get_probs(Q[state], epsilon, nA)
                    action = np.random.choice(np.arange(nA), p=probs)
                else:
                    action = np.random.randint(nA)

                # execute action
                next_state, reward, done, info = env.step(action)

                # update Q
                Q[state][action] += alpha * (reward +
                                             gamma * np.max(Q[next_state]) - Q[state][action])

                # update state
                state = next_state

                if not done:
                    # player 2
                    next_state, reward, done, info = env.step(opponent.action(state))
                    state = next_state

                if done:
                    if info["end"] == "error":
                        results[episode] = -1
                    elif info["end"] == "win":
                        results[episode] = info["winner"]
                    elif info["end"] == "draw":
                        pass  # =0
                    break

        return dict(Q), results

    @staticmethod
    def get_probs(Q_s, epsilon, nA):
        # iniciamos todas las probabilidades a epsilon/nA
        policy_s = np.ones(nA) * epsilon / nA
        # buscamos la acción de más valor
        max_action = np.argmax(Q_s)
        # establecemos su probabilidad
        policy_s[max_action] = 1 - epsilon + (epsilon / nA)
        # retornamos las probabilidades de cada acción del estado
        return policy_s
