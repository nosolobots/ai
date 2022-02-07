from TicTacToeEnv import TicTacToeEnv
from TicTacToeAgent import TicTacToeAgent, TicTacToeRandomAgent, \
        TicTacToeQLAgent, TicTacToeMinimaxAgent
import numpy as np
from collections import defaultdict
import pickle
import time
import random
import matplotlib.pyplot as plt

def train(env, opponent, num_episodes=1000, alpha=0.02, eps_decay=0.9999965,
            gamma=1.0, log=False, render=False, Q_ini=None):
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
    if Q_ini:
        Q = Q_ini
    else:
        #Q = defaultdict(lambda: np.zeros(nA))
        Q = defaultdict(lambda: np.random.uniform(low=-1.0,high=1.0,size=nA)/1e6)

    epsilon = EPS_START

    players = ["QL", opponent]

    total_stats_X = []
    total_stats_O = []

    # for each episode
    for episode in range(num_episodes):
        # update epsilon with epsilon-decay
        epsilon = max(epsilon * eps_decay, EPS_MIN)

        # start episode
        state = env.reset()

        player_id = 1

        # process episode
        while True:
            # select player
            player = players[player_id - 1]

            if isinstance(player, TicTacToeAgent):
                next_state, reward, done, info = env.step(opponent.action(state))
                #reward *= -1

            else:
                """
                # learn from errors
                if state in Q:
                    probs = get_probs(Q[state], epsilon, nA)
                    action = np.random.choice(np.arange(nA), p=probs)
                else:
                    action = np.random.randint(nA)
                """
                # not errors allowed
                valid_actions = [i for i,s in enumerate(state) if s == 0]
                if state in Q:
                    probs = get_probs([Q[state][v] for v in valid_actions],
                                epsilon, len(valid_actions))
                    action = np.random.choice(valid_actions, p=probs)
                else:
                    action = np.random.choice(valid_actions)

                # execute action
                next_state, reward, done, info = env.step(action)

                # update Q
                Q[state][action] = (1 - alpha) * Q[state][action] + \
                                        alpha * (reward + gamma * np.max(Q[next_state]))

            """
            if done and info["end"]=="error":
                print("error --->",state,action,next_state)
            """

            # update state
            state = next_state

            if done:
                break

            # next player
            player_id = player_id % 2 + 1

        # episode finsihed
        if episode % 25 == 0:
            stats_X, stats_O = test_agent(dict(Q))
            total_stats_X.append(stats_X)
            total_stats_O.append(stats_O)
            print(f"{episode//25:05d} - X: {stats_X} - O: {stats_O}")

        # rotate players
        players.reverse()

    return dict(Q), total_stats_X, total_stats_O



def get_probs(Q_s, epsilon, nA):
    # iniciamos todas las probabilidades a epsilon/nA
    policy_s = np.ones(nA) * epsilon / nA
    # buscamos la acción de más valor
    max_action = np.argmax(Q_s)
    # establecemos su probabilidad
    policy_s[max_action] = 1 - epsilon + (epsilon / nA)
    # retornamos las probabilidades de cada acción del estado
    return policy_s


def test_agent(Q, num_eps=100):
    """Test agent num_eps as X and num_eps as O."""

    env = TicTacToeEnv()
    agent = TicTacToeQLAgent(env, Q)
    opo = TicTacToeRandomAgent(env)

    stats_X = np.zeros(4) # win, lose, draw, err

    # agent opening
    players = [agent, opo]
    for i in range(num_eps):
        info = run_episode(env, players)
        if info["end"] == "win":
            if info["winner"] == 1:
                stats_X[0] += 1
            else:
                stats_X[1] += 1
        elif info["end"] == "draw":
            stats_X[2] += 1
        elif info["end"] == "error":
            stats_X[3] += 1

    # opo opening
    players.reverse()

    stats_O = np.zeros(4)

    for i in range(num_eps):
        info = run_episode(env, players)
        if info["end"] == "win":
            if info["winner"] == 2:
                stats_O[0] += 1
            else:
                stats_O[1] += 1
        elif info["end"] == "draw":
            stats_O[2] += 1
        elif info["end"] == "error":
            stats_O[3] += 1

    return stats_X, stats_O

def run_episode(env, players):
    state = env.reset()

    player_id = 1

    done = False
    while not done:
        next_state, rew, done, info = env.step(players[player_id-1].action(state))
        state = next_state
        player_id = player_id % 2 + 1

    return info


def show_stats(stats_X, stats_O):
    x = np.arange(len(stats_X))

    np_X = np.array(stats_X)
    np_O = np.array(stats_O)

    plt.subplot(2, 1, 1)
    plt.title("Agent as X")
    plt.xlabel("iteration")
    plt.ylabel("result")
    plt.plot(x,np_X[:,0],label="win")
    plt.plot(x,np_X[:,1],label="lose")
    plt.plot(x,np_X[:,2],label="draw")
    plt.plot(x,np_X[:,3],label="err")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("Agent as O")
    plt.xlabel("iteration")
    plt.ylabel("result")
    plt.plot(x,np_O[:,0],label="win")
    plt.plot(x,np_O[:,1],label="lose")
    plt.plot(x,np_O[:,2],label="draw")
    plt.plot(x,np_O[:,3],label="err")
    plt.legend()

    plt.show()


def main():
    env = TicTacToeEnv()
    opo = TicTacToeRandomAgent(env)
    #opo = TicTacToeMinimaxAgent(env)


    if input("create QL agent from Q_train.dat? ").upper() in ('Y', 'YES', 'S', 'SI'):
        with open('Q_train.dat', 'rb') as f:
            q = pickle.load(f)
            opo = TicTacToeQLAgent(env, Q=q)

    t_ini = time.perf_counter()

    Q, stats_X, stats_O = train(env, opo, num_episodes=50000, gamma=.25)

    t_end = time.perf_counter()

    print(f"Total training time: {(t_end - t_ini):.2f} sec")

    # save QV
    print("Saving Q...")
    print(f"Number of states: {len(Q)}")
    with open('Q_train_out.dat', 'wb') as f:
        pickle.dump(Q, f)

    show_stats(stats_X, stats_O)

    """

    print(f"QL win: {np.sum(results==1)}")
    print(f"Rn win: {np.sum(results==2)}")
    print(f"draw: {np.sum(results==0)}")
    print(f"err: {np.sum(results==-1)}")

    print("last 100 >>>> ")
    print(f"QL win: {np.sum(results[-100:]==1)}")
    print(f"Rn win: {np.sum(results[-100:]==2)}")
    print(f"draw: {np.sum(results[-100:]==0)}")
    print(f"err: {np.sum(results[-100:]==-1)}")

    """

if __name__ == '__main__':
    main()
