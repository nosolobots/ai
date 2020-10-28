import numpy as np
import gym
from gym import wrappers
import gym_gridworld
import matplotlib.pyplot as plt
import time

ITERATIONS = 10
GAMMA = .9
SLEEP = .5
MAX_STEPS = 50

"""MONTE CARLO
"""
def compute_V_MC_inc(e, gamma, iterations, V_ini=None):
    """Compute state values using Monte Carlo with incremental mean update
    """
    if V_ini is None:
        V = np.zeros(e.nS)
    else:
        V = np.array(V_ini)
    N = np.zeros(e.nS)
    delta = np.zeros(e.nS)
    delta_list = []

    for _ in range(iterations):
        episode = gen_episode(e)
        G = 0.
        for i, (s_ini, a, r, s_end) in enumerate(episode[::-1]):
            G = gamma*G + r
            if s_ini not in [eps[0] for eps in episode[:len(episode)-i-1]]:
                N[s_ini] += 1
                V_old = V[s_ini]
                V[s_ini] += (1/N[s_ini])*(G - V_old)
                delta[s_ini] = np.abs(V_old - V[s_ini])
        delta_list.append(np.max(delta))

    return V, delta_list

"""UTILITY FUNCTIONS
"""
def current_milli_time(): return int(round(time.time() * 1000))


def gen_episode(e):
    """Generate episodes using a random policy.
    """
    episode = []
    s_ini = e.reset()
    done = False
    while not done:
        a = e.action_space.sample()
        s_next, r, done, _ = e.step(a)
        episode.append((s_ini, a, r, s_next))
        s_ini = s_next
    return episode


def print_V(V, nrows, ncols):
    """Print state-values
    """
    print("\nV[s]:")
    for r in range(nrows):
        for c in range(ncols):
            print(f"{V[r*ncols + c]:.2f}\t", end="")
            if ((c+1) % ncols == 0):
                print()


def test_agent(e, V, i):
    def select_action(s):
        # get (action, next_state value) for every action from current state
        # we add some random value to avoid taking always the first action when there are
        # repeated state values
        a_ns_value = [(a, V[n[0][1]] + np.random.random()/1E3)
                      for a, n in e.P[s].items() if n[0][1] != s]
        # get the action with greater next state value (greedy)
        print(a_ns_value)
        a = a_ns_value[np.argmax(a_ns_value, axis=0)[1]][0]
        return a
        """
        a, maxv = a_ns_value[0]
        options = []
        options.append((a, int(maxv)))
        for n_a,n_maxv in a_ns_value[1:]:
            if int(n_maxv) == int(maxv):
                options.append((n_a, int(n_maxv)))
            elif n_maxv > maxv:
                a, maxv = n_a, n_maxv
                options.clear()
                options.append((a, int(maxv)))
                
        print(options)
        return options[int(np.random.random()*len(options))][0]
        """
    def render():
        print("\033[2J\033[H")  # clear + home
        print("EPISODE:", i)
        print("==================")
        e.render()

    s = e.reset()
    done = False
    while not done:
        render()
        print_V(V, 4, 4)
        a = select_action(s)
        s, r, done, _ = e.step(a)
        print("next action:", a)
        time.sleep(SLEEP)
    render()
    print_V(V, 4, 4)
    time.sleep(SLEEP)

if __name__ == "__main__":
    e = gym.make("gridworld-v0")
    e.max_episode_steps = MAX_STEPS
    e = gym.wrappers.Monitor(e, "video_rec", force=True)

    V = np.zeros(e.nS)
    for i in range(ITERATIONS):
        # render agent episode for current state values
        test_agent(e, V, i+1)

        # iterate from the previous state values
        V, delta_list = compute_V_MC_inc(e, GAMMA, ITERATIONS, V)

    e.close()


