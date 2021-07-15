import numpy as np
import gym
import matplotlib.pyplot as plt
import time
import pandas as pd

"""MONTE CARLO"""
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
    episode_list = []

    for _ in range(iterations):
        episode = gen_episode(e)
        episode_list.append(episode)
        G = 0.
        for i, (s_ini, a, r, s_end) in enumerate(episode[::-1]):
            G = gamma*G + r
            if s_ini not in [eps[0] for eps in episode[:len(episode)-i-1]]:
                N[s_ini] += 1
                V_old = V[s_ini]
                V[s_ini] += (1/N[s_ini])*(G - V_old)
                delta[s_ini] = np.abs(V_old - V[s_ini])
        delta_list.append(np.max(delta))

    return V, delta_list, episode_list


"""UTILITY FUNCTIONS"""
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


ITERATIONS = 5000
GAMMA = .95

e = gym.make("Taxi-v3")
t_ini = current_milli_time()
V, delta_list, episode_list = compute_V_MC_inc(e, GAMMA, ITERATIONS)
t_end = current_milli_time()

print("Iterations:", ITERATIONS)
print("Gamma:", GAMMA)
print("Time (s):", (t_end-t_ini)/100.0)
print("Last-100 delta avg:", np.sum(delta_list[-100:])/100)
print("Last-100 no. steps avg:", np.sum([len(x) for x in episode_list[-100:]])/100.0)

# plot learning curve
plt.plot(delta_list)
plt.show()

# save V to a .csv file
pd.DataFrame(V).to_csv('taxi_V.csv')

# test agent
#test_agent(V)
