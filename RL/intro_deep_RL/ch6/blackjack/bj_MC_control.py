import gym
import numpy as np
from collections import defaultdict

NUM_EPISODES = 1000000
ALPHA = 0.02
EPS_DECAY = 0.9999965
GAMMA = 1.0

def get_probs(Q_s, epsilon, nA):
    """Define las probabilidades de las acciones para el estado.

    La acción de mayor valor tendrá probabilidad: 1 - epsilon + (epsilon / nA)
    El resto de acciones tendrán probabilidad: (epsilon / nA)
    """
    policy_s = np.ones(nA) * epsilon / nA
    max_action = np.argmax(Q_s)
    policy_s[max_action] = 1 - epsilon + (epsilon / nA)
    return policy_s

def generate_episode_from_Q(env, Q, epsilon, nA):
    """Genera un nuevo episodio a partir de la función Q actual y una política ε-greedy."""
    episode = []
    state = env.reset()
    while True:
        probs = get_probs(Q[state], epsilon, nA)
        action = np.random.choice(np.arange(nA), p=probs) if state in Q else env.action.sample()
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

def update_Q(env, episode, Q, alpha, gamma):
    # desempaquetamos los datos del episodio
    states, actions, rewards = zip(*episode)

    for i, state in enumerate(states):
        old_Q = Q[state][actions[i]]
        G = sum(r*gamma**k for k,r in enumerate(rewards[i:]))
        Q[state][actions[i]] = old_Q + alpha*(G - old_Q)

    return Q

def MC_control(env, num_episodes, alpha, eps_decay, gamma, log=False):
    EPS_START = 1.0
    EPS_MIN = 0.05

    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))

    epsilon = EPS_START
    for episode in range(1, num_episodes + 1):
        # actualiza el valor de epsilon
        epsilon = max(epsilon*eps_decay, EPS_MIN)
        if episode % 10000 == 0: print(f'Episode {episode:>7} -> epsilon {epsilon}')

        # genera un nuevo episodio
        episode_generated = generate_episode_from_Q(env, Q, epsilon, nA)

        # actualiza la función Q
        Q = update_Q(env, episode_generated, Q, alpha, gamma)

    # actualiza la política
    policy = dict((state, np.argmax(actions)) for state, actions in Q.items())

    return policy, Q

if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    policy, Q = MC_control(env, NUM_EPISODES, ALPHA, EPS_DECAY, GAMMA)
