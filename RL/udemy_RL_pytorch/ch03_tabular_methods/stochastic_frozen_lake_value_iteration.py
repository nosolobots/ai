import time
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt

EPISODES = 1000
NUM_STATES = 16
NUM_ACTIONS = 4
GAMMA = .99     # discount factor

def clear_scr():
    print(chr(0x1B) + "[2J")
    print(chr(0x1B) + "[H")

def value_iteration(env):
    """Build the States Value table from scratch.

    Evals every possible action for each state
    """
    Qvalues = torch.zeros(NUM_STATES)

    max_iterations = 1500

    for _ in range(max_iterations):
        # for each state we search for best move
        for st in range(NUM_STATES):
            max_value, _ = next_step_evaluation(env, st, Qvalues)
            Qvalues[st] = max_value.item()

    return Qvalues

def next_step_evaluation(env, state, Vvalues):
    """Returns the best possible move or a given state.
    """
    Vtemp = torch.zeros(NUM_ACTIONS)

    for action in range(NUM_ACTIONS):
        for prob, new_state, reward, _ in env.env.P[state][action]:
            Vtemp[action] += (prob * (reward + GAMMA*Vvalues[new_state]))

    max_value, indx = torch.max(Vtemp, 0)

    return max_value, indx

def build_policy(env, Vvalues):
    """Once we have the state-values table, we can build a policy to take the best action.
    """
    Vpolicy = torch.zeros(NUM_STATES)

    for state in range(NUM_STATES):
        _, index = next_step_evaluation(env, state, Vvalues)
        Vpolicy[state] = index.item()

    return Vpolicy


total_steps = []
rewards = []

env = gym.make("FrozenLake-v0")

# Build policy
V = value_iteration(env)
Vpolicy = build_policy(env, V)
print(Vpolicy)

for n in range(EPISODES):
    state = env.reset()
    steps = 0

    while True:
        # select the action
        action = Vpolicy[state]

        # take action
        new_state, reward, done, info  = env.step(int(action))
        steps += 1

        # update state
        state = new_state

        if done:
            total_steps.append(steps)
            rewards.append(reward)
            break

print(f"Average of episodes solved: {sum(rewards)/EPISODES}")
print(f"Average of last-100 episodes solved: {sum(rewards[-100:])/100}")
print(f"Average of steps taken: {sum(total_steps)/EPISODES}" )
print(f"Average of last-100 episodes steps taken: {sum(total_steps[-100:])/100}")

plt.style.use('ggplot')

plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(np.arange(len(rewards)), rewards, alpha=0.6, color='green')
plt.show()

plt.figure(figsize=(12, 5))
plt.title("Steps in episode")
plt.plot(total_steps, alpha=0.6, color='red')
plt.show()
