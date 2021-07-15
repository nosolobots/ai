import torch
import numpy as np
import gym
import time
import matplotlib.pyplot as plt 

def clear_scr():
    print(chr(0x1B) + "[2J")
    print(chr(0x1B) + "[H")

EPISODES = 1000
EPS = 0.7
EPS_FINAL = .1
EPS_DECAY = .999

num_states = 16
num_actions = 4
gamma = .9

q_values = torch.zeros((num_states, num_actions))

total_steps = []
rewards = []

#env = gym.make("FrozenLake-v0")
env = gym.make("FrozenLake-v0", is_slippery=False)

for n in range(EPISODES):
    state = env.reset()
    steps = 0

    while True:
        if(np.random.random() < EPS):
            # random action (exploration)
            action = env.action_space.sample()
        else:
            # select the action with greater Q value
            actions = q_values[state] + torch.randn(1, num_actions)/1E3
            action = torch.max(actions, 1)[1][0].item()

        # take action
        new_state, reward, done, info  = env.step(action)
        steps += 1

        # update Q values
        q_values[state, action] = reward + gamma*torch.max(q_values[new_state])

        # update state
        state = new_state

        # update greedy epsilon
        if EPS>EPS_FINAL:
            EPS *= EPS_DECAY

        #clear_scr()
        #env.render()
        #print(steps)
        #time.sleep(1)

        if done: 
            total_steps.append(steps)
            rewards.append(reward)
            break

print(q_values)

print(f"Average of episodes solved: {sum(rewards)/EPISODES}")
print(f"Average of last-100 episodes solved: {sum(rewards[-100:])/100}")
print(f"Average of steps taken: {sum(total_steps)/EPISODES}")
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
