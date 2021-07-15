"""
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich
    Description:
    There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.
    Observations: 
    There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations. 
    
    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi
    
    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
        
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
    
    Rewards: 
    There is a default per-step reward of -1,
    except for delivering the passenger, which is +20,
    or executing "pickup" and "drop-off" actions illegally, which is -10.
    
    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations
    
    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
"""
import torch
import numpy as np
import gym
import time
import matplotlib.pyplot as plt
import pandas as pd 

clear_scr = lambda: print("\033[2J\033[H")

EPISODES = 1000
NUM_STATES = 500
NUM_ACTIONS = 6
GAMMA = .95         # discount factor
ALPHA = .5         # learning rate

q_values = torch.zeros((NUM_STATES, NUM_ACTIONS))

total_steps = []
rewards = []
episodes_solved = []

env = gym.make("Taxi-v3")

for n in range(EPISODES):
    clear_scr()
    print("Episode:", n)

    state = env.reset()
    steps = 0
    episode_reward = 0

    done = False
    while not done:
        #clear_scr()
        #env.render()

        # select the action with greater Q value
        actions = q_values[state] + torch.randn(1, NUM_ACTIONS)/1E4
        action = torch.max(actions, 1)[1][0].item()

        # take action
        new_state, reward, done, info = env.step(action)
        steps += 1
        episode_reward += reward

        # update Q values
        q_values[state, action] = (1 - ALPHA)*q_values[state, action] + \
            ALPHA*(reward + GAMMA*torch.max(q_values[new_state]))

        state = new_state

    total_steps.append(steps)
    rewards.append(episode_reward)
    episodes_solved.append(int(reward == 20))


#print(q_values)
#pd.DataFrame(q_values).to_csv('taxi_q_values.csv')

print("TOTALS:")
print(f"Episodes solved: {sum(episodes_solved)/EPISODES}")
print(f"Steps taken: {sum(total_steps)/EPISODES}")
print(f"Reward: {sum(rewards)/EPISODES}")
print("LAST-100:")
print(f"Episodes solved: {sum(episodes_solved[-100:])/100}")
print(f"Steps taken: {sum(total_steps[-100:])/100}")
print(f"Reward: {sum(total_steps[-100:])/100}")

plt.style.use('ggplot')

plt.figure(figsize=(12, 5))
plt.title("Rewards")
plt.bar(np.arange(len(rewards)), rewards, alpha=0.6, color='green')
plt.show()

plt.figure(figsize=(12, 5))
plt.title("Steps in episode")
plt.plot(total_steps, alpha=0.6, color='red')
plt.show()

# Play an episode
for _ in range(5):
    clear_scr()
    print("Starting new episode...")
    time.sleep(1)

    state = env.reset()
    done = False
    reward = 0
    total_reward = 0
    total_steps = 0

    while not done:
        clear_scr()

        env.render()

        print(f'reward: {reward}')
        print(f'total reward: {total_reward}')
        print(f'total steps: {total_steps}')

        time.sleep(0.5)

        # take action
        state_actions = q_values[state] + torch.randn(1, NUM_ACTIONS)/1E4
        action = torch.max(state_actions, 1)[1][0].item()
        new_state, reward, done, info = env.step(action)

        state = new_state

        total_reward += reward
        total_steps += 1

    clear_scr()

    env.render()

    print(f'reward: {reward}')
    print(f'total reward: {total_reward}')
    print(f'total steps: {total_steps}')
    print("DONE!!!!")
    time.sleep(2)
