import gym
import time

_MAX_EPISODES = 1000

env = gym.make('CartPole-v1')
env._max_episode_steps = _MAX_EPISODES

total_reward = 0
total_steps = 0

obs = env.reset()
env.render()

action = 0

input("[RET] to start")

tic = time.perf_counter()
err_acum = 0

last_error = 0
while True:
    error = 0 - 0.1*obs[0] - obs[2]
    new_tic = time.perf_counter()
    err_acum += error*(new_tic - tic)
    tic = new_tic
    next_action = 1.0*error + 1.7*(error - last_error) + 0.25*err_acum
    last_error = error

    if next_action < 0:
        action = 1
    else:
        action = 0

    print("x:", obs[0], "angle:", obs[2], "next_action:", action)
    
    obs, reward, done, _ = env.step(action)

    total_reward += reward
    total_steps += 1

    env.render()

    if done:
        break

print("--> done:", done, "last x:", obs[0], "angle:", obs[2])
print(f'Episode done in {total_steps} steps, total reward {total_reward:.2f} ')

input("[RET] to close")

env.close()

