import gym
from matplotlib import pyplot as plt

if __name__ == '__main__':
    _MAX_EPISODES = 20
    episode_steps = []
    episode_reward = []

    env = gym.make('CartPole-v0')

    for i in range(_MAX_EPISODES):
        total_reward = 0.
        steps = 0

        env.reset()

        done = False
        action = env.action_space.sample()
        while not done:
            # env.render()
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            action = int(obs[2]>0)  # Choose action based on stick angle

        episode_reward.append(total_reward)
        episode_steps.append(steps)
        print(f"Episode {i+1:>2d} reward: {total_reward}")

    env.close()

    print(f"Average reward: {sum(episode_reward)/_MAX_EPISODES}")
    print(f"Average steps: {sum(episode_reward)/_MAX_EPISODES}")

    plt.plot(episode_steps)
    plt.show()

