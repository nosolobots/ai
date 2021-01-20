import gym

if __name__ == '__main__':
    _MAX_EPISODES = 20
    mean_reward = 0.

    env = gym.make('CartPole-v0')

    for i in range(_MAX_EPISODES):
        total_reward = 0.

        env.reset()

        done = False
        action = env.action_space.sample()
        while not done:
            env.render()
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            action = int(obs[2]>0)  # Choose action based on stick angle

        mean_reward += total_reward
        print(f"Episode {i+1:>2d} reward: {total_reward}")

    env.close()

    print(f"Mean reward: {mean_reward/_MAX_EPISODES}")

