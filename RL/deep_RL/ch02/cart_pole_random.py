import gym

if __name__ == '__main__':
    episodes = 20
    mean_reward = 0.

    env = gym.make('CartPole-v0')

    for i in range(episodes):
        total_reward = 0.

        env.reset()

        done = False
        while not done:
            env.render()
            obs, reward, done, _ = env.step(env.action_space.sample())
            total_reward += reward

        mean_reward += total_reward
        print(f"Episode {i+1:>2d} reward: {total_reward}")

    env.close()

    print(f"Mean reward: {mean_reward/episodes}")

