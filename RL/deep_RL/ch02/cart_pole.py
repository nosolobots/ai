import gym

if __name__ == '__main__':
    total_reward = 0.

    env = gym.make('CartPole-v0')
    env.reset()

    try:
        while True:
            env.render()
            obs, reward, done, _ = env.step(env.action_space.sample())
            total_reward += reward

    except KeyboardInterrupt:
        env.close()
        print("\nTotal reward:", total_reward)

