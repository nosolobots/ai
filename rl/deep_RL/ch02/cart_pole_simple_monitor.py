import gym

if __name__ == '__main__':
    _MAX_EPISODES = 10
    mean_reward = 0.

    env = gym.make('CartPole-v0')

    # Monitor wrapper
    # lambada video_callable (Optional[function, False]) :
    # function that takes in the index of the episode and outputs a boolean, indicating whether we 
    # should record a video on this episode. The default (for video_callable is None) is to take 
    # perfect cubes, capped at 1000. False disables video recording. 
    # So we need to define video_callable because it is False by default. 
    env = gym.wrappers.Monitor(env, "rec_cart_pole_simple", video_callable=lambda episode_id: True, force=True)

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

