#!/usr/bin/env python3

import time
from frozen_lake_wrappers import FrozenLakeWinWrapper

if __name__ == '__main__':
    env = FrozenLakeWinWrapper(title="Give me power!!", slippery=False)

    env.set_im_goal('art/batt.png')

    env.hole_color = (25, 25, 25)
    env.grid_color = (220, 220, 255)
    env.home_color = (200, 255, 200)
    env.goal_color = (255, 200, 200)

    env.rewd_wall = -0.1    # reward for hitting a wall
    env.rewd_hole = -0.25   # reward for falling into a hole

    for i in range(3):
        print(f"Episode: {i+1}")

        env.reset()
        env.run_once()

        time.sleep(0.5)

        done = False
        while env.is_alive and not done:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            print(reward)

            env.run_once()
            time.sleep(0.25)

        time.sleep(0.5)

    # Wait for QUIT event
    while env.is_alive:
        env.run_once()


