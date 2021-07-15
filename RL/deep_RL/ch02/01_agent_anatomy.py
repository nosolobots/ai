import random

class Environment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.steps_left = 10

    def get_observation(self):
        return [0.0, 0.0, 0.0]

    def get_actions(self):
        return [0, 1]

    def is_done(self):
        return self.steps_left == 0

    def action(self, action):
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        return random.random() # reward

class Agent:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_reward = 0.0

    def step(self, env):
        current_obs = env.get_observation()
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward

if __name__ == "__main__":
    env = Environment()
    agent = Agent()

    NUM_EPISODES = 10
    mean_rew = 0.0

    for i in range(1, NUM_EPISODES+1):
        env.reset()
        agent.reset()

        while not env.is_done():
            agent.step(env)

        print(f"Episode {i:>02d} - Total reward got: {agent.total_reward:.4f}")
        mean_rew += agent.total_reward

    print(f"Mean reward: {mean_rew/NUM_EPISODES:.4f}")

