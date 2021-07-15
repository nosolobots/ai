import gym
from gym import error, logger

env = gym.make('FrozenLake-v0')
logger.set_level(0)
#env = gym.wrappers.Monitor(env, './video_rec', video_callable=lambda i: True, force=True)
env = gym.wrappers.Monitor(env, 'video_rec', force=True)
#env = gym.wrappers.Monitor(env, './video_rec', video_callable=False, force=True)
env.reset()
while True:
	#env.render(mode='ansi')
	env.render(mode='human')
	obs, r, done, info = env.step(env.action_space.sample())
	if done: break
env.close()
