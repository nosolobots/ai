from TicTacToeEnv import TicTacToeEnv
from TicTacToeAgent import TicTacToeRandomAgent, TicTacToeQLAgent, \
                        TicTacToeMinimaxAgent, TicTacToeHumanAgent
import pickle

with open('Q_train_1_random.dat', 'rb') as f:
    Q = pickle.load(f)

env = TicTacToeEnv()
#a1 = TicTacToeRandomAgent(env)
#a1 = TicTacToeQLAgent(env, Q)
a2 = TicTacToeMinimaxAgent(env)
#a2 = TicTacToeMinimaxAgent(env)
a1 = TicTacToeHumanAgent(env)
#a1 = TicTacToeRandomAgent(env)

num_episodes = 5
win = [0, 0]
draw = 0
err = 0

for i in range(num_episodes):
    s = env.reset()
    env.render()

    while True:
        print("Juega 1...")
        ns, r, done, info = env.step(a1.action(s))
        s = ns
        env.render()
        if not done:
            print("Juega 2...")
            ns, r, done, info = env.step(a2.action(s))
            s = ns
            env.render()
        if done:
            if info["end"] == "error":
                err += 1
            elif info["end"] == "win":
                win[info["winner"] - 1] += 1
            elif info["end"] == "draw":
                draw += 1
            break

print(f"win: {win}")
print(f"draw: {draw}")
print(f"err: {err}")
