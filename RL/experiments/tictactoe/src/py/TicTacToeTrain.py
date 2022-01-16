from TicTacToeEnv import TicTacToeEnv
from TicTacToeAgent import TicTacToeRandomAgent, TicTacToeQLAgent
import numpy as np
import pickle
import time

env = TicTacToeEnv()

"""
with open('Q_train_1.dat', 'rb') as f:
    Q = pickle.load(f)
"""

t_ini = time.perf_counter()
Q, results = TicTacToeQLAgent.train_X(env, TicTacToeRandomAgent(env), render=True)
#Q, results = TicTacToeQLAgent.train_X(env, TicTacToeQLAgent(env, Q))
t_end = time.perf_counter()


print(f"training time: {(t_end - t_ini):.2f} sec")

print(f"QL win: {np.sum(results==1)}")
print(f"Rn win: {np.sum(results==2)}")
print(f"draw: {np.sum(results==0)}")
print(f"err: {np.sum(results==-1)}")

print("last 100 >>>> ")
print(f"QL win: {np.sum(results[-100:]==1)}")
print(f"Rn win: {np.sum(results[-100:]==2)}")
print(f"draw: {np.sum(results[-100:]==0)}")
print(f"err: {np.sum(results[-100:]==-1)}")

# save QV
print("Saving Q...")
print(f"Number of states: {len(Q)}")
with open('Q_train_1_random.dat', 'wb') as f:
    pickle.dump(Q, f)
