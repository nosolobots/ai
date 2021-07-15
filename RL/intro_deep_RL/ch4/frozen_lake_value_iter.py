"""Resolución de FrozenLake mediante algoritmo Value Iteration (DP)."""

import gym
import numpy as np
from time import sleep

from torch.utils.tensorboard import SummaryWriter

REWARD_THRESHOLD = 0.85
TEST_EPISODES = 40
GAMMA = 0.95

class Agent():
    def __init__(self, env, gamma=0.95):
        self.GAMMA = gamma
        self.env = env
        
        # inicializamos el Valor de los estados
        self.V = np.zeros(env.observation_space.n) 
        
    def calc_action_value(self, state, action):
        """Aplicamos Bellman para calcular Q(s,a)."""
        return sum([p*(r + self.GAMMA*self.V[s_]) 
                    for p, s_, r, _ in self.env.P[state][action]])

    def select_action(self, state):
        """Selecciona la acción que maximiza Q(s,a) para el estado actual."""
        return np.argmax([self.calc_action_value(state, a_) 
                          for a_ in range(self.env.action_space.n)])

    def value_iteration(self):
        """Actualiza el valor de los estados."""
        for s_ in range(self.env.observation_space.n):
            self.V[s_] = max([self.calc_action_value(s_, a_)
                             for a_ in range(self.env.action_space.n)])
            
    def get_policy(self):
        """Devuelve la política que maximiza el valor de Q(s,a)."""
        policy = np.zeros(self.env.observation_space.n, dtype=int)
        for s in range(self.env.observation_space.n):
            policy[s] = np.argmax([self.calc_action_value(s, a_)
                                   for a_ in range(self.env.action_space.n)])
        return policy

def process_iteration(agent, n, show=False, pause=.0):
    """Lanza una iteración de N episodios y devuelve la recompesa media."""
    it_rwd = 0
    for i in range(n):
        eps_rwd = 0.0
        
        state = agent.env.reset()
        if show: 
            agent.env.render()
            sleep(pause)
        
        is_done = False
        while not is_done:
            action = agent.select_action(state)
            new_state, rwd, is_done, _ = agent.env.step(action)
            eps_rwd += rwd
            state = new_state
            if show: 
                agent.env.render()
                sleep(pause)
        
        it_rwd += eps_rwd
        
    return it_rwd/n
        
def main():
    # entorno
    env = gym.make("FrozenLake-v0", map_name="8x8") 
    
    # agente
    agent = Agent(env, gamma=GAMMA)
    
    # ---------------------------------------------------------------------
    # ENTRENAMIENTO

    # Tensorbaord record file init
    writer = SummaryWriter("runs/working_directory")
    
    it = 0
    best_rwd = 0.0
    while best_rwd < REWARD_THRESHOLD:
        agent.value_iteration() # actualiza el valor de los estados
        
        it += 1 # iteración
        
        # lanza una iteración de N episodios y obtiene la recompensa media
        it_rwd = process_iteration(agent, TEST_EPISODES)
        
        # guardamos la recompensa media para Tensorboard
        writer.add_scalar("reward", it_rwd, it)
        
        if it_rwd > best_rwd:
            print(f"Best reward updated {it_rwd:.2f} at iteration {it}")
            best_rwd = it_rwd

    # Tensorboard file close
    writer.close()            
    
    # ---------------------------------------------------------------------
    # POLITICA DEL AGENTE
    policy = agent.get_policy()
    print(policy)
    arrows = ('←', '↓', '→', '↑')
    ns = env.observation_space.n
    nsq = int(np.sqrt(ns))
    for s in range(ns):
        if s%nsq == 0: print()
        print(arrows[policy[s]], end="")
            
    # ---------------------------------------------------------------------
    # TEST
    process_iteration(agent, 1, show=True, pause=.0)
    
    
    
if __name__ == "__main__":
    main()