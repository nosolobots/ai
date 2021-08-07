"""
    Resolución de FrozenLake mediante algoritmo Value Iteration (DP).
    
    Realiza una estimación del modelo (recompensas y probabilidades de transición).
"""

import gym
import numpy as np
from time import sleep

from collections import defaultdict, Counter

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
        
        # inicializa la estructura para la estimación de las recompensas y transiciones
        self.state = self.env.reset()
        self.rewards = defaultdict(float) # diccionario para las recompensas. K=(s,a,s'), V=float
        self.transits = defaultdict(Counter) # diccionario de transiciones. K=(s,a), V=Counter(K=(s'), V=int)
        
    def calc_action_value(self, state, action):
        """
            Aplicamos Bellman para calcular Q(s,a).
            
            Ahora no conocemos la tabla de transiciones P. Debemos usar los datos de las estimaciones
        """
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = .0
        
        for s_next, count in target_counts.items():
            r = self.rewards[(state, action, s_next)]
            p = count/total
            action_value += p*(r + self.GAMMA * self.V[s_next])
            
        return action_value

    def select_action(self, state):
        """Selecciona la acción que maximiza Q(s,a) para el estado actual."""
        return np.argmax([self.calc_action_value(state, a_)
                          for a_ in range(self.env.action_space.n)])

    def value_iteration(self):
        """
            Actualiza el valor de los estados.
            Antes, necesita realizar una serie de iteraciones para obtener información del entorno.            
        """
        self.play_n_random_steps(1000)
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

    def play_n_random_steps(self, n):
        """Realiza n interacciones con el entorno para obtener información del mismo."""
        for _ in range(n):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[self.state, action, new_state] = reward
            self.transits[self.state, action][new_state] += 1
            self.state = self.env.reset() if is_done else new_state
        

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
    writer = SummaryWriter("runs/working_directory/x8E")

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
    arrows = ('←', '↓', '→', '↑')
    ns = env.observation_space.n
    nsq = int(np.sqrt(ns))
    for s in range(ns):
        if s%nsq == 0: print()
        print(arrows[policy[s]], end="")

    # ---------------------------------------------------------------------
    # TEST
    #process_iteration(agent, 1, show=True, pause=.0)



if __name__ == "__main__":
    main()
