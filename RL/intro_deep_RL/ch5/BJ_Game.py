# -*- coding: utf-8 -*-
"""
Blackjack Player
"""

import gym

class BJConsoleHumanAgent():
    def __init__(self, env):
        self._env = env
        self._state = None
        
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, state):
        self._state = state
        
    def action(self):
        return int(input("action [stick=0 | hit=1]? "))

def header():
    print()
    print("====================")
    print("=   BLACKJACK-v0   =")
    print("====================")
    print()    

def main():
    header()    
    
    env = gym.make("Blackjack-v0")
    agent = BJConsoleHumanAgent(env)
    
    while True:
        print("New game...")
        
        agent.state = env.reset()
        
        done = False
        while not done:
            print("Current state:", agent.state)
            action = agent.action()
            new_state, reward, done, _ = env.step(action)
            agent.state = new_state
            
        print("\nGame done...")
        print("Finale state:", agent.state)
        if reward == 1:
            print("You win")
        elif reward == -1:
            print("You lose")
        else:
            print("Draw")
        
        new = str()
        while new not in('Y','N'):
            new = input("\nNew run [Y/N]? ").upper()
        if new == 'N':
            break
    
if __name__ == '__main__':
    main()


