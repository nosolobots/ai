import gym
from gym import error, spaces, utils
from gym.utils import seeding

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
MAX_EPISODE_STEPS = 200

class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

        self.nrow = self.ncol = 4
        self.nS = self.nrow * self.ncol
        self.nA = 4
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.steps = 0

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for s in range(self.nS):
            for a in range(self.nA):
                self.P[s][a].append(self._getProbabilities(s, a))
        
        self.s = self.reset()

    def _getProbabilities(self, s, a):
        p = 1
        r = -1
        done = False

        # we are in a terminal state (no matter which action we take)
        if(s==0 or s==(self.nrow*self.ncol-1)):
            return (p, s, 0, True)
        
        # compute new state given the selected action
        s_col = ns_col = s%self.ncol
        s_row = ns_row = s//self.ncol
        
        if a==0 and ns_col>0: ns_col -= 1            
        if a==1 and ns_row<(self.nrow - 1): ns_row += 1
        if a==2 and ns_col<(self.ncol - 1): ns_col += 1
        if a==3 and ns_row>0: ns_row -= 1

        ns = ns_row*self.ncol + ns_col

        # if we end up in a terminal state, reward=0 and done=True
        if(ns == 0 or ns == (self.observation_space.n-1)):
            r = 0
            done = True
            
        return (p, ns, r, done)

    def step(self, action):
        p, s, r, done = self.P[self.s][action][0]
        self.s = s
        
        self.steps += 1
        if self.steps == self.max_episode_steps: 
            done = True
        
        return (s, r, done, {"prob": p})

    def reset(self):
        self.steps = 0
        s = 0
        while s==0 or s==(self.observation_space.n-1): 
            s = self.observation_space.sample()
        self.s = s
        return s

    def render(self, close=False):
        # agent position
        a_col = self.s % self.ncol
        a_row = self.s//self.ncol

        # terminal states
        t = (0, self.observation_space.n-1)

        for r in range(self.nrow):
            for c in range(self.ncol):
                if r == a_row and c == a_col:
                    print("O", end="")
                elif (r*self.ncol + c) in t:
                    print("X", end="")
                else:
                    print(" ", end="")
                if c < self.ncol-1: print("╎", end="")
            if r<self.nrow-1: print("\n╌ ╌ ╌ ╌")
        print()
        
