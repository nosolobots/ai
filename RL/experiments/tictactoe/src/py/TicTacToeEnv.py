import gym
import os
import json

class TicTacToeEnv(gym.Env):
    """Entorno TicTacToe simple.

    Simula el juego del 3 en raya
    """
    def __init__(self):
        super(TicTacToeEnv, self).__init__()

        # carga la configuración
        self._load_config()

        # inicializa el estado
        self._initialize()
        self.action_space = gym.spaces.Discrete(9)

    def _initialize(self):
        """Inicializa el estado del juego."""
        self.player = 1       # jugador actual
        self.states = 9*[0]   # estados

    def _load_config(self):
        """Carga el archivo de configuración.

        El archivo de configuración es un json se llama 'config.json' y tiene
        el siguiente formato:
            {
              "symbols": ["x", "o"],
              "rewards":
                {
                    "win":  20.0,
                    "draw": 10.0,
                    "lose": -20.0,
                    "in_game": 0.0
                }
            }
        """

        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                'config.json')

        with open(config_file) as f:
            config = json.load(f)
            self.symbols = config["symbols"]
            self.rewards = dict(config['rewards'].items())

    def reset(self):
        self._initialize()
        return tuple(self.states)

    def step(self, action):
        if self.states[action] != 0:
            return tuple(self.states), self.rewards["lose"], True, {"end": "error", "err": "invalid move"}

        # actualiza posición en el tablero
        self.states[action] = self.player

        # check si jugador actual gana
        if self._check_winner():
            return tuple(self.states), self.rewards["win"], True, {"end": "win", "winner": self.player}

        # check si empate
        if self._check_draw():
            return tuple(self.states), self.rewards["draw"], True, {"end": "draw"}

        # siguiente jugador
        self.player = (self.player % 2) + 1

        return tuple(self.states), self.rewards["in_game"], False, {}

    def render(self):
        symbol = [' ', self.symbols[0], self.symbols[1]]
        data = [symbol[i] for i in self.states]
        print(f' {data[0]} | {data[1]} | {data[2]}')
        print('――― ――― ―――')
        print(f' {data[3]} | {data[4]} | {data[5]}')
        print('――― ――― ―――')
        print(f' {data[6]} | {data[7]} | {data[8]}')

    def _check_draw(self):
        """Determina si hay empate en el estado actual."""
        for st in self.states:
            if st == 0:
                return False
        return True

    def _check_winner(self):
        """Determina si hay un tres en raya para algún jugador."""
        # horizontal
        for i in range(0, 9, 3):
            if self.states[i] == self.states[i+1] == self.states[i+2] != 0:
                return True

        # vertical
        for i in range(3):
            if self.states[i] == self.states[i+3] == self.states[i+6] != 0:
                return True

        # diagonal ppal
        if self.states[0] == self.states[4] == self.states[8] != 0:
            return True

        # diagonal sec
        if self.states[2] == self.states[4] == self.states[6] != 0:
            return True

        return False

