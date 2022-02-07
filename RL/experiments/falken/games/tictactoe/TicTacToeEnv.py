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
        self._player = 1       # jugador actual
        self._states = 9*[0]   # estados

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
            self._symbols = config["symbols"]
            self._rewards = dict(config['rewards'].items())

    def reset(self):
        self._initialize()
        return tuple(self._states)

    def step(self, action):
        if self._states[action] != 0:
            return tuple(self._states), self._rewards["lose"], True, {"end": "error", "err": "invalid move"}

        # actualiza posición en el tablero
        self._states[action] = self._player

        # check si jugador actual gana
        if self._check_winner():
            return tuple(self._states), self._rewards["win"], True, {"end":
                    "win", "winner": self._player}

        # check si empate
        if self._check_draw():
            return tuple(self._states), self._rewards["draw"], True, {"end": "draw"}

        # siguiente jugador
        self._player = (self._player % 2) + 1

        return tuple(self._states), self._rewards["in_game"], False, {"next player":str(self._player)}

    def render(self):
        symbol = [' ', self._symbols[0], self._symbols[1]]
        data = [symbol[i] for i in self._states]
        print(f' {data[0]} | {data[1]} | {data[2]}')
        print('――― ――― ―――')
        print(f' {data[3]} | {data[4]} | {data[5]}')
        print('――― ――― ―――')
        print(f' {data[6]} | {data[7]} | {data[8]}')

    def _check_draw(self):
        """Determina si hay empate en el estado actual."""
        for st in self._states:
            if st == 0:
                return False
        return True

    def _check_winner(self):
        """Determina si hay un tres en raya para algún jugador."""
        # horizontal
        for i in range(0, 9, 3):
            if self._states[i] == self._states[i+1] == self._states[i+2] != 0:
                return True

        # vertical
        for i in range(3):
            if self._states[i] == self._states[i+3] == self._states[i+6] != 0:
                return True

        # diagonal ppal
        if self._states[0] == self._states[4] == self._states[8] != 0:
            return True

        # diagonal sec
        if self._states[2] == self._states[4] == self._states[6] != 0:
            return True

        return False

