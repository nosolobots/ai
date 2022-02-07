import pygame
from TicTacToeEnv import TicTacToeEnv


class TicTacToeWinWrapper(TicTacToeEnv):
    """TicTacToeWinWrapper class.

        ver: 0.1
        - initial commit
    """

    _VER = 0.1

    _CELL_W = 100
    _CELL_H = 100

    _DEF_BACK_COL = (255, 255, 255)
    _DEF_GRID_COL = (200, 200, 225)

    def __init__(self, title="Tic Tac Toe", sound=False, render=True):
        super().__init__()
        self._render = render
        self._sound = sound
        self._ncol = 3
        self._nrow = 3
        self._w = self._ncol * self._CELL_W
        self._h = self._nrow * self._CELL_H

        # init pygame window
        pygame.init()
        if render:
            self._alive = True  # window alive
            self._scr = pygame.display.set_mode((self._w, self._h), pygame.DOUBLEBUF, 32)
            pygame.display.set_caption(title)

            # set colors
            self._grid_col = self._DEF_GRID_COL
            self._back_col = self._DEF_BACK_COL

            # init sound:
            if self._sound:
                self.init_sound()

        else:
            self._alive = False

    @property
    def version(self):
        return self._VER

    @property
    def is_alive(self):
        return self._alive

    def draw_grid(self):
        """Draws the board grid."""

        line_width = 10

        # borde
        pygame.draw.lines(
            self._scr,
            self._grid_col,
            True,
            [(0, 0), (0, self._h), (self._w, self._h), (self._h, 0)],
            line_width)

        # columnas
        for col in range(1, self._ncol):
            pygame.draw.line(
                self._scr,
                self._grid_col,
                (col * self._CELL_W, 0), (col * self._CELL_W, self._h),
                line_width)

        # filas
        for row in range(1, self._nrow):
            pygame.draw.line(
                self._scr,
                self._grid_col,
                (0, row * self._CELL_H), (self._w, row * self._CELL_H),
                line_width)

    def init_sound(self):
        pass
        # Load sounds
        # pygame.mixer.music.load(self._SND_MUSIC)
        # self._snd_hit = pygame.mixer.Sound(self._SND_HIT)

        # Start music play
        # pygame.mixer.music.play(-1)

    def close(self):
        """close window and exit."""
        pygame.display.quit()
        pygame.quit()
        self._alive = False
        # exit()

    def run_once(self):
        """executes one iteration of the main display.
        """
        if self._alive:
            self._scr.fill(self._back_col)
            self.draw_grid()

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()


if __name__ == '__main__':
    env = TicTacToeWinWrapper()
    env.reset()
    while env.is_alive:
        env.run_once()
