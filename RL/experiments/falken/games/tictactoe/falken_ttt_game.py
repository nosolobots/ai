# -*- coding: utf-8 -*-
"""Juego de Tic-Tac-Toe para falken."""

import os
import pygame
from TicTacToeEnv import TicTacToeEnv
from TicTacToeAgent import TicTacToeRandomAgent, TicTacToeMinimaxAgent, \
            TicTacToeQLAgent

class TTT_Game():
    _GRID_LINE_WIDTH = 20
    _MARK_X_LINE_WIDTH = 40
    _MARK_O_LINE_WIDTH = 30
    _CELL_PADDING = 50

    _PLAYER_HUMAN = 1
    _PLAYER_RANDOM = 2
    _PLAYER_MINIMAX = 3
    _PLAYER_QL = 4

    _Q_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Q.dat")

    def __init__(self, console, player_types, sleep=250):
        self._console = console
        self._sleep = sleep
        self._color = self._console.ink
        self._gym_env = TicTacToeEnv()
        self._state = self._gym_env.reset()
        self._players = self.create_players(player_types)
        self._player = 1
        self._done = False
        self._info = {}

        # dimensions
        self._scr_width = console.canvas.scr.get_width()
        self._scr_height = console.canvas.scr.get_height()
        self._center_x = self._scr_width // 2
        self._center_y = self._scr_height // 2
        self._board_width = min(self._scr_width, self._scr_height)
        self._cell_size = self._board_width // 4

        self.draw_board()

    @property
    def done(self):
        return self._done

    @property
    def info(self):
        return self._info

    def create_players(self, player_types):
        players = {}
        for player_id, player_type in enumerate(player_types):
            if player_type == TTT_Game._PLAYER_HUMAN:
                players[player_id + 1] = TTT_HumanAgent(self._console)
            elif player_type == TTT_Game._PLAYER_RANDOM:
                players[player_id + 1] = TicTacToeRandomAgent(self._gym_env)
            elif player_type == TTT_Game._PLAYER_MINIMAX:
                players[player_id + 1] = TicTacToeMinimaxAgent(self._gym_env)
            elif player_type == TTT_Game._PLAYER_QL:
                players[player_id + 1] = TicTacToeQLAgent(self._gym_env,
                        Q_file= TTT_Game._Q_FILE)
        return players

    def spin_once(self):
        # self.draw_board()
        self.make_move()
        self.draw_board()
        self._console.make_beep()
        self._player = self._player % 2 + 1
        pygame.time.wait(self._sleep)

    def draw_board(self):
        self._console.clear()
        self.draw_grid()
        self.draw_marks()
        pygame.display.update()

    def make_move(self):
        action = self._players[self._player].action(self._state)
        self._state, r, self._done, self._info = self._gym_env.step(action)

    def draw_grid(self):
        cell_half_size = self._cell_size // 2

        # vertical lines
        for x in (self._center_x - cell_half_size,
                  self._center_x + cell_half_size):
            pygame.draw.line(
                self._console.canvas.scr,
                self._color,
                (x, self._center_y - 3 * cell_half_size),
                (x, self._center_y + 3 * cell_half_size),
                TTT_Game._GRID_LINE_WIDTH)

        # horizontal lines
        for y in (self._center_y - cell_half_size,
                  self._center_y + cell_half_size):
            pygame.draw.line(
                self._console.canvas.scr,
                self._color,
                (self._center_x - 3 * cell_half_size, y),
                (self._center_x + 3 * cell_half_size, y),
                TTT_Game._GRID_LINE_WIDTH)

    def draw_marks(self):
        for i, mark in enumerate(self._state):
            row = i // 3
            col = i % 3
            x_pos = self._center_x - self._cell_size
            y_pos = self._center_y - self._cell_size
            if mark == 1:
                self.draw_X(x_pos + col*self._cell_size,
                            y_pos + row*self._cell_size)
            elif mark == 2:
                self.draw_O(x_pos + col*self._cell_size,
                            y_pos + row*self._cell_size)

    def draw_X(self, x, y):
        cell_half_size = self._cell_size // 2

        pygame.draw.line(
                self._console.canvas.scr,
                self._color,
                (x - cell_half_size + TTT_Game._CELL_PADDING,
                    y - cell_half_size + TTT_Game._CELL_PADDING),
                (x + cell_half_size - TTT_Game._CELL_PADDING,
                    y + cell_half_size - TTT_Game._CELL_PADDING),
                TTT_Game._MARK_X_LINE_WIDTH)
        pygame.draw.line(
                self._console.canvas.scr,
                self._color,
                (x - cell_half_size + TTT_Game._CELL_PADDING,
                    y + cell_half_size - TTT_Game._CELL_PADDING),
                (x + cell_half_size - TTT_Game._CELL_PADDING,
                    y - cell_half_size + TTT_Game._CELL_PADDING),
                TTT_Game._MARK_X_LINE_WIDTH)

    def draw_O(self, x, y):
        pygame.draw.circle(
                self._console.canvas.scr,
                self._color,
                (x, y),
                self._cell_size // 2 - TTT_Game._CELL_PADDING,
                TTT_Game._MARK_O_LINE_WIDTH)


class TTT_HumanAgent():
    def __init__(self, console):
        self._console = console

    def action(self, state):
        valid = [n for n,player in enumerate(state) if player==0]

        self._console.cursor = True

        self._console.set_cursor((1, self._console.canvas._nrows-4))
        self._console.print("MOVE?\n")

        while True:
            self._console.set_cursor((0, self._console.canvas._nrows-3))
            move = int(self._console.input(" "))
            if move in valid:
                self._console.cursor = False
                return move

        """
        while(True):
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and \
                        event.key >= pygame.K_0 and \
                        event.key <= pygame.K_8:
                    return int(pygame.key.name(event.key))
        """


def select_players(console):
    console.print("\n SELECT TWO PLAYERS FROM THE LIST:\n\n ")

    cur_tts = console.tts
    console.tts = False
    console.print(
            "1 - HUMAN\n " +
            "2 - AI AGENT (RANDOM )\n " +
            "3 - AI AGENT (MINIMAX HEURISTIC )\n " +
            "4 - AI AGENT (Q-LEARNING )\n\n")

    players = []
    for player in (1, 2):
        players.append(int(console.input(" PLAYER " + str(player) + "? ")))

    console.tts = cur_tts

    num_games = 1
    if TTT_Game._PLAYER_HUMAN not in players:
        num_games = int(console.input(" HOW MANY GAMES? "))

    return players, num_games

def update_stats(stats, info):
    """Actualiza las estadísticas de multi-partida.

    Args:
        stats (list [int, int, int]): [draws, p1 wins, p2 wins]
        info (dict): game.info
    """
    if info["end"] == "draw":
        stats[0] += 1
    if info["end"] == "win":
        stats[info["winner"]] += 1

def show_stats(console, stats):
    console.set_cursor((1, console.canvas._nrows - 8))

    console.print("\n DRAWS: " + str(stats[0]))
    console.print("\n WINS:\n > PLAYER 1: " + str(stats[1]))
    console.print("\n > PLAYER 2: " + str(stats[2]) + "\n\n ")


def show_result(console, info):
    console.set_cursor((1, console.canvas._nrows - 4))

    if info["end"] == "draw":
        console.print("STALEMATE.\n ")
    elif info["end"] == "win":
        console.print("PLAYER " + str(info["winner"]) + " WINS\n\n ")


def init_game(console):
    console.clear()
    console.set_cursor((1, 1))
    console.print("LET'S PLAY TIC TAC TOE!\n ")

    # new game
    while True:
        # select player types and num of games
        players, num_games = select_players(console)

        console.cursor = False
        console.tts = False

        # run game
        it = 0
        stats = [0, 0, 0] # draw, wins 1, wins 2
        while it < num_games:
            # create a new game
            game = TTT_Game(console, players)

            while not game.done:
                game.spin_once()

            # end game beep
            for i in range(3):
                console.make_beep()
                pygame.time.wait(50)

            if num_games>1:
                update_stats(stats, game.info)
            it += 1

        # show result
        console.cursor = True
        if num_games>1:
            show_stats(console, stats)
        else:
            show_result(console, game.info)

        # play again?
        console.tts = True
        if console.input("WANT TO PLAY AGAIN?\n ") not in ("Y", "YES", "OK"):
            break

        console.clear()
