#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Falken Program."""

import sys
sys.path.append(".")
#sys.path.append("./console")
sys.path.append("./games/tictactoe")

import pygame
from console.console import WinConsole
from games.tictactoe.falken_ttt_game import init_game as ttt_game_launch

_GAME_ID_TIC_TAC_TOE = 2


def welcome(console):
    """Bienvenida a Falken.

    Sólo se ejecuta una vez al iniciar la app. Al final, pregunta si queremos
    iniciar un nuevo juego.

    Returns:
        bool: True si nuevo juego.
    """
    console.set_cursor((1, 1))
    pygame.time.wait(2000)

    console.print("GREETINGS PROFESSOR FALKEN.\n\n ")
    console.print("IT'S BEEN A LONG TIME. HOW ARE YOU FEELING TODAY?\n\n ")

    console.input()

    console.print("\n EXCELENT! SHALL WE PLAY A GAME?\n\n ")

    resp = console.input()

    return resp in ("Y", "YES", "OK")


def select_game(console):
    """Selects a game to play"""

    while True:
        console.clear()

        console.set_cursor((1, 1))
        console.print("PLEASE! SELECT A GAME FROM THE LIST:\n\n ")

        cur_tts = console.tts
        console.tts = False
        console.print("1 - FALKEN'S MAZE" +
                "\n 2 - TIC-TAC-TOE" +
                "\n 3 - BLACK JACK" +
                "\n 4 - CONNECT-4" +
                "\n 5 - CHECKERS" +
                "\n 6 - GLOBAL THERMONUCLEAR WAR\n\n ")

        game = int(console.input())

        if game in (_GAME_ID_TIC_TAC_TOE,):
            console.tts = cur_tts
            return game

        console.tts = cur_tts
        console.print("\n OH! SORRY. THAT GAME IS NOT CURRENTLY AVAILABLE.\n\n ")
        pygame.time.wait(1000)


def launch_ttt(console):
    """Launch Falken Tic Tac Toe game."""

    pass


def main(width, height, full=False):
    """Main program."""

    # Crear la ventana principal
    win = None
    if full:
        # pantalla completa
        win  = WinConsole(0, 0, full=True, tts=True, font_size=64)
    else:
        # ventana width x height
        win  = WinConsole(width, height, title="WinConsole Demo", tts=True)

    # Obtener una referencia a la consola principal
    console = win.console

    # Ajustamos la velocidad del tts
    console.tts_engine.setProperty('rate', 150)
    console.tts_engine.setProperty('voice', 'english-us')

    # welcoming
    if not welcome(console):
        console.print("\n OH! SORRY TO HEAR THAT.\n MAYBE ANOTHER DAY.\n\n ")
        win.exit()

    game_id = select_game(console)

    """
    if game_id == _GAME_ID_TIC_TAC_TOE:
        ttt_game_launch(console)
    """

    ttt_game_launch(console)

    win.exit()

    """
    pygame.display.update()

    # pygame event dispatcher
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                app_exit()
    """

if __name__ == '__main__':
    if len(sys.argv) == 2:
        # window init
        w,h = sys.argv[1].split("x")
        main(int(w), int(h))
    else:
        # full screen init
        main(0, 0, full=True)
