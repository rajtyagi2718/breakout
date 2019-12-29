#!/usr/local/bin/python3

from player import User, AI, Random
from ql3 import ql
from table import Table
from game import Game
from ui import UI

def ui_play(player):
    table = Table()
    game = Game(table, player)
    ui = UI(game)
    ui.start()

def game_play(player):
    table = Table()
    game = Game(table, player)
    game.play()
    print('bricks remaining:', len(table.bricks))
    print('lives remaining:', game.lives)

import time

if __name__ == '__main__':
    # ui_play(AI())
    ui_play(ql)
    # game_play(AI())
    # for _ in range(100):
        # ai = AI()
        # game_play(ai)
        # print(ai.wins, ai.loss)
