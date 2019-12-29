import random

class Player:

    def __init__(self, name):
        self.name = name
        self.wins = 0
        self.loss = 0

    def strategy(self, table):
        """Call table to move current position up or left or do nothing."""

    def strategy_diff(self, table, source, target):
        diff = source - target
        if diff < -table.unit:
            table.move_left()
        elif diff > table.unit:
            table.move_right()

class User(Player):

    def __init__(self, name='user'):
        super().__init__(name)

class AI(Player):

    def __init__(self, name='ai'):
        super().__init__(name)

    def strategy(self, table):
        self.strategy_diff(table, table.ball.x, table.paddle.centerx)

class Random(Player):

    def __init__(self, name='random'):
        super().__init__(name)

    def strategy(self, table):
        r = random.randrange(3)
        if r == 1:
            table.move_left()
        elif r == 2:
            table.move_right()

class Model(Player):

    def __init__(self, name='model'):
        super().__init__(name)
        self.target = None
        self.target_set = False

    def strategy(self, table, position):
        if position == 'left_pad':
            if table.ball.dx < 0:
                self.strategy_model(table, position)
            else:
                self.strategy_center(table, position)
        elif table.ball.dx > 0:
            self.strategy_model(table, position)
        else:
            self.strategy_center(table, position)

    def strategy_center(self, table, position):
        self.target_set = False
        self.strategy_diff(table, position, getattr(table, position).centery,
                           table.half)

    def strategy_model(self, table, position):
        if self.target_set:
            self.strategy_diff(table, position, getattr(table,
                position).centery, self.target)
        else:
            self.strategy_diff(table, position, getattr(table,
                position).centery, table.model(position))
            self.target_set = True
