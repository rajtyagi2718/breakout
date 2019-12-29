class Game:

    def __init__(self, table, player, lives=10):
        self.table = table
        self.rally = 0
        self.player = player
        self.lives = lives

    def is_terminal(self):
        return not self.lives or not self.table.bricks

    def end(self):
        if not self.table.bricks:
            self.player.wins += 1
        else:
            self.player.loss += 1

    def advance_player(self):
        self.player.strategy(self.table)

    def reset(self):
        self.reset_scores()
        self.reset_table()

    def reset_scores(self):
        self.lives -= 1
        self.rally = 0

    def reset_table(self):
        self.table.reset_ball()
        for _ in range(5):
            self.advance_player()

    def increment_rally(self):
        self.rally += 1
        if not self.rally % 3:
            self.table.speed_up()
            # print('speed up:', self.table.ball.dt)

    def advance_ball(self):
        for flag in self.table.advance_ball():
            if flag:
                return True
            self.increment_rally()
        return False

    def advance(self):
        result = self.advance_ball()
        self.advance_player()
        return result

    def play(self):
        while True:
            if self.advance():
                self.reset_scores()
                if self.is_terminal():
                    break
                self.reset_table()
        self.end()
