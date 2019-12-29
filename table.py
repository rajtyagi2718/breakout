from pygame import Rect
from math import sin, cos, pi, sqrt
import random
from ball import Ball
from brick import Brick

class Table:

    def __init__(self, unit=8):
        self.unit = unit
        self.width = unit*64
        self.height = unit*96
        self.half = unit*32

        self.paddle = Rect(0, 0, unit*12, unit*4)
        self.paddle.midbottom = (self.half, self.height-unit*4)

        self.start_speed = unit
        self.ball = Ball(self.half, self.width,
            *self.random_start_velocity(), self.start_speed, unit*2)

        self.bricks = set()
        # RL features
        self.bricks_map = {}
        self.bricks_copy = set()
        self.bricks_health = []

        self.init_bricks()

    def init_bricks(self):
        # add odd row bricks, leave gap between walls
        for left in range(0, 64, 8):
            self.bricks.add(Brick(self.unit, left, 12, (0,0,0)))
            self.bricks.add(Brick(self.unit, left, 20, (0,.3,1)))
            self.bricks.add(Brick(self.unit, left, 28, (.3,.9,1)))
        # add even row bricks, span width
        for left in range(4, 60, 8):
            self.bricks.add(Brick(self.unit, left, 16, (0,.2,1)))
            self.bricks.add(Brick(self.unit, left, 24, (.4,.9,1)))
            self.bricks.add(Brick(self.unit, left, 32, (.9,1,1)))

        # RL features
        bricks_lst = sorted(self.bricks,
                            key=lambda b: (b.rect.top, b.rect.left))
        self.bricks_map = {b:i for i,b in enumerate(bricks_lst)}
        self.bricks_copy = set(self.bricks)
        self.bricks_health = [b.health for b in bricks_lst]

    def move_left(self):
        self.paddle.left = max(0, self.paddle.left - self.unit)

    def move_right(self):
        self.paddle.right = min(self.width, self.paddle.right + self.unit)

    def move_ball(self):
        self.ball.x += self.ball.dx
        self.ball.y += self.ball.dy
        self.ball.update_rect()

    def hit_brick(self):
        for brick in self.bricks:
            if (self.ball.rect.colliderect(brick) and
                self.hit_brick_flag(brick.rect)):
                flag = brick.hit()
                if not flag:
                    self.bricks.remove(brick)
                if flag == 2:
                    self.random_brick_bounce()
                self.bricks_health[self.bricks_map[brick]] -= 1
                break

    def random_brick_bounce(self):
        # hit_brick_flag was True, so dx dy keep sign
        theta = pi/3 * random.random()**.5
        self.ball.dx = sin(theta) if self.ball.dx >= 0 else -sin(theta)
        self.ball.dy = cos(theta) if self.ball.dy >= 0 else -cos(theta)
        self.move_ball()

    def hit_brick_flag(self, b):
        if b.left+1 <= self.ball.x < b.right+1:
            self.ball.dy *= -1
            return True
        elif b.bottom+1 <= self.ball.y < b.top+1:
            self.ball.dx *= -1
            return True
        else:
            return self.hit_corner(b)

    def hit_paddle(self):
        return (self.ball.dy > 0 and self.ball.rect.colliderect(self.paddle)
                and self.hit_paddle_flag())

    def hit_paddle_flag(self):
        if self.paddle.left+1 <= self.ball.x < self.paddle.right+1:
            self.bounce_paddle()
        elif self.paddle.bottom+1 <= self.ball.y < self.paddle.top+1:
            if self.ball.x < self.paddle.left:
                self.bounce_corner(-1, -1)
            else:
                self.bounce_corner(1, -1)
        else:
            return self.hit_corner(self.paddle)
        return True

    def hit_corner(self, paddle):
        if self.ball.collide_top_left(paddle):
            self.bounce_corner(1, 1)
        elif self.ball.collide_top_right(paddle):
            self.bounce_corner(-1, 1)
        elif self.ball.collide_bottom_right(paddle):
            self.bounce_corner(-1, -1)
        elif self.ball.collide_bottom_left(paddle):
            self.bounce_corner(1, -1)
        else:
            return False
        return True

    def bounce_corner(self, x_sgn, y_sgn):
        # add small degree of randomness, prevent unseen loops
        r = random.random()
        theta = pi/3 + r * (pi/24)
        self.ball.dx = x_sgn * sin(theta)
        self.ball.dy = y_sgn * cos(theta)

    def bounce_paddle(self):
        diff = self.ball.x - self.paddle.centerx
        max_diff = self.unit*6
        max_theta = pi/4
        theta = max_theta * min((diff / max_diff), 1)
        # print('paddle diff=%.2f, max_diff=%.2f' % (diff, max_diff))
        self.ball.dx = sin(theta)
        self.ball.dy = -cos(theta)

    def hit_side(self):
        if ((self.ball.x <= self.ball.r and self.ball.dx < 0) or
            (self.ball.x >= self.width-self.ball.r and self.ball.dx > 0)):
            self.ball.dx *= -1

    def hit_ceiling(self):
        if self.ball.y <= self.ball.r and self.ball.dy < 0:
            self.ball.dy *= -1

    def hit_floor(self):
        return self.ball.y >= self.height + self.ball.r

    def advance_ball(self):
        result = []
        for _ in range(self.ball.dt):
            self.move_ball()
            self.hit_side()
            self.hit_ceiling()
            if self.hit_brick():
                pass
            elif self.hit_paddle():
                # print(self.ball.y/self.unit)
                result.append(0)
            elif self.hit_floor():
                # print(self.ball.y/self.unit)
                result.append(1)
                break
        # return False
        return result

    def get_round_ball_center(self):
        return (round(self.ball.x, 2), round(self.ball.y, 2))

    def speed_up(self):
        self.ball.dt += self.unit // 4

    def reset_ball(self):
        self.ball.x = self.half
        self.ball.y = self.width
        self.ball.dx, self.ball.dy = self.random_start_velocity()
        self.ball.dt = self.start_speed

    def random_start_velocity(self):
        theta = pi/6 * (random.random() - .5)
        return (sin(theta), cos(theta))

    def reset(self):
        self.paddle.midbottom = (self.half, self.height-self.unit*4)
        self.reset_ball()
        self.bricks = set()
        self.init_bricks()
