from pygame import Rect
import random

class Brick:


    def __init__(self, unit, left, top, health_dist):
        self.rect = Rect(left*unit, top*unit, unit*8, unit*4)
        self.health = self.init_health(health_dist)

    def init_health(self, health_dist):
        r = random.random()
        if r < health_dist[0]:
            return 1
        if r < health_dist[1]:
            return 2
        if r < health_dist[2]:
            return 3
        return random.randrange(9, 13)

    def hit(self):
        self.health -= 1
        if not self.health:
            return 0
        if self.health < 4:
            return 1
        return 2
