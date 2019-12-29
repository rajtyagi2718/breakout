from pygame import Rect
from numpy import round

class Ball:

    def __init__(self, x, y, dx, dy, dt, r):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.r = r
        self.rect = Rect(x-r, x-y, 2*r, 2*r)
        self.top_left_mask = []
        self.top_right_mask = []
        self.bottom_left_mask = []
        self.bottom_right_mask = []
        self.init_mask()

    def update_rect(self):
        self.rect.center = (int(round(self.x)), int(round(self.y)))

    def init_mask(self):
        ball = [(i,j) for i in range(0, 1+self.r)
                      for j in range(0, 1+self.r)
                      if i**2 + j**2 <= self.r**2]
        self.bottom_right_mask = [(i,j) for i,j in ball
                                        if (i+1, j) not in ball
                                        or (i, j+1) not in ball]
        self.top_left_mask = [(-i, -j) for i,j in self.bottom_right_mask]
        self.top_right_mask = [(i, -j) for i,j in self.bottom_right_mask]
        self.bottom_left_mask = [(-i, j) for i,j in self.bottom_right_mask]

    def collide_top_left(self, pad):
        return any(pad.collidepoint(i+self.x, j+self.y)
                   for i,j in self.top_left_mask)

    def collide_top_right(self, pad):
        return any(pad.collidepoint(i+self.x, j+self.y)
                   for i,j in self.top_right_mask)

    def collide_bottom_left(self, pad):
        return any(pad.collidepoint(i+self.x, j+self.y)
                   for i,j in self.bottom_left_mask)

    def collide_bottom_right(self, pad):
        return any(pad.collidepoint(i+self.x, j+self.y)
                   for i,j in self.bottom_right_mask)
