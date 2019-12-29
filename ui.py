import pygame
from pygame import Color

class UI:

    brick_color = {1:Color('white'), 2:Color('red'), 3:Color('blue')}
    brick_color.update((i, Color('yellow')) for i in range(4, 13))

    def __init__(self, game):
        self.game = game
        self.table = game.table

        pygame.init()
        pygame.display.set_caption('Breakout')
        self.surface = pygame.display.set_mode((self.table.width,
                                                self.table.height))
        self.font = pygame.font.SysFont('monospace', self.table.unit*16)

        self.table.ball.rect = pygame.draw.circle(self.surface, Color('red'), (self.table.ball.x, self.table.ball.y), self.table.ball.r)

        self.clock = pygame.time.Clock()

    def start(self):
        self.pause_loop(1)
        self.main_loop()

    def reset(self):
        self.pause_loop(.5)
        self.game.reset()
        self.update()
        self.pause_loop(2.5)

    def pause_loop(self, sec):
        for _ in range(int(sec*60)):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            self.key_press(pygame.key.get_pressed())
            self.update()
            self.clock.tick(60)

    def get_surf_array(self):
        return pygame.surfarray.array2d(self.surface).astype(bool)

    def main_loop(self):
        while True:
            if self.game.is_terminal():
                self.pause_loop(5)
                pygame.quit()
                return
                ### if left_wins do ... ###

            self.advance_frame()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            self.key_press(pygame.key.get_pressed())
            self.update()
            self.clock.tick(60)

    def advance_frame(self):
        flag = self.game.advance()
        self.update()
        if flag:
            self.reset()

    def update(self):
        self.surface.fill(Color('black'))
        self.draw_objects()
        pygame.display.flip()

    def draw_objects(self):
        pygame.draw.rect(self.surface, Color('white'), self.table.paddle)
        self.draw_lives()
        # self.draw_lines()
        self.draw_bricks()
        pygame.draw.circle(self.surface, Color('red'), (int(self.table.ball.x), int(self.table.ball.y)), self.table.ball.r)

    def draw_bricks(self):
        for b in self.table.bricks:
            pygame.draw.rect(self.surface, self.brick_color[b.health], b.rect)
            pygame.draw.rect(self.surface, Color('black'), b.rect, self.table.unit//2)

    def draw_lines(self):
        pygame.draw.line(self.surface, (255,255,0), (0,0), (1023,0), 1)
        pygame.draw.line(self.surface, (255,255,0), (0,0), (0,511), 1)
        pygame.draw.line(self.surface, (255,255,0), (1023,0), (1023,511), 1)
        pygame.draw.line(self.surface, (255,255,0), (0,511), (1023,511), 1)

        pygame.draw.line(self.surface, (255,255,0), (32,0), (32,511), 1)
        pygame.draw.line(self.surface, (255,255,0), (991,0), (991,511), 1)

        pygame.draw.line(self.surface, (255,255,0), (0,32), (1023,32), 1)
        pygame.draw.line(self.surface, (255,255,0), (0,479), (1023,479), 1)
        pygame.draw.line(self.surface, (255,255,0), (479, 0), (479,511), 1)
        pygame.draw.line(self.surface, (255,255,0), (544,0), (544,511), 1)

    def draw_lives(self):
        font = self.font.render(str(self.game.lives), True, Color('white'))
        rect = font.get_rect()
        rect.top = 1.5 * self.table.unit
        rect.centerx = self.table.half
        self.surface.blit(font, rect)

    def key_press(self, pressed):
        if pressed[pygame.K_a]:
            self.table.move_left()
        if pressed[pygame.K_d]:
            self.table.move_right()
        if pressed[pygame.K_LEFT]:
            self.table.move_left()
        if pressed[pygame.K_RIGHT]:
            self.table.move_right()
