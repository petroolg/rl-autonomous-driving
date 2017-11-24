import pygame
from pygame.locals import K_SPACE, K_DOWN, K_LEFT, K_RIGHT, K_UP, QUIT
import numpy as np

white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
l_blue = (153, 217, 234)
orange=(255,128,64)
violet = (128, 0, 128)
d_green = (0, 176, 0)
grey = (95, 95, 95)

class Gui:

    def __init__(self, car, ov):
        self.camera_y = car.m_pos[1]
        self.move_ticker = 0

        pygame.init()
        self.DISPLAY = pygame.display.set_mode((200, 500), 0, 32)

        pygame.font.init()  # you have to call this at the start,
        # if you want to use this module.
        self.myfont = pygame.font.SysFont('Helvetica', 16)

        self.car = car
        self.ov = ov

    def render(self, targ_line, targ_vel):
        self.DISPLAY.fill(white)
        pygame.draw.polygon(self.DISPLAY, d_green, [(0, 0), (50, 0), (50, 500), (0, 500)])
        pygame.draw.polygon(self.DISPLAY, d_green, [(150, 0), (200, 0), (200, 500), (150, 500)])
        pygame.draw.polygon(self.DISPLAY, grey, [(50, 0), (150, 0), (150, 500), (50, 500)])

        for i in range(26):
            pygame.draw.polygon(self.DISPLAY, white,
                                [(98, 0 - self.camera_y % 20 + i * 20), (102, 0 - self.camera_y % 20 + i * 20),
                                 (102, 10 - self.camera_y % 20 + i * 20), (98, 10 - self.camera_y % 20 + i * 20)])

        self.car.draw(self.DISPLAY, font=self.myfont, x=0, y=self.camera_y[0])
        for c in self.ov:
            c.draw(self.DISPLAY, font=self.myfont, x=0, y=self.camera_y[0])
        text = str('Keys: %.2f, %.2f, %.2f' % (self.car.steering, self.car.brakes, self.car.throttle))
        text = self.myfont.render(text, True, (0, 0, 0))

        self.DISPLAY.blit(text, (0, 0))
        text = self.myfont.render('line:' + str(targ_line) + ' speed:' + str(targ_vel), True, (0, 0, 0))
        self.DISPLAY.blit(text, (0, 15))

        pygame.display.update()

        quit_ = False
        for event in pygame.event.get():
            if event.type == QUIT:
                quit_ = True

        return quit_

    def process_keys(self):
        keys = pygame.key.get_pressed()
        lst = np.array([0,0,0])
        if keys[K_RIGHT]:
            self.steering = 1
        elif keys[K_LEFT]:
            self.steering = -1
        else:
            self.steering = 0
        # if keys[K_UP]:
        #     if self.throttle < 0.9:
        #         self.throttle += 0.1
        # else:
        #     self.throttle -= 0.1 if self.throttle > 0.1 else 0
        # if keys[K_DOWN] or keys[K_SPACE]:
        #     self.brakes = 1
        # else:
        #     self.brakes -= 0.1 if self.brakes > 0.1 else 0
        # if keys[K_LEFT] or keys[K_RIGHT] or keys[K_UP] or keys[K_DOWN]:
        #     self.move_ticker = 20
        lst[self.steering+1] = 1.0
        return lst