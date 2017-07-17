import numpy as np
import pygame
from pygame.locals import *
import time
import sys
from Rigid_body import Rigid_body, rot
from New_model import *
from pygame import examples
from pygame.examples import eventlist

white = (255,255,255)
black = (0,0,0)
green = (0,255,0)
red = (255,0,0)
blue = (0,0,255)
d_green=(0,176,0)
grey=(95,95,95)

class Game:

    def __init__(self):
        self.steering = 0
        self.throttle = 0
        self.brakes = 0
        self.vehicle = Vehicle({'y_vel':-80, 'x':30})
        self.ov = []
        self.camera_x = self.vehicle.m_pos[0]
        self.camera_y = self.vehicle.m_pos[1]

        self.maneure = False

        pygame.init()
        self.DISPLAY = pygame.display.set_mode((200, 500), 0, 32)

        pygame.font.init()  # you have to call this at the start,
        # if you want to use this module.
        self.myfont = pygame.font.SysFont('Helvetica', 16)

    def do_frame(self):
        self.DISPLAY.fill(white)
        self.vehicle.set_steering(self.steering)
        self.vehicle.set_brakes(self.brakes)
        self.vehicle.set_throttle(self.throttle)
        self.vehicle.update(0.01)
        for v in self.ov:
            v.update(0.01)

    def render(self):

        pygame.draw.polygon(self.DISPLAY, d_green, [(0, 0), (50, 0), (50, 500), (0, 500)])
        pygame.draw.polygon(self.DISPLAY, d_green, [(150, 0), (200, 0), (200, 500), (150, 500)])
        pygame.draw.polygon(self.DISPLAY, grey, [(50, 0), (150, 0), (150, 500), (50, 500)])

        for i in range(26):
            pygame.draw.polygon(self.DISPLAY, white, [(98, 0-self.camera_y%20+i*20), (102, 0-self.camera_y%20+i*20),
                                                      (102, 10-self.camera_y%20+i*20), (98, 10-self.camera_y%20+i*20)])

        self.vehicle.draw(self.DISPLAY, font=self.myfont, x=0, y=self.camera_y[0])
        for c in self.ov:
            c.draw(self.DISPLAY, x=0, y=self.camera_y[0])
        text = str('Keys: %.2f, %.2f, %.2f' % (self.steering, self.brakes, self.throttle))
        text = self.myfont.render(text, True, (0, 0, 0))
        self.DISPLAY.blit(text, (0, 30))
        pygame.display.update()


    def process_keys(self):
        keys = pygame.key.get_pressed()

        if keys[K_RIGHT]:
            self.steering = 1
        elif keys[K_LEFT]:
            self.steering = -1
        else:
            self.steering = 0
        if keys[K_UP]:
            if self.throttle < 0.9:
                self.throttle += 0.1
        else:
            self.throttle -= 0.1 if self.throttle > 0.1 else 0
        if keys[K_DOWN] or keys[K_SPACE]:
            self.brakes = 1
        else:
            self.brakes -= 0.1 if self.brakes > 0.1 else 0
        if keys[K_LEFT] or keys[K_RIGHT] or keys[K_UP] or keys[K_DOWN]:
            self.move_ticker = 20


if __name__ == '__main__':

    body = Game()


    body.ov.append(Vehicle({'width': 10, 'length': 20, 'color': green, 'x': 30.0, 'y': 60.0, 'y_vel':-30}))
    body.ov.append(Vehicle({'width': 10, 'length': 20, 'color': red, 'x': 30.0, 'y': -60.0, 'y_vel':-30}))

    # body.ov.append(Vehicle({'width': 10, 'length': 20,'color': red, 'x': -30.0, 'y': 0.0, 'y_vel':-50}))
    body.ov.append(Vehicle({'width': 10, 'length': 20, 'color': black, 'x': -30.0, 'y': 200.0, 'y_vel':-50}))
    body.ov.append(Vehicle({'width': 10, 'length': 20, 'color': green, 'x': -30.0, 'y': 40.0, 'y_vel':-50}))
    targ = 2
    err =0
    while True:
        # offs = body.world_to_rel(np.array([20, 0])[np.newaxis].T)
        # body.update(0.01)
        # body.add_force(np.array([0, 0.0005])[np.newaxis].T, offs)
        # body.draw()
        # body.process_keys()
        body.do_frame()
        body.render()
        err = body.vehicle.keep_line(targ, -30, err, body)
        # time.sleep(1)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()