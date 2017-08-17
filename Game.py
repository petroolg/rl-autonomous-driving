import sys

import pygame
from pygame.locals import K_SPACE, K_DOWN, K_LEFT, K_RIGHT, K_UP, QUIT

from Vehicle import *

white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
l_blue = (153,217,234)
orange=(255,128,64)
violet = (128,0,128)
d_green = (0, 176, 0)
grey = (95, 95, 95)


class Game:
    def __init__(self, aut_car, other_cars, learning=True):
        self.steering = 0
        self.throttle = 0
        self.brakes = 0
        self.time = 0
        self.vehicle = aut_car
        self.ov = other_cars
        self.camera_x = aut_car.m_pos[0]
        self.camera_y = aut_car.m_pos[1]
        self.move_ticker = 0
        self.game_over = False
        self.same_line, self.over_line = [], []
        self.last_a = 1
        self.all_actions = [0, 1, 2]
        self.learning = learning
        if not learning:
            pygame.init()
            self.DISPLAY = pygame.display.set_mode((200, 500), 0, 32)

            pygame.font.init()  # you have to call this at the start,
            # if you want to use this module.
            self.myfont = pygame.font.SysFont('Helvetica', 16)


    def do_frame(self, n):

        self.vehicle.set_steering(self.steering)
        self.vehicle.set_brakes(self.brakes)
        self.vehicle.set_throttle(self.throttle)
        for i in range(n):
            self.vehicle.update(0.01)
        for v in self.ov:
            v.update(0.01*n)
        self.time += 0.01*n

    def render(self, targ_line, targ_vel):
        self.DISPLAY.fill(white)
        pygame.draw.polygon(self.DISPLAY, d_green, [(0, 0), (50, 0), (50, 500), (0, 500)])
        pygame.draw.polygon(self.DISPLAY, d_green, [(150, 0), (200, 0), (200, 500), (150, 500)])
        pygame.draw.polygon(self.DISPLAY, grey, [(50, 0), (150, 0), (150, 500), (50, 500)])

        for i in range(26):
            pygame.draw.polygon(self.DISPLAY, white,
                                [(98, 0 - self.camera_y % 20 + i * 20), (102, 0 - self.camera_y % 20 + i * 20),
                                 (102, 10 - self.camera_y % 20 + i * 20), (98, 10 - self.camera_y % 20 + i * 20)])

        self.vehicle.draw(self.DISPLAY, font=self.myfont, x=0, y=self.camera_y[0])
        for c in self.ov:
            c.draw(self.DISPLAY, font=self.myfont, x=0, y=self.camera_y[0])
        text = str('Keys: %.2f, %.2f, %.2f' % (self.steering, self.brakes, self.throttle))
        text = self.myfont.render(text, True, (0, 0, 0))

        self.DISPLAY.blit(text, (0, 0))
        text = self.myfont.render('line:' + str(targ_line) + ' speed:' + str(targ_vel), True, (0, 0, 0))
        self.DISPLAY.blit(text, (0, 15))

        pygame.display.update()

    def process_keys(self):
        keys = pygame.key.get_pressed()

        if keys[K_RIGHT]:
            self.steering = 1
        elif keys[K_LEFT]:
            self.steering = -1
        # else:
        #     self.steering = 0
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
        return self.steering

    # searching for 4 nearest neighbours:
    # 2 in the same line and 2 in other line
    def neighbours(self):
        over_line, same_line = {}, {}  # type: dict
        for w in self.ov:  # type: Vehicle
            line = same_line if np.sign(w.x) > 0 else over_line
            dist = np.linalg.norm(self.vehicle.m_pos - w.m_pos)
            if len(line) < 2:
                line[w] = dist
            else:
                max_v = max(line, key=line.get)
                if dist < line[max_v]:
                    line.pop(max_v)
                    line[w] = dist

        self.over_line = sorted(over_line.keys(), key=lambda v: v.y)
        self.same_line = sorted(same_line.keys(), key=lambda v: v.y)

    def is_collision(self):
        self.neighbours()
        neigh = self.same_line + self.over_line
        lst = [not self.vehicle.body.intersect(w.body) for w in neigh]
        return not np.all(lst)

    def game_over(self):
        return self.game_over

    def get_state(self):
        vy = self.vehicle.y_vel
        vx = self.vehicle.x_vel
        state = [vx, vy, self.vehicle.x]

        for v in (self.same_line + self.over_line):
            vel = v.y_vel
            dy = (self.vehicle.y - v.y)
            dy = 200*np.sign(dy) if abs(dy) > 200 else dy
            dx = (self.vehicle.x - v.x)
            # state += [vel, dy, dx]
            state += [vel, dy]
        return np.array(state).astype(float)

    def reward(self, a):
        if self.is_collision():
            self.game_over = True
            return -10000
        r = 0
        dst = -(self.vehicle.y - self.same_line[0].y)
        if dst > 0:
            self.game_over = True
            return 1000
        if self.last_a != a:
            r += -2
        # if self.vehicle.line == 0:
        #     r += -20
        # dst = -(self.vehicle.y - self.same_line[0].y)
        # dst = 0 if dst < 0 else dst
        # r += dst
        r -= 1
        if self.vehicle.y < -2000:
            self.game_over = True

        return r

    def move(self, targ_line):
        # targ_line = self.process_keys()
        if targ_line == 0:
            targ_line = self.last_a

        reference = self.same_line if targ_line == 1 else self.over_line
        targ_vel = reference[0].y_vel_ref

        # print(targ_vel)
        rew = self.reward(targ_line)
        self.last_a = targ_line

        self.throttle, self.steering, self.brakes = self.vehicle.keep_line(targ_line, targ_vel)

        self.do_frame(10)

        if not self.learning:
            self.render(targ_line, targ_vel)

            # print(self.vehicle.y)

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
        return rew
