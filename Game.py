import numpy as np
from GUI import *

class Game:

    def __init__(self, aut_car, other_cars, learning=True):
        self.time = 0
        self.vehicle = aut_car
        self.ov = other_cars

        self.game_over = False
        self.same_line, self.over_line = [], []
        self.last_a = 1
        self.learning = learning
        if not learning:
            self.gui = Gui(aut_car, other_cars)

    def do_frame(self, n):
        self.vehicle.update(0.01, n)
        for v in self.ov:
            v.update(0.01*n)
        self.time += 0.01*n

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
            return -1000
        r = 0
        dst = -(self.vehicle.y - self.same_line[0].y)
        if dst > 0:
            self.game_over = True
            return 0
        # if self.last_a != a:
        #     r += -2
        # if self.vehicle.line == 0:
        #     r += -20
        # dst = -(self.vehicle.y - self.same_line[0].y)
        # dst = 0 if dst < 0 else dst
        # r += dst
        r -= 5
        if self.vehicle.y < -1000:
            self.game_over = True

        return r

    def move(self, targ_line):
        # targ_line = self.process_keys()

        if targ_line == 0:
            targ_line = self.last_a

        reference = self.same_line if targ_line == 1 else self.over_line
        targ_vel = reference[0].y_vel

        # print(targ_vel)
        rew = self.reward(targ_line)
        self.last_a = targ_line

        self.vehicle.keep_line(targ_line, targ_vel)

        self.do_frame(10)

        if not self.learning:
            quit_ = self.gui.render(targ_line, targ_vel)

        return rew, quit_
