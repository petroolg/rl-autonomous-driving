import pygame
from pygame.locals import *

from Rect import *


class RigidBody:
    def __init__(self, **kwargs):

        # graphical
        self.id = kwargs.get('id', 0)
        self.width = kwargs.get('width', 10)
        self.length = kwargs.get('length', 20)
        self.color = kwargs.get('color', (0, 0, 255))
        self.body = Rect(kwargs.get('x', 0.0), kwargs.get('y', 0.0), self.width, self.length, kwargs.get('angle', 0))

        self.m_vel = np.array([kwargs.get('x_vel', 0.0), kwargs.get('y_vel', 0.0)])[np.newaxis].T
        self.m_forces = np.array([0.0, 0.0])[np.newaxis].T
        self.m_mass = kwargs.get('mass', 10.0)
        self.m_pos = np.array([kwargs.get('x', 0.0), kwargs.get('y', 0.0)])[np.newaxis].T

        self.m_angle = 0.0
        self.m_ang_vel = 0.0
        self.m_torque = 0.0
        self.m_inertia = (1.0 / 12.0) * 6**2 * 15**2 * self.m_mass

    @property
    def x(self):
        return self.m_pos[0][0]

    @property
    def y(self):
        return self.m_pos[1][0]

    @property
    def x_vel(self):
        return self.m_vel[0]

    @property
    def y_vel(self):
        return self.m_vel[1]

    def update(self, timestep):
        acc = self.m_forces / self.m_mass
        self.m_vel += acc * timestep
        self.m_pos += self.m_vel * timestep
        # print('Forces:', self.m_forces[0], self.m_forces[1], end='')
        # print('Velocity:', self.m_vel)

        ang_acc = self.m_torque / self.m_mass
        self.m_ang_vel += ang_acc * timestep
        self.m_angle += self.m_ang_vel * timestep
        # print(' Torque:', self.m_torque)
        self.m_torque = 0
        self.m_forces = np.zeros_like(self.m_forces)

    def point_vel(self, w_off):
        tangent = np.array([-w_off[1], w_off[0]])
        return tangent*self.m_ang_vel + self.m_vel

    def add_force(self, w_f, w_off):
        self.m_forces += w_f
        self.m_torque += np.cross(w_off.T, w_f.T)[0]

    def world_to_rel(self, world):
        trans = rot(-self.m_angle)
        return trans.dot(world)

    def rel_to_world(self, rel):
        trans = rot(self.m_angle)
        return trans.dot(rel)

    def draw(self, display, font=None, x=0, y=0):
        self.body.rotate(self.m_angle)
        self.body.translate(self.x, self.y)
        pygame.draw.polygon(display, self.color,
                            self.body.spirit(-x + display.get_width() / 2, -y + display.get_height() / 2))
        # if self.id == 0:
        #     text = str('Force: [%.2f, %.2f]' % (self.m_forces[0], self.m_forces[1]))
        #     text2 = str('Torque: %.2f,' % self.m_torque)
        #     text = font.render(text, True, (0, 0, 0))
        #     display.blit(text, (0, 0))
        #     text2 = font.render(text2, True, (0, 0, 0))
        #     display.blit(text2, (0, 15))

        text = font.render(str(self.m_vel[1]), True, (0, 0, 0))
        display.blit(text, (
            self.x - x + display.get_width() / 2 - text.get_width() / 2,
            self.y - y + display.get_height() / 2 - text.get_height() / 2))

        # self.m_torque = 0
        # self.m_forces = np.zeros_like(self.m_forces)


class DummyVehicle:
    def __init__(self, *args, **kwargs):
        self.id = kwargs.get('id', 0)
        self.width = kwargs.get('width', 10)
        self.length = kwargs.get('length', 20)
        self.color = kwargs.get('color', (0, 0, 255))
        self.body = Rect(kwargs.get('x', 0.0), kwargs.get('y', 0.0), self.width, self.length, kwargs.get('angle', 0))
        self.m_vel = np.array([kwargs.get('x_vel', 0.0), kwargs.get('y_vel', 0.0)])[np.newaxis].T
        self.m_pos = np.array([kwargs.get('x', 0.0), kwargs.get('y', 0.0)])[np.newaxis].T
        self.m_vel_ref = np.array([kwargs.get('x_vel_ref', 0.0), kwargs.get('y_vel_ref', 0.0)])[np.newaxis].T
        self.next_dist = kwargs.get('next_dist', None)  # distance to next car
        self.line = np.sign(self.x)
        # self.m_acc = np.array([0.0, 0.0])[np.newaxis].T
        self.cntr_dist = 120.0  # distance to keep between me and next car

    def draw(self, display, font=None, x=0, y=0):
        self.body.translate(self.x, self.y)
        pygame.draw.polygon(display, self.color,
                            self.body.spirit(-x + display.get_width() / 2, -y + display.get_height() / 2))
        text = font.render(str(self.m_vel[1]), True, (0, 0, 0))
        display.blit(text,
                     (self.x - x + display.get_width() / 2 - text.get_width() / 2,
                      self.y - y + display.get_height() / 2 - text.get_height() / 2))

    def update(self, timestep):
        self.strategy()
        self.control()
        self.m_pos += self.m_vel * timestep

    def control(self):
        if self.next_dist is not None:
            dist_err = (self.y - self.next_dist) - self.cntr_dist
            throttle = dist_err / 6.0
            throttle = 1.0 if throttle > 1.0 else throttle
            throttle = -1.0 if throttle < -1.0 else throttle
            self.m_vel[1] = self.m_vel[1] - throttle
            self.m_vel[1] = -60.0 if self.m_vel[1] < -60.0 else self.m_vel[1]
            self.m_vel[1] = -40.0 if self.m_vel[1] > -40.0 else self.m_vel[1]

    def strategy(self, a=0.005):
        if np.random.rand() < a:
            self.cntr_dist = 130.0 if self.cntr_dist > 130.0 else 220.0


    @property
    def x(self):
        return self.m_pos[0][0]

    @property
    def y(self):
        return self.m_pos[1][0]

    @property
    def y_vel_ref(self):
        return self.m_vel_ref[1]

    @property
    def x_vel(self):
        return self.m_vel[0]

    @property
    def y_vel(self):
        return self.m_vel[1]
