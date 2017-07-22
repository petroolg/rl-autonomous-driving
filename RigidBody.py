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
        self.m_mass = kwargs.get('mass', 10)
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

        text = font.render(str(self.id), True, (0, 0, 0))
        display.blit(text, (
        self.x - x + display.get_width() / 2 - self.width, self.y - y + display.get_height() / 2 - self.length))

        # self.m_torque = 0
        # self.m_forces = np.zeros_like(self.m_forces)