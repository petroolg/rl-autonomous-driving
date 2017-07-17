import numpy as np
import pygame
from pygame.locals import *


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


class Rigid_body:

    def __init__(self, params):

        # graphical
        self.width = params.get('width', 6)
        self.length = params.get('length', 15)
        self.color = params.get('color', (0,0,255))
        self.icon = np.array([[self.width, self.width, -self.width, -self.width],
            [self.length, -self.length, -self.length, self.length]])

        self.m_pos = np.array([params.get('x', 0.0), params.get('y', 0.0)])[np.newaxis].T
        self.m_vel = np.array([params.get('x_vel', 0.0), params.get('y_vel', 0.0)])[np.newaxis].T
        self.m_forces = np.array([0.0, 0.0])[np.newaxis].T
        self.m_mass = params.get('mass', 10)

        self.m_angle = 0.0
        self.m_ang_vel = 0.0
        self.m_torque = 0.0
        self.m_inertia = (1.0 / 12.0) * 6**2 * 15**2 * self.m_mass

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


    def draw(self, DISPLAY, font=None, x=0, y=0):

        trans = rot(self.m_angle)
        par = trans.dot(self.icon) + np.repeat(self.m_pos+np.array([[-x+DISPLAY.get_width()/2],[-y+DISPLAY.get_height()/2]]), 4, axis=1)
        par = tuple(list(par.T))
        pygame.draw.polygon(DISPLAY, self.color, par)
        if font:
            text = str('Force: [%.2f, %.2f]'%(self.m_forces[0], self.m_forces[1]))
            text2 = str('Torque: %.2f,' % self.m_torque)
            text = font.render(text, True, (0,0,0))
            DISPLAY.blit(text, (0, 0))
            text2 = font.render(text2, True, (0, 0, 0))
            DISPLAY.blit(text2, (0, 15))
        self.m_torque = 0
        self.m_forces = np.zeros_like(self.m_forces)
        # pygame.draw.line(DISPLAY, (0, 255, 0), (self.m_pos[0] + w_off[0],
        #                                         self.m_pos[1] + w_off[1]),
        #                  (self.m_pos[0] + w_off[0] - 1000 * self.m_torque * w_off[1] / np.linalg.norm(w_off),
        #                   self.m_pos[1] + w_off[1] + 1000 * self.m_torque * w_off[0] / np.linalg.norm(w_off)))



