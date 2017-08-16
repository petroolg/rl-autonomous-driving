import numpy as np


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


class Rect:
    def __init__(self, x, y, width, length, angle=0):
        self.pos = np.array([x, y])
        self.angle = angle
        self.width = width
        self.length = length
        self.icon = np.array([[self.width, self.width, -self.width, -self.width],
                              [self.length, -self.length, -self.length, self.length]])

    def rotate(self, angle):
        self.angle = angle

    def translate(self, x=0, y=0):
        self.pos = np.array([x, y])

    def spirit(self, x_trans=0, y_trans=0):
        out = rot(self.angle).dot(self.icon) + np.repeat(self.pos[np.newaxis].T + np.array([[x_trans], [y_trans]]), 4,
                                                         axis=1)
        out = tuple(list(out.T))
        return out

    def intersect(self, rect):
        axes = []
        a = self.spirit()
        b = rect.spirit()
        axes.append(a[0] - a[1])
        axes.append(a[0] - a[3])
        axes.append(b[0] - b[1])
        axes.append(b[0] - b[3])

        for axis in axes:
            proj_a = []
            proj_b = []
            nrm = np.linalg.norm(axis)
            for i in range(4):
                p = a[i]
                proj_a.append(axis.T.dot(p.T.dot(axis) / (nrm * nrm) * axis))
            for i in range(4):
                p = b[i]
                proj_b.append(axis.T.dot(p.T.dot(axis) / (nrm * nrm) * axis))

            min_v = proj_a if min(proj_a) < min(proj_b) else proj_b
            max_v = proj_b if min_v == proj_a else proj_a

            if max(max_v) <= max(min_v) or max(min_v) >= min(max_v):
                continue
            else:
                return False

        return True
