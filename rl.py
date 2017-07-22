import numpy as np

from Game import *
from Vehicle import Vehicle


class RL:
    def __init__(self):
        self.GAMMA = 0.9

    def init_game(self, learning=True):
        car = Vehicle(y_vel=-50, x=30)
        other_cars = []
        other_cars.append(
            Vehicle(id=1, width=10, length=20, color=green, x=30.0, y=100.0, y_vel=-50, y_vel_ref=-50, line_ref=1))
        other_cars.append(
            Vehicle(id=2, width=10, length=20, color=red, x=30.0, y=-60.0, y_vel=-50, y_vel_ref=-50, line_ref=1))
        other_cars.append(
            Vehicle(id=6, width=10, length=20, color=blue, x=-30.0, y=200.0, y_vel=-70, y_vel_ref=-70, line_ref=2))
        other_cars.append(
            Vehicle(id=3, width=10, length=20, color=red, x=-30.0, y=40.0, y_vel=-70, y_vel_ref=-70, line_ref=2))
        other_cars.append(
            Vehicle(id=5, width=10, length=20, color=green, x=-30.0, y=280.0, y_vel=-70, y_vel_ref=-70, line_ref=2))

        self.game = Game(car, other_cars, learning)

    def sa_to_x(self, s, a):
        x = []
        f = [1 if a == act else 0 for act in self.game.all_actions]
        for a in f:
            x += [state * a for state in s]
        x.append(1)
        return np.array(x).T

    def random_action(self, a_i, eps=0.1):
        fort = np.random.random()
        if fort < (1 - eps):
            return self.game.all_actions[a_i]
        else:
            poss_act = self.game.all_actions.copy()
            # if windy:
            #     poss_act.remove(a)
            return np.random.choice(poss_act)


if __name__ == '__main__':

    rl = RL()
    theta = np.load('theta.npy')
    # theta = np.random.rand(43)
    t = 1
    k = 0
    for i in range(1000):

        rl.init_game(learning=True)

        game = rl.game

        game.neighbours()
        s = game.get_state()
        # print(theta)
        # print(rl.sa_to_x(s,1))
        a_i = np.argmax([theta.dot(rl.sa_to_x(s, a)) for a in game.all_actions])
        a = rl.random_action(a_i)

        if k % 10 == 0:
            t += 0.001
        print(k)

        while not game.game_over:
            game.neighbours()

            r = game.move(a)
            # print(a)

            sp = game.get_state()
            a_i = np.argmax([theta.dot(rl.sa_to_x(s, a)) for a in game.all_actions])
            ap = rl.random_action(a_i)
            # if r != 0:
            #     print(r)
            ALPHA = 0.001 / t
            # print(rl.sa_to_x(sp, ap))
            theta += ALPHA * (r + rl.GAMMA * theta.dot(rl.sa_to_x(sp, ap)) - theta.dot(rl.sa_to_x(s, a))) * rl.sa_to_x(
                s, a)
            s = sp
            a = ap
        np.save('theta', theta)
        print(theta)
        k += 1

    rl.init_game()
    game = rl.game
    while not game.game_over:
        game.neighbours()

        a_i = np.argmax([theta.dot(rl.sa_to_x(s, a)) for a in game.all_actions])
        a = game.all_actions[a_i]

        reference = game.same_line if a == 1 else game.over_line
        vel = list(reference.items())[1][0].y_vel_ref
        r = game.move(a, vel, learning=False)
