import numpy as np
from rl import FeatureTransformer, init_game, plot_running_avg
from sklearn.linear_model import SGDRegressor
from matplotlib import pyplot as plt
import sys
from Game import *
from RigidBody import DummyVehicle
from Vehicle import Vehicle

def softmax(x, theta):
    return 1/(1+np.exp(-x.dot(theta)))

class Model:
    def __init__(self, D, var, feature_transformer: FeatureTransformer, alpha=0.05, beta = 0.05, start_over=True):
        self.alpha = alpha
        self.beta = beta
        self.lam = 0.1
        self.var_bound = var
        self.train_tuples, self.n_ex = np.load('train_tuples.npy')

        self.ft = feature_transformer
        # if start_over:

        #parameters of policy distribution

        self.theta = np.random.random_sample((D,3))/np.sqrt(D)
        self.G = 0.0
        self.Var = 0.0

        self.K = 0
        # else:
        #     self.models = np.load('models.npy')[0]
        #     print(self.models[0].weights)
        #     self.K = np.load('models.npy')[1]
        # self.ft = feature_transformer

    def predict(self, s):
        x = self.ft.transform(np.atleast_2d(s))
        assert (len(x.shape) == 2)
        pred = softmax(x, self.theta)
        return np.argmax(pred) - 1
        # return list(map(lambda model, x: model.predict(x)[0], self.models, [x]*3))

    def update(self, R, zk):
        self.G += self.alpha * (R - self.G)
        self.Var += self.alpha * (R * R - self.G * self.G - self.Var)
        g_prime = 0.0 if (self.Var - self.var_bound) < 0 else 2 * (self.Var - self.var_bound)
        self.theta += self.beta * (R - self.lam * g_prime * (R * R - 2 * self.G)) * zk

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return np.random.choice([-1, 0, 1])
        else:
            pred = self.predict(s)
            return pred

    def log_like(self, s):
        x = self.ft.transform(np.atleast_2d(s))
        v = np.exp(np.multiply(x,self.theta)) + 1
        v = x.T.dot(v)/v.dot(v.T)
        return v

class RL_Var:
    def __init__(self, model):
        self.GAMMA = 0.9
        self.model = model

    def play_one(self, n, eps=0.05, gamma=0.99):
        game = init_game(learning=False)

        game.neighbours()
        s = game.get_state()
        a = self.model.sample_action(s, 0.05)

        game.game_over = False
        iters = 0
        total_rew, zk = 0.0, 0.0
        q, quit_ = False, False
        history = []

        while not game.game_over:

            if n < 0:
                a = int(game.gui.process_keys())
            else:
                a = self.model.sample_action(s, eps)

            history.append((s,a))
            prev_s = s

            rew, q = game.move(a)
            s = game.get_state()

            total_rew += rew
            zk += self.model.log_like(s)

            self.model.train_tuples = np.append(self.model.train_tuples, np.array([[prev_s, a, rew]]), axis=0)

            iters += 1
            quit_ |= q
            # time.sleep(0.1)
        return total_rew, zk, quit_

def main():

    ft = FeatureTransformer()
    model = Model(1400, 4000000, ft, start_over=True)
    rl = RL_Var(model)
    N = 300
    k = model.K
    totalrewards = []
    gamma = 1.0
    manual_training = False

    # init_trn()

    # if not manual_training:
    #     if model.train_tuples[0] == [0, 0, 0]:
    #         model.train_tuples = np.delete(model.train_tuples, 0 ,0)
    #
    #     np.random.shuffle(model.train_tuples)
    #     for tup in model.train_tuples:
    #         model.update(*tup)

    for i in range(k, N + k):
        eps = 0.5 / np.sqrt(i + 1)
        # eps = 0
        totalreward, zk, quit = rl.play_one(i, eps=eps, gamma=gamma)
        totalrewards.append(totalreward)
        model.update(totalreward, zk)

        print("episode:", i, "total reward:", totalreward, "eps:", eps)
        print(model.theta[0])
        print('Var', model.Var, 'J', model.G)
        # np.save('models', [np.array(model.models), i])
        if quit:
            break
        # np.save('train_tuples',np.array([model.train_tuples, model.n_ex+i]))
    totalrewards = np.array(totalrewards)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())


    plt.plot(totalrewards)
    plt.title('Rewards')
    plt.show()

    plot_running_avg(totalrewards)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()