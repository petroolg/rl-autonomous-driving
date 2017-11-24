import numpy as np
from rl import FeatureTransformer, init_game
from sklearn.linear_model import SGDRegressor
from matplotlib import pyplot as plt
import sys
from Game import *
from RigidBody import DummyVehicle
from Vehicle import Vehicle

def softmax(x, theta):
    return 1/(1+np.exp(-x.dot(theta)))

class Model:
    def __init__(self, D, feature_transformer: FeatureTransformer, alpha=0.05, beta = 0.05, var_bound=0.0,start_over=True):
        self.train_tuples, self.n_ex = np.load('train_tuples.npy')
        self.actions = np.array([-1, 0, 1])
        self.ft = feature_transformer
        # if start_over:

        #parameters of policy distribution

        self.state_lenth = 1400
        self.theta = (np.random.sample((self.state_lenth, len(self.actions))) - 0.5) / np.sqrt(self.state_lenth)

        self.G = 0.0  # estimation of total reward
        self.Var = 0.0  # estimation of variance of reward
        self.var_bound = var_bound  # threshold of variance
        self.alpha_step = 0.005  # step of gradient ascent
        self.beta_step = 0.005  # step of gradient ascent
        self.lam = 0.1  # penalization, related to the approximation of COP solution, equations (9) and (10)

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

        # Mean reward, variance and parameter update function, equations (13)
    def update(self, R, zk, type='Var', update_theta=True):
        self.G += self.alpha_step * (R - self.G)
        self.Var += self.alpha_step * (R * R - self.G * self.G - self.Var)
        print('Variance', self.Var)
        print('ZK:', np.min(zk), np.max(zk))
        if update_theta:
            g_prime = 0.0 if (self.Var - self.var_bound) < 0.0 else 2.0 * (self.Var - self.var_bound)
            if type == 'Var':
                print('Update:', (R - self.lam * g_prime * (R * R - 2.0 * self.G)))
                self.theta += self.beta_step * (R - self.lam * g_prime * (R * R - 2.0 * self.G)) * zk
            if type == 'Sharpe':
                self.theta += self.beta_step / np.sqrt(self.Var) * \
                              (R - (self.G * R * R - 2.0 * self.G * self.G * R) / (2 * self.Var)) * zk

    # function returns probability of being in th 2nd state using softmax policy
    def sample_action(self, state, temperature=20.0):
        x = self.ft.transform(np.atleast_2d(state))[0]
        # http://incompleteideas.net/sutton/book/ebook/node17.html
        lst = x.dot(self.theta)/temperature
        # print(lst)
        e_lst = np.exp(lst)
        return e_lst / np.sum(e_lst)

    # gradient of log-likelihood used for computing zk
    def log_like(self, state, temperature=2.0):
        x = self.ft.transform(np.atleast_2d(state))[0]
        lst = x.dot(self.theta) / temperature
        e_lst = np.exp(lst)
        d_lst = np.repeat([[(e_lst[1] + e_lst[2]), (e_lst[0] + e_lst[2]), (e_lst[0] + e_lst[1])]], self.state_lenth,
                          axis=0)
        d_lst = np.multiply(x[np.newaxis].T, d_lst / np.sum(e_lst))

        return d_lst


    def play_one(self, T, eps=0.05, gamma=0.99):
        game = init_game(learning=False)

        game.neighbours()
        s = game.get_state()
        a = self.sample_action(s)

        game.game_over = False
        i = 0
        total_rew, zk = 0.0, 0.0
        q, quit_ = False, False
        # history = []

        while i<T:# and not game.game_over:

            if False:
                p_a = game.gui.process_keys()
            else:
                p_a = self.sample_action(s)
            # history.append((s,p_a))
            # prev_s = s

            s = game.get_state()
            zk += self.log_like(s)

            a = np.random.choice(self.actions, p=p_a)
            rew, q = game.move(a)
            total_rew += rew

            # self.train_tuples = np.append(self.train_tuples, np.array([[prev_s, a, rew]]), axis=0)

            i += 1
            quit_ |= q
            # time.sleep(0.1)
        return total_rew, zk, quit_

# function plotting running average of vector vec, n specifies width of window
def plot_run_avg(vec, n, **kwargs):
    p = []
    vec = np.array(vec)
    for i in range(len(vec)):
        p.append(np.mean(vec[int(max(0, i - n/2)) : int(min(i+n/2, len(vec)-1))]))
    plt.plot(p, **kwargs)

def main():

    ft = FeatureTransformer()

    N_games_learn = 500  # number of games to play for learning
    N_games_test = 200  # number of games to play for data gathering
    length_of_game = 200  # number of steps in one game
    theta_update_step = 75

    gamma = 1.0
    manual_training = False
    variance_bounds = [100.0, 0.0]  # variance bounds

    tr_plot, theta_plot, Var_plot, G_plot = [], [], [], []  # data for plots

    # init_trn()

    # if not manual_training:
    #     if model.train_tuples[0] == [0, 0, 0]:
    #         model.train_tuples = np.delete(model.train_tuples, 0 ,0)
    #
    #     np.random.shuffle(model.train_tuples)
    #     for tup in model.train_tuples:
    #         model.update(*tup)
    for v in variance_bounds:
        game = Model(1400, ft, var_bound=v, start_over=True)
        k = game.K
        total_rews, theta, Var, G = [], [], [], []
        for i in range(k, N_games_learn + k):
            ut = False  # parameter which specifies if theta is updated in that iteration
            if i == int(2 * N_games_learn / 3):  # in the middle of the game make lambda to be almost 1.0
                # this is related to the approximation of COP solution approximation
                # equations (9) and (10) and 7 lines of text under
                game.lam = 0.99
            if i % theta_update_step == 0 and i != 0:
                ut = True  # theta gets updated every 20. iteration
                # theta.append(game.theta.copy())  # gathering data for graph
            Var.append(game.Var)  # gathering data for graph
            G.append(game.G)  # gathering data for graph
            total_rew, zk, quit_ = game.play_one(length_of_game)
            print('Total reward, iteration:', total_rew, i)
            game.update(total_rew, zk, update_theta=ut)  # finally update everything
            if quit_:
                return

        for i in range(N_games_test):
            total_rew, _, quit_ = game.play_one(length_of_game)  # gathering data for graph without update
            total_rews.append(total_rew)  # gathering data for graph

        tr_plot.append(total_rews)  # gathering data for graph
        # theta_plot.append(theta)  # gathering data for graph
        Var_plot.append(Var)  # gathering data for graph
        G_plot.append(G)  # gathering data for graph

            # np.save('train_tuples',np.array([model.train_tuples, model.n_ex+i]))
    plt.figure()
    for rew, v in zip(tr_plot,variance_bounds):
        plt.hist(rew, label='Var bound %f'%v)
    plt.legend()
    plt.title('Total rewards')

    # plt.figure()
    # for theta, v in zip(theta_plot,variance_bounds):
    #     plt.plot(np.arange(N_games_learn / theta_update_step - 1), np.array(theta)[:, 0],
    #              np.arange(N_games_learn / theta_update_step - 1), np.array(theta)[:, 1],
    #              np.arange(N_games_learn / theta_update_step - 1), np.array(theta)[:, 2],
    #              label='Var bound %f'%v)
    # plt.legend()
    # plt.title('Patameters theta')

    plt.figure()
    for rew, v in zip(Var_plot,variance_bounds):
        plot_run_avg(rew, 100, label='Var bound %f'%v)
    plt.legend()
    plt.title('Variance')

    plt.figure()
    for rew, v in zip(G_plot,variance_bounds):
        plot_run_avg(rew, 200, label='Var bound %f'%v)
    plt.legend()
    plt.title('Mean reward')
    plt.show()

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()