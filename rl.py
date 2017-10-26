import sys
from matplotlib import pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from Game import *
from RigidBody import DummyVehicle
from Vehicle import Vehicle


class FeatureTransformer:
    def __init__(self):
        N = 20000
        observation_examples = np.hstack((np.random.random((N, 1)) * 60.0 - 30.0,  # vx
                                          -np.random.random((N, 1)) * 100.0,  # vy
                                          np.random.random((N, 1)) * 80.0 - 40.0,  # x
                                          -np.random.random((N, 1)) * 100.0,  # vy1
                                          np.random.random((N, 1)) * 400.0 - 200.0,  # d1
                                          -np.random.random((N, 1)) * 100.0,  # vy2
                                          np.random.random((N, 1)) * 400.0 - 200.0,  # d2
                                          -np.random.random((N, 1)) * 100.0,  # vy3
                                          np.random.random((N, 1)) * 400.0 - 200.0,  # d3
                                          -np.random.random((N, 1)) * 100.0,  # vy4
                                          np.random.random((N, 1)) * 400.0 - 200.0,  # d4
                                          ))
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma=0.05, n_components=700)),
            ('rbf2', RBFSampler(gamma=0.1, n_components=700)),
            # ('rbf3', RBFSampler(gamma=0.3, n_components=700)),
            # ('rbf4', RBFSampler(gamma=0.5, n_components=700)),
            # ('rbf5', RBFSampler(gamma=0.6, n_components=700)),
            # ('rbf6', RBFSampler(gamma=0.7, n_components=700)),
            # ('rbf7', RBFSampler(gamma=0.8, n_components=700)),
            # ('rbf8', RBFSampler(gamma=0.9, n_components=700)),
            # ('rbf9', RBFSampler(gamma=1.0, n_components=700))
        ])

        examples = featurizer.fit_transform(scaler.transform(observation_examples))
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, obs):
        scaled = self.scaler.transform(obs)
        return self.featurizer.transform(scaled)


class Model:
    def __init__(self, feature_transformer: FeatureTransformer, learning_rate=0.05, start_over=True):
        game = init_game(learning=False)
        game.neighbours()
        s = game.get_state()
        self.train_tuples, self.n_ex = np.load('train_tuples.npy')

        self.ft = feature_transformer
        # if start_over:
        self.models = []  # type: list[SGDRegressor]
        for i in range(3):
            model = SGDRegressor(learning_rate=learning_rate)
            model.partial_fit(feature_transformer.transform([s]), [0])
            self.models.append(model)
            self.K = 0
            # else:
            #     self.models = np.load('models.npy')[0]
            #     print(self.models[0].weights)
            #     self.K = np.load('models.npy')[1]
            # self.ft = feature_transformer

    def predict(self, s):
        x = self.ft.transform(np.atleast_2d(s))
        assert (len(x.shape) == 2)
        return np.array([m.predict(x)[0] for m in self.models])
        # return list(map(lambda model, x: model.predict(x)[0], self.models, [x]*3))

    def update(self, s, a, G):
        x = self.ft.transform(np.atleast_2d(s))
        self.models[a + 1].partial_fit(x, [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return np.random.choice([-1, 0, 1])
        else:
            pred = self.predict(s)
            return np.argmax(pred) - 1


def init_game(learning=True):
    car = Vehicle(y_vel=-30, x=30)
    other_cars = []
    colors = [blue, white, l_blue, violet, green, orange]
    y = 20
    n = np.random.randint(5,9)

    other_cars.append(
        DummyVehicle(id=3, color=blue, x=-30.0, y=y, y_vel=-50, y_vel_ref=-50,
                     next_dist=None))

    for i in range(4, n+4):
        c = colors[np.random.choice(6)]
        y += np.random.choice([60, 100])
        other_cars.append(
            DummyVehicle(id=i, color=c, x=-30.0, y=y, y_vel=-50, y_vel_ref=-50,
                         next_dist=other_cars[-1].m_pos[1]))

    other_cars.append(
        DummyVehicle(id=1, color=green, x=30.0, y=100.0, y_vel=-30, y_vel_ref=-30))
    other_cars.append(
        DummyVehicle(id=2, color=red, x=30.0, y=-60.0, y_vel=-30, y_vel_ref=-30))

    return Game(car, other_cars, learning)

class RL:
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
        total_rew = 0
        q, quit = False, False

        while True:# not game.game_over:

            if n < 10:
                a = int(game.gui.process_keys())
            else:
                a = self.model.sample_action(s, eps)
            prev_s = s

            rew, q = game.move(a)
            s = game.get_state()

            total_rew += rew

            G = rew + gamma * np.max(self.model.predict(s))
            self.model.update(prev_s, a, G)
            self.model.train_tuples = np.append(self.model.train_tuples, np.array([[prev_s, a, G]]), axis=0)

            iters += 1
            quit |= q
            # time.sleep(0.1)
        return total_rew, quit


def plot_running_avg(totalrewards: np.ndarray):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for i in range(N):
        running_avg[i] = totalrewards[max(0, i - 100):(i + 1)].mean()
    plt.plot(running_avg)
    plt.show()

def init_trn():
    trn = np.array([[0, 0, 0]])
    trnt = np.array([trn,0])
    arr, n = np.load('train_tuples.npy')


def main():

    ft = FeatureTransformer()
    model = Model(ft, 'constant', start_over=True)
    rl = RL(model)
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
        totalreward, quit = rl.play_one(i, eps=eps, gamma=gamma)
        totalrewards.append(totalreward)

        print("episode:", i, "total reward:", totalreward, "eps:", eps)
        np.save('models', [np.array(model.models), i])
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
