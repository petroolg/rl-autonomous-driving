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
        observation_examples = np.hstack((np.random.random((N, 1)) * 40.0 - 20.0,  # vx
                                          -np.random.random((N, 1)) * 120.0 + 20.0,  # vy
                                          np.random.random((N, 1)) * 80.0 - 40.0,  # x
                                          -np.random.random((N, 1)) * 100.0,  # vy1
                                          np.random.random((N, 1)) * 200.0 - 100.0,  # d1
                                          np.ones((N, 1)) * 30,  # line1
                                          -np.random.random((N, 1)) * 100.0,  # vy2
                                          np.random.random((N, 1)) * 200.0 - 100.0,  # d2
                                          np.ones((N, 1)) * 30,  # line2
                                          -np.random.random((N, 1)) * 100.0,  # vy3
                                          np.random.random((N, 1)) * 200.0 - 100.0,  # d3
                                          -np.ones((N, 1)) * 30,  # line3
                                          -np.random.random((N, 1)) * 100.0,  # vy4
                                          np.random.random((N, 1)) * 200.0 - 100.0,  # d4
                                          -np.ones((N, 1)) * 30  # line4
                                          ))
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma=0.05, n_components=1000)),
            ('rbf2', RBFSampler(gamma=0.1, n_components=1000)),
            ('rbf3', RBFSampler(gamma=0.5, n_components=1000)),
            ('rbf4', RBFSampler(gamma=1.0, n_components=1000))])

        examples = featurizer.fit_transform(scaler.transform(observation_examples))
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, obs):
        scaled = self.scaler.transform(obs)
        return self.featurizer.transform(scaled)


class Model:
    def __init__(self, env, feature_transformer: FeatureTransformer, learning_rate=0.1, start_over=True):
        self.env = env
        env.init_game(learning=False)
        game = env.game
        game.neighbours()
        s = game.get_state()

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
        self.models[a].partial_fit(x, [G])

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return np.random.choice([-1, 0, 1])
        else:
            pred = self.predict(s)
            return np.argmax(pred)


class RL:
    def __init__(self):
        self.GAMMA = 0.9

    def init_game(self, learning=True):
        car = Vehicle(y_vel=-30, x=30)
        other_cars = []
        other_cars.append(
            DummyVehicle(id=1, width=10, length=20, color=green, x=30.0, y=100.0, y_vel=-30, y_vel_ref=-30, line_ref=1))
        other_cars.append(
            DummyVehicle(id=2, width=10, length=20, color=red, x=30.0, y=-60.0, y_vel=-30, y_vel_ref=-30, line_ref=1))
        other_cars.append(
            DummyVehicle(id=4, width=10, length=20, color=blue, x=-30.0, y=400.0, y_vel=-50, y_vel_ref=-50, line_ref=2))
        other_cars.append(
            DummyVehicle(id=3, width=10, length=20, color=red, x=-30.0, y=240.0, y_vel=-50, y_vel_ref=-50, line_ref=2))
        other_cars.append(
            DummyVehicle(id=5, width=10, length=20, color=green, x=-30.0, y=480.0, y_vel=-50, y_vel_ref=-50,
                         line_ref=2))

        self.game = Game(car, other_cars, learning)

    def random_action(self, a_i, eps=0.1):
        fort = np.random.random()
        if fort < (1 - eps):
            return self.game.all_actions[a_i]
        else:
            poss_act = self.game.all_actions.copy()
            # if windy:
            #     poss_act.remove(a)
            return np.random.choice(poss_act)


def play_one(model, eps=0.05, gamma=0.99):
    model.env.init_game(learning=False)
    game = model.env.game
    game.neighbours()
    s = game.get_state()
    a = model.sample_action(s, 0.05)

    game.game_over = False
    iters = 0
    total_rew = 0
    while not game.game_over:
        a = model.sample_action(s, eps)
        prev_s = s

        rew = game.move(a)
        game.neighbours()
        s = game.get_state()

        total_rew += rew

        G = rew + gamma * np.max(model.predict(s))
        model.update(prev_s, a, G)

        iters += 1
        # time.sleep(0.1)
    return total_rew


def plot_running_avg(totalrewards: np.ndarray):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for i in range(N):
        running_avg[i] = totalrewards[max(0, i - 100):(i + 1)].mean()
    plt.plot(running_avg)
    plt.show()


def main():
    env = RL()
    ft = FeatureTransformer()
    model = Model(env, ft, 'constant', start_over=True)

    N = 300
    k = model.K
    totalrewards = np.empty(N)
    gamma = 0.99

    for i in range(k, N + k):
        eps = 0.5 / np.sqrt(i + 1)
        # eps = 0
        totalreward = play_one(model, eps=eps, gamma=gamma)
        totalrewards[i - k] = totalreward

        print("episode:", i, "total reward:", totalreward, "eps:", eps)
        np.save('models', [np.array(model.models), i])
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", totalrewards.sum())

    plt.plot(totalrewards)
    plt.title('Rewards')
    plt.show()

    plot_running_avg(totalrewards)


if __name__ == '__main__':
    main()
