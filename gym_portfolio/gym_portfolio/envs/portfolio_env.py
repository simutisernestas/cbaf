import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
import numpy as np

INIT_PORTFOLIO_SIZE = 1000
MAX_ACTION_SIZE = INIT_PORTFOLIO_SIZE * 0.1


class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PortfolioEnv, self).__init__()

        self.index = 0
        self.df = pd.read_pickle(
            '/home/ernestas/Desktop/stuff/py/cbaf/awesome.pkl')
        self.row_count = len(self.df.index)
        self.ep_len = 300

        # TODO: normalize these
        # self.df['last_n']
        # self.df['next_step']

        self.observation_space = spaces.Box(
            -np.inf, np.inf,
            shape=self.df['state_features'].shape,
            dtype=np.float32)
        self.action_space = spaces.Tuple(
            (spaces.Discrete(3), spaces.Box(0, MAX_ACTION_SIZE, (1,))))
        self.reward_range = (-np.inf, np.inf)

        self.portfolio_size = INIT_PORTFOLIO_SIZE

    def _get_obs(self):
        # ob = []
        # for s in range(len(self.df['state_features'][self.index])):
        #     ob.append(self.df['state_features'][self.index][s])
        # ob = np.array(ob, dtype=np.float32)

        return self.df['state_features'][self.index]

    def _get_reward(self, action):
        discrete = action[0]
        amount = action[1][0]
        
        if discrete == 0:  # buy
            last_high = self.df['last_n'][self.index][-1]['high']
            next_low = self.df['next_step'][self.index]['low']
            buy_delta = last_high - next_low
        elif discrete == 1:  # hold
            pass
        elif discrete == 2:  # sell
            pass

        self.df['last_n'][self.index][-1]

        return self.portfolio_size - INIT_PORTFOLIO_SIZE

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if self.index >= (self.row_count-1) or self.index < 0:
            self.index = 0
            return np.zeros((512,)), 0, True, {}

        self.index += 1

        episode_over = (self.portfolio_size < (INIT_PORTFOLIO_SIZE * 0.5))

        return self._get_obs(), self._get_reward(), episode_over, {}

    def reset(self):
        while self.index < 0:
            self.index = np.random.randint(0, high=self.row_count-self.ep_len)
            mod = self.index % self.ep_len
            self.index -= mod
        return self._get_obs()

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    env = PortfolioEnv()
    print(env.action_space.sample())
    print(env._get_reward(env.action_space.sample()))
    # print(env.reset())
    # print(env.step(0))
    # obs = env.reset()
    # for i in range(1000):
    #     # action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(0)
    #     env.render()
    #     if dones:
    #         break
    #     print(obs, rewards, dones)
