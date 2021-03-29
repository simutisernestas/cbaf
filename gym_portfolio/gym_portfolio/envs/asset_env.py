import random
import gym
from gym import spaces
import pandas as pd
import numpy as np
from enum import Enum

INITIAL_ACCOUNT_BALANCE = 0.001
MAX_ACTION_SIZE_FROM_NET = 0.1
MIN_EP_LEN = 60
MAX_EP_LEN = 60*3
HOLD_EPSILON = 0.075

class AssetEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(AssetEnv, self).__init__()

        self.current_step = 0
        self.df = pd.read_pickle(
            '/home/ernestas/Desktop/stuff/py/cbaf/awesome.pkl')
        self.row_count = len(self.df.index)

        his_feature_len = self.df['state_features'][self.current_step] \
            .reshape((-1,)).shape[0]
        resent_steps_len = self.df['last_n'][self.current_step][:-1] \
            .reshape((-1,)).shape[0]

        self.observation_space = spaces.Box(
            -np.inf, np.inf,
            shape=((his_feature_len+resent_steps_len,)),
            dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, (1,))
        # self.action_space = spaces.Tuple( # TODO
        #     (spaces.Discrete(3), spaces.Box(0, MAX_ACTION_SIZE, (1,))))
        self.reward_range = (-np.inf, np.inf)

    def _next_observation(self):
        steps_tail = self.df['last_n'][self.current_step][:-1].reshape(-1)

        his_features = self.df['state_features'][self.current_step]

        return np.append(his_features, steps_tail)

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df["last_n"][self.current_step][-1][2],
            self.df["last_n"][self.current_step][-1][3])

        action = action[0]
        amount = abs(self.net_worth * MAX_ACTION_SIZE_FROM_NET * action)

        if abs(action) < HOLD_EPSILON:
            '''Hold'''
            pass

        elif action > 0:
            '''Buy amount % of balance in shares'''

            # Limit buy amount to held balance
            amount = min(amount, self.balance)

            shares_bought = amount / current_price
            
            self.balance -= amount
            self.shares_held += shares_bought

        elif action < 0:
            '''Sell amount % of shares held'''
            shares_sold = amount / current_price

            # Limit sell to held shared only
            shares_sold = min(shares_sold, self.shares_held)

            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold

        self.net_worth = self.balance + self.shares_held * current_price

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1
        self.ep_len += 1

        reward = self.net_worth - INITIAL_ACCOUNT_BALANCE

        if self.current_step >= (self.row_count) or self.current_step < 0 \
            or self.ep_len > MAX_EP_LEN:
            self.current_step = 0
            return np.zeros((self.observation_space.shape[0],)), reward, True, {}

        done = (self.net_worth <= INITIAL_ACCOUNT_BALANCE * 0.5)

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.ep_len = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(0, self.row_count-MIN_EP_LEN)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print(f'Balance:\t{self.balance}')
        print(f'Profit:\t{profit}')
        print(f'Shared:\t{self.shares_held}')
        print(f'Basis:\t{self.cost_basis}')


if __name__ == '__main__':
    env = AssetEnv()
    env.reset()
    # print(env.action_space.sample())
    # for i in range(10):
    #     print(env.step(env.action_space.sample()))
    # print(env._next_observation())
    # print(len(env._next_observation()))
    # print(env.observation_space.shape)
    for i in range(999999999999):
        action = env.action_space.sample()
        a, b, c, d = env.step(action)
        print(b)
        if c:
            print(f"done {i} reward {b}")
            break
