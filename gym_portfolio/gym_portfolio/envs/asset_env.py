import random
import gym
from gym import spaces
import pandas as pd
import numpy as np
from enum import Enum
import talib

INITIAL_ACCOUNT_BALANCE = 500
INITIAL_ASSET_AMOUNT = 0.01
MAX_ACTION_SIZE_FROM_NET = 1.0
EP_LEN = 60*3
HOLD_EPSILON = 2.0/3.0/2.0  # 2/3/2  # divide [-1;1] range into three parts


class AssetColumns(Enum):
    TIMESTAMP = 'timestamp'
    LOW = 'low'
    HIGH = 'high'
    OPEN = 'open'
    CLOSE = 'close'
    VOLUME = 'volume'

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return str(self.value)


class AssetEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df=None, window=20):
        super(AssetEnv, self).__init__()
        self.window = window
        if not isinstance(df, pd.DataFrame):
            df = pd.read_csv('/home/ernestas/Desktop/stuff/py/cbaf/eggs.csv', names=[
                'timestamp', 'low', 'high', 'open', 'close', 'volume'])[::-1].reset_index()
        self._set_df(df)
        self.action_space = spaces.Box(-1, 1, (1,), dtype="float32")
        self.reward_range = (-np.inf, np.inf)

    def _set_df(self, df):
        for c in AssetColumns:
            if str(c) not in df.columns:
                raise RuntimeError(
                    f"{c} column doesn't exist in the dataframe")

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['month'] = df['datetime'].dt.month
        df['weekday'] = df['datetime'].dt.weekday
        df['hour'] = df['datetime'].dt.hour
        df = df.drop(['datetime', 'timestamp', 'index'], 1)
        df['sma'] = talib.SMA(df['close'], self.window)
        df['rsi'] = talib.RSI(df['close'], self.window)
        df['mom'] = talib.MOM(df['close'], self.window)
        df['bop'] = talib.BOP(df['open'], df['high'], df['low'], df['close'])
        df['aroonosc'] = talib.AROONOSC(df['high'], df['low'], self.window)

        self.min = df.min()
        self.max = df.max()

        self.df = df

        self.row_count = len(self.df.index)
        if self.row_count < EP_LEN:
            raise RuntimeError(f"not enought rows")

        # add balance, asset_held, net_worth at each timestamp
        n_features = (len(df.columns) * self.window) + 3

        self.observation_space = spaces.Box(
            -np.inf, np.inf,
            shape=((n_features,)),
            dtype=np.float32)

    def _next_observation(self):
        features = self.df.loc[self.current_step -
                               self.window+1:self.current_step]
        # features = (features-self.min)/(self.max-self.min)
        features = features.to_numpy().reshape(-1)
        features = np.append(features,
                             [self.balance, self.shares_held, self.net_worth])

        features = (np.array(features)-np.mean(features))/np.std(features)

        assert any(np.isnan(features)) == False, (features,
                                                  self.df.loc[self.current_step:self.current_step + self.window], self.current_step)

        return features

    def _current_asset_price(self):
        # Gaussian between high and low
        mean = (self.df[str(AssetColumns.LOW)][self.current_step] +
                self.df[str(AssetColumns.HIGH)][self.current_step]) / 2
        sigma = (self.df[str(AssetColumns.HIGH)][self.current_step] -
                 self.df[str(AssetColumns.LOW)][self.current_step]) / 2 / 3
        return np.random.normal(mean, sigma, (1,))[0]

    def _take_action(self, action):
        action = action[0]
        amount = abs(self.net_worth * MAX_ACTION_SIZE_FROM_NET * action)

        price = self._current_asset_price()

        if abs(action) < HOLD_EPSILON:
            '''Hold'''
            self.hold_pen += 0.1

        elif action > 0:
            '''Buy amount % of balance in shares'''

            # Limit buy amount to held balance
            amount = min(amount, self.balance)

            shares_bought = amount / price

            self.balance -= amount
            self.shares_held += shares_bought

            self.hold_pen = (self.hold_pen+1) * (shares_bought==0)

        elif action < 0:
            '''Sell amount % of shares held'''
            shares_sold = amount / price

            # Limit sell to held shared only
            shares_sold = min(shares_sold, self.shares_held)

            self.balance += shares_sold * price
            self.shares_held -= shares_sold

            self.hold_pen = (self.hold_pen+1) * (shares_sold==0)

        self.net_worth = self.balance + self.shares_held * price

    def step(self, action):
        if self.df is None:
            raise RuntimeError("No dataframe!")

        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1
        self.ep_len += 1

        # TODO: decay of reward with long hold
        pnl = (self.net_worth - self.entry_net_worth) / self.entry_net_worth
        # reward = self.net_worth - INITIAL_ACCOUNT_BALANCE
        reward = pnl - (self.hold_pen/self.window)

        obs = self._next_observation()

        if self.current_step >= (self.row_count) or self.current_step < 0 \
                or self.ep_len > EP_LEN:
            return obs, reward, True, {}

        done = (self.net_worth <= 0)

        return obs, reward, done, {}

    def reset(self):
        if self.df is None:
            raise RuntimeError("No dataframe!")

        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.shares_held = INITIAL_ASSET_AMOUNT
        self.ep_len = 0
        self.hold_pen = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            self.window*2, self.row_count-EP_LEN)

        self.net_worth = self.balance + self.shares_held * self._current_asset_price()
        self.entry_net_worth = self.net_worth

        return self._next_observation()

    def render(self, mode='human', close=False):
        if self.df is None:
            raise RuntimeError("No dataframe!")

        # Render the environment to the screen
        profit = self.net_worth - self.entry_net_worth
        print(f'Balance:{self.balance}')
        print(f'Profit:{profit}')
        print(f'Asset:{self.shares_held}\n')


if __name__ == '__main__':
    # df = pd.read_csv('C:\\Users\\simut\\Desktop\\stuff\\code\\coinbase\\eggs.csv', names=[
    #                  'timestamp', 'low', 'high', 'open', 'close', 'volume'])[::-1].reset_index()
    df = None
    env = AssetEnv(df)
    env.reset()

    rews = []
    while True:
        for i in range(10000):
            action = env.action_space.sample()
            a, b, c, d = env.step(action)
            env.render()
            rews.append(b)
            if c:
                print(f"done {i} reward {b}")
                print(env._next_observation().shape)
                print(env.observation_space.shape)
                break

        env.reset()
        break
    # import matplotlib.pyplot as plt
    # plt.plot(rews)
    # plt.show()
