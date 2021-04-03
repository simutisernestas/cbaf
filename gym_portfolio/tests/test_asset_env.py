import unittest
import gym
import pandas as pd
import numpy as np
from gym_portfolio.envs.asset_env import AssetEnv, AssetColumns
from gym_portfolio.envs.asset_env import INITIAL_ACCOUNT_BALANCE, MAX_ACTION_SIZE_FROM_NET

# TODO: util for printing df data
# for k in range(100):
#     print(df.loc[k,'last_n'][-1][0], df.loc[k,'last_n'][-1][1])


class TestStringMethods(unittest.TestCase):

    # TODO: ???
    # def setUp(self):
    #     self.env.reset()

    @classmethod
    def setUpClass(cls):
        "Hook method for setting up class fixture before running tests in the class."
        cls.env = gym.make('gym_portfolio:asset-v0')

    def test_env_doesnt_step_without_dataframe(self):
        self.env = gym.make('gym_portfolio:asset-v0')
        with self.assertRaises(RuntimeError):
            self.env.step(self.env.action_space.sample())

    def test_env_doesnt_reset_without_dataframe(self):
        self.env = gym.make('gym_portfolio:asset-v0')
        with self.assertRaises(RuntimeError):
            self.env.reset()

    def test_env_doesnt_render_without_dataframe(self):
        self.env = gym.make('gym_portfolio:asset-v0')
        with self.assertRaises(RuntimeError):
            self.env.render()

    def test_step_without_reset_throws_error(self):
        # TODO:
        self.assertTrue(True)

    def make_empty_asset_history_df(self):
        d = {}
        for c in AssetColumns:
            d[c.value] = []
        return pd.DataFrame(data=d)

    def make_increasing_sequence(self):
        df = self.make_empty_asset_history_df()
        last_n = np.zeros((11, 5))
        # [ low, high, open, close, volume ]
        for i, step in enumerate(last_n):
            step[0] = i * 1.2
            step[1] = i * 1.3
            step[2] = i * 1.2
            step[3] = i * 1.3
            step[4] = 0.5
        for j in range(100):
            last_n[-1][0] = last_n[-1][0]*1.12
            last_n[-1][1] = last_n[-1][1]*1.12
            last_n[-1][2] = last_n[-1][2]*1.12
            last_n[-1][3] = last_n[-1][3]*1.12
            df.loc[j] = [np.ones((50,)), np.copy(last_n)]
        return df

    def test_appreciating_asset_continous_buy_results_in_profit(self):
        df = self.make_increasing_sequence()
        self.env._set_df(df)
        self.env.reset()
        start_index = self.env.current_step
        self.assertEqual(self.env.balance, INITIAL_ACCOUNT_BALANCE)
        while self.env.balance > 0:
            _, _, done, _ = self.env.step([1])
            if done:
                break
        self.assertTrue(self.env.balance < INITIAL_ACCOUNT_BALANCE)
        self.assertTrue(self.env.net_worth > INITIAL_ACCOUNT_BALANCE)

        end_index = self.env.current_step

        start_low = self.env.df.loc[start_index, 'last_n'][-1][0]
        end_low = self.env.df.loc[end_index, 'last_n'][-1][0]

        ration = end_low/start_low
        bid = INITIAL_ACCOUNT_BALANCE * MAX_ACTION_SIZE_FROM_NET
        growth_on_const_buy_amount = ration * bid

        self.assertTrue(growth_on_const_buy_amount < (
            self.env.net_worth - INITIAL_ACCOUNT_BALANCE))

    def test_appreciating_asset_buy_profit_is_correct(self):
        df = self.make_increasing_sequence()
        self.env._set_df(df)
        self.env.reset()
        self.assertEqual(self.env.balance, INITIAL_ACCOUNT_BALANCE)
        self.env.step([1])
        self.assertTrue(self.env.net_worth == INITIAL_ACCOUNT_BALANCE,
                        f"{self.env.net_worth} == {INITIAL_ACCOUNT_BALANCE}")
        self.env.step([1])
        self.assertTrue(self.env.net_worth > INITIAL_ACCOUNT_BALANCE,
                        f"{self.env.net_worth} > {INITIAL_ACCOUNT_BALANCE}")

        end_index = self.env.current_step - 1

        current_price_bottom = self.env.df.loc[end_index, 'last_n'][-1][0]
        current_price_top = self.env.df.loc[end_index, 'last_n'][-1][1]

        range_bottom = self.env.shares_held * current_price_bottom + self.env.balance
        range_top = self.env.shares_held * current_price_top + self.env.balance

        self.assertTrue(self.env.net_worth > range_bottom,
                        f"{self.env.net_worth} > {range_bottom}")
        self.assertTrue(self.env.net_worth < range_top,
                        f"{self.env.net_worth} < {range_top}")

    def make_decreasing_sequence(self):
        df = self.make_empty_asset_history_df()
        last_n = np.zeros((11, 5))
        # [ low, high, open, close, volume ]
        for i, step in enumerate(last_n):
            step[0] = i * 0.7
            step[1] = i * 0.9
            step[2] = i * 0.7
            step[3] = i * 0.9
            step[4] = 0.5
        for j in range(100):
            last_n[-1][0] = last_n[-1][0]*0.7
            last_n[-1][1] = last_n[-1][1]*0.7
            last_n[-1][2] = last_n[-1][2]*0.7
            last_n[-1][3] = last_n[-1][3]*0.7
            df.loc[j] = [np.ones((50,)), np.copy(last_n)]
        return df

    def test_depreciating_asset_continous_buy_results_in_loss(self):
        df = self.make_decreasing_sequence()
        self.env._set_df(df)
        self.env.reset()
        start_index = self.env.current_step
        self.assertEqual(self.env.balance, INITIAL_ACCOUNT_BALANCE)
        while self.env.balance > 0:
            _, _, done, _ = self.env.step([1])
            if done:
                break
        self.assertTrue(self.env.balance < INITIAL_ACCOUNT_BALANCE)
        self.assertTrue(self.env.net_worth < INITIAL_ACCOUNT_BALANCE)

        end_index = self.env.current_step

        start_low = self.env.df.loc[start_index, 'last_n'][-1][0]
        end_low = self.env.df.loc[end_index, 'last_n'][-1][0]

        ration = end_low/start_low
        bid = INITIAL_ACCOUNT_BALANCE * MAX_ACTION_SIZE_FROM_NET
        loss_on_const_buy_amount = ration * bid

        self.assertTrue(loss_on_const_buy_amount > (
            self.env.net_worth - INITIAL_ACCOUNT_BALANCE))

    def test_depreciating_asset_buy_loss_is_correct(self):
        df = self.make_decreasing_sequence()
        self.env._set_df(df)
        self.env.reset()
        self.assertEqual(self.env.balance, INITIAL_ACCOUNT_BALANCE)
        self.env.step([1])
        self.assertTrue(self.env.net_worth == INITIAL_ACCOUNT_BALANCE,
                        f"{self.env.net_worth} == {INITIAL_ACCOUNT_BALANCE}")
        self.env.step([1])
        self.assertTrue(self.env.net_worth < INITIAL_ACCOUNT_BALANCE,
                        f"{self.env.net_worth} < {INITIAL_ACCOUNT_BALANCE}")

        end_index = self.env.current_step - 1

        current_price_bottom = self.env.df.loc[end_index, 'last_n'][-1][0]
        current_price_top = self.env.df.loc[end_index, 'last_n'][-1][1]

        range_bottom = self.env.shares_held * current_price_bottom + self.env.balance
        range_top = self.env.shares_held * current_price_top + self.env.balance

        self.assertTrue(self.env.net_worth > range_bottom,
                        f"{self.env.net_worth} > {range_bottom}")
        self.assertTrue(self.env.net_worth < range_top,
                        f"{self.env.net_worth} < {range_top}")

    def test_appreciating_asset_sell_results_in_profit(self):
        df = self.make_increasing_sequence()
        self.env._set_df(df)
        self.env.reset()

        # Buy to possess some asset
        self.assertEqual(self.env.balance, INITIAL_ACCOUNT_BALANCE)
        self.env.step([1])
        self.assertTrue(self.env.net_worth == INITIAL_ACCOUNT_BALANCE,
                        f"{self.env.net_worth} == {INITIAL_ACCOUNT_BALANCE}")
        self.env.step([1])
        self.assertTrue(self.env.net_worth > INITIAL_ACCOUNT_BALANCE,
                        f"{self.env.net_worth} > {INITIAL_ACCOUNT_BALANCE}")

        net_worth_before_sell = self.env.net_worth

        # Sell increased asset
        self.env.step([-1])
        self.env.step([-1])

        # Net worth should increase
        self.assertTrue(net_worth_before_sell < self.env.net_worth,
                        f"{net_worth_before_sell} < {self.env.net_worth}")

    # TODO: test_hold
    # TODO: test_sell_depreciating
    # TODO: test_buy_sell_appreciating
    # TODO: test_buy_sell_depreciating
    # TODO: test_sell_buy_appreciating
    # TODO: test_sell_buy_depreciating
    # TODO: test_sell_buy_sell_maybe
    # TODO: test_buy_sell_buy_maybe


if __name__ == '__main__':
    unittest.main()
