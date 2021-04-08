from gym.envs.registration import register

register(
    id='portfolio-v0',
    entry_point='gym_portfolio.envs:PortfolioEnv',
)
register(
    id='asset-v0',
    entry_point='gym_portfolio.envs:AssetEnv',
    kwargs={'df': None, 'window': 20}
)
