import gym
from stable_baselines3 import PPO
import pandas as pd
from gym_portfolio.envs.asset_env import EP_LEN

if __name__ == '__main__':
    df = pd.read_csv('/home/ernestas/Desktop/stuff/py/cbaf/eggs.csv', names=[
        'timestamp', 'low', 'high', 'open', 'close', 'volume'])[::-1].reset_index()

    # TODO: generate playground data
    #       more real data
    #       visual feedback might help

    WINDOW = 10
    env = gym.make('gym_portfolio:asset-v0', df=df, window=WINDOW)
    model = PPO('MlpPolicy', env, verbose=1, batch_size=128,  # n_steps=EP_LEN,
                gamma=0.98,
                learning_rate=1.9600225183586454e-05,
                ent_coef=1.9359447498724613e-08,
                clip_range=0.1,
                n_epochs=20,
                gae_lambda=0.9,
                max_grad_norm=1,
                vf_coef=0.25377376563703613,
                use_sde=False)

    for i in range(1000):
        model.learn(total_timesteps=1e4)
        obs = env.reset()
        for j in range(EP_LEN):
            action, _state = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            print(f"Action: {action}")
            env.render()
            if done:
                obs = env.reset()

    import time
    model.save(f"awesome_model_{WINDOW}_{int(time.time())}")

# Trial 3 finished with value: 0.004644200000000003 and parameters: {'batch_size': 128, 'n_steps': 256, 'gamma': 0.95,
# 'lr': 0.25580840116395, 'ent_coef': 0.03314308769479659, 'clip_range': 0.4, 'n_epochs': 20, 'gae_lambda': 0.92,
# 'max_grad_norm': 0.5, 'vf_coef': 0.45872598424507405, 'net_arch': 'medium', 'activation_fn': 'tanh'}. Best is trial 3 with value: 0.004644200000000003.

# [I 2021-04-04 13:56:53,376] Trial 3 finished with value: 0.21484999999999993 and parameters: {'batch_size': 32, 'n_steps': 8, 'gamma': 0.98,
#  'lr': 1.9600225183586454e-05, 'ent_coef': 1.9359447498724613e-08, 'clip_range': 0.1, 'n_epochs': 5, 'gae_lambda': 0.9, 'max_grad_norm': 1,
#  'vf_coef': 0.25377376563703613, 'net_arch': 'medium', 'activation_fn': 'tanh'}. Best is trial 3 with value: 0.21484999999999993.