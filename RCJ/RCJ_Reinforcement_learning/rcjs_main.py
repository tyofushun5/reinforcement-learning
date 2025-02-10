from stable_baselines3 import PPO

from rcjs_environment import Environment

env = Environment([0, 0, 0])

model = PPO("MlpPolicy", env, device="cuda", verbose=1)

model.learn(total_timesteps=1000000)

model.save("RCJ_ppo_model")

env.close()