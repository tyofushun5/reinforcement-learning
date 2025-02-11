import os

from stable_baselines3 import PPO

from rcjs_environment import Environment

save_dir = "model"
os.makedirs(save_dir, exist_ok=True)

def main():
    env = Environment()

    model = PPO("MlpPolicy", env, device="cpu", verbose=1)

    model.learn(total_timesteps=1000000)

    model.save(os.path.join(save_dir, "RCJ_ppo_model"))

    env.close()

if __name__ == "__main__":
    main()