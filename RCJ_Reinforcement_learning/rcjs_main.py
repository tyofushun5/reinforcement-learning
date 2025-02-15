import os

from stable_baselines3 import PPO

from RCJ_Reinforcement_learning.environment.rcjs_environment import Environment

save_dir = "model"
os.makedirs(save_dir, exist_ok=True)

def main():
    env = Environment()

    model = PPO("MlpPolicy",
                env,
                device="cuda",
                verbose=1,
                n_steps=4096,
                batch_size=256)

    model.learn(total_timesteps=1000000)

    model.save(os.path.join(save_dir, "RCJ_ppo_model"))

    env.close()

if __name__ == "__main__":
    main()