import os

from stable_baselines3 import PPO

from RCJ_Reinforcement_learning.environment.rcjs_first_environment import Environment

save_dir = "model"
os.makedirs(save_dir, exist_ok=True)


def main():
    env = Environment(max_epoch=25000, create_position=[0, 0, 0])

    model = PPO("MlpPolicy",
                env,
                device="cuda",
                verbose=1,
                n_epochs=20,
                n_steps=4096,
                batch_size=256,
                gamma=0.99)

    model.learn(total_timesteps=2000000)

    model.save(os.path.join(save_dir, "RCJ_ppo_model"))

    env.close()


if __name__ == "__main__":
    main()
