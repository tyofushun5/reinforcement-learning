import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from RCJ_Reinforcement_learning.environment.rcjs_first_environment import Environment


def make_env():
    def _init():
        env = Environment(max_steps=15000,
                          create_position=[0, 0, 0],
                          gui=False)
        return env
    return _init

def main():
    save_dir = "model"
    os.makedirs(save_dir, exist_ok=True)

    num_envs = 12
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    model = PPO("MlpPolicy",
                env,
                device="cpu",
                verbose=1,
                n_epochs=15,
                n_steps=2048,
                batch_size=12288,
                gamma=0.99)

    model.learn(total_timesteps=1000000)
    model.save(os.path.join(save_dir, "RCJ_ppo_model_v1"))

    env.close()

if __name__ == "__main__":
    main()
