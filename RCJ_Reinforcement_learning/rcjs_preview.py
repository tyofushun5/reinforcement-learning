import os

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from RCJ_Reinforcement_learning.environment.rcjs_first_environment import FirstEnvironment

save_dir = "model"

def main():
    preview_env = FirstEnvironment(max_steps=1000000,
                                   create_position=[0, 0, 0],
                                   magnitude=10.0,
                                   gui=True)

    model_path = os.path.join(save_dir, "RCJ_ppo_model_v1")
    loaded_model = RecurrentPPO.load(model_path, env=preview_env)

    observation, info = preview_env.reset()
    while True:
        action, _states = loaded_model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = preview_env.step(action)
        if terminated or truncated:
            preview_env.reset()
            break

    preview_env.close()

if __name__ == "__main__":
    main()
