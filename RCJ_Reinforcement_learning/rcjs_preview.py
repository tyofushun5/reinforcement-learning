import os

from stable_baselines3 import PPO

from RCJ_Reinforcement_learning.environment.rcjs_first_environment import Environment

save_dir = "model"

def main():
    preview_env = Environment(max_steps=10000,
                           create_position=[0, 0, 0],
                              gui=True)

    model_path = os.path.join(save_dir, "RCJ_ppo_model_v1")
    loaded_model = PPO.load(model_path, env=preview_env)

    # テストの実行
    obs, info = preview_env.reset()
    while True:
        action, _states = loaded_model.predict(obs, deterministic=True)
        observation, reward, terminated, truncated, info = preview_env.step(action)
        if terminated:
            preview_env.reset()
            break

    preview_env.close()


if __name__ == "__main__":
    main()
