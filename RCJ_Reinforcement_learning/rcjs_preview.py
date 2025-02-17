import os

from stable_baselines3 import PPO

from RCJ_Reinforcement_learning.environment.rcjs_first_environment import Environment

save_dir = "model"

def main():
    env = Environment(max_epoch=20000, create_position=[0, 0, 0])

    model = PPO("MlpPolicy",
                env,
                device="cpu",
                verbose=1,
                n_epochs=20,
                n_steps=4096,
                batch_size=4096,
                gamma=0.99)

    model.learn(total_timesteps=5000000)

    model.save(os.path.join(save_dir, "RCJ_ppo_model_v1"))

    env.close()


def preview():
    # テスト環境の作成
    test_env = Environment(max_epoch=1000,
                           create_position=[0, 0, 0])

    # 学習済みモデルのロード
    model_path = os.path.join(save_dir, "RCJ_ppo_model_v1")
    loaded_model = PPO.load(model_path, env=test_env)

    # テストの実行
    obs = test_env.reset()
    for _ in range(1000):  # テストのステップ数
        action, _states = loaded_model.predict(obs, deterministic=True)
        obs, rewards, dones, info = test_env.step(action)
        if dones:
            test_env.reset()
            break

    test_env.close()


if __name__ == "__main__":
    main()
    preview()
