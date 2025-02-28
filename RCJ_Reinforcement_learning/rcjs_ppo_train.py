import os

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from RCJ_Reinforcement_learning.environment.rcjs_first_environment import FirstEnvironment


save_dir = "model"
os.makedirs(save_dir, exist_ok=True)

def main():
    # 環境の作成
    env = FirstEnvironment(max_steps=10000,
                           create_position=[0, 0, 0],
                           magnitude=10.0,
                           gui=True)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        device="cpu",
        verbose=1,
        n_epochs=10,
        n_steps=4096,
        batch_size=4096,
        gamma=0.99,
        policy_kwargs={"lstm_hidden_size": 256}
    )

    model.learn(total_timesteps=5000000)

    model.save(os.path.join(save_dir, "RCJ_ppo_model_v1"))

    env.close()

if __name__ == "__main__":
    main()
