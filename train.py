import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

WildlifeCorridorEnv = __import__('custEnv').WildlifeCorridorEnv

def main():
    """
    Train a reinforcement learning agent to navigate the Wildlife Corridor environment.
    """
    # Create directories for saving models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Initialize the custom environment
    def make_env():
        return gym.wrappers.TimeLimit(WildlifeCorridorEnv(), max_episode_steps=100)

    env = DummyVecEnv([make_env])  # Wrap training environment in DummyVecEnv
    eval_env = DummyVecEnv([make_env])  # Wrap evaluation environment in DummyVecEnv

    # Initialize the DQN agent
    model = DQN(
        policy="MlpPolicy",  # Use Multi-Layer Perceptron policy
        env=env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./logs/"
    )

    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./models/",
        name_prefix="dqn_wildlife_corridor"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # Train the agent
    total_timesteps = 50000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    # Save the final trained model
    model.save("models/wildlife_corridor_policy.zip")

    print("Training complete! Model saved as 'wildlife_corridor_policy.zip'.")


if __name__ == "__main__":
    main()
