import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import logging

# Import the custom environment
WildlifeCorridorEnv = __import__('custEnv').WildlifeCorridorEnv

def create_env(env_id, max_steps):
    """
    Factory function to create the custom environment wrapped with TimeLimit.
    """
    def make_env():
        return gym.wrappers.TimeLimit(WildlifeCorridorEnv(), max_episode_steps=max_steps)
    return make_env

def setup_logger():
    """
    Set up a logger for training progress and issues.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("WildlifeCorridorRL")
    return logger

def setup_directories():
    """
    Create necessary directories for storing models and logs.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def main():
    """
    Train a reinforcement learning agent to navigate the Wildlife Corridor environment.
    """
    logger = setup_logger()
    setup_directories()

    # Environment parameters
    max_episode_steps = 100
    total_timesteps = 50000
    env_id = "WildlifeCorridorEnv"

    logger.info("Initializing environments...")
    
    # Create training and evaluation environments
    env = DummyVecEnv([create_env(env_id, max_episode_steps)])
    eval_env = DummyVecEnv([create_env(env_id, max_episode_steps)])

    # Add monitoring for evaluation environment
    eval_env = VecMonitor(eval_env, filename="./logs/eval_monitor", info_keywords=())

    logger.info("Environments initialized successfully.")

    # Initialize the DQN agent
    logger.info("Initializing DQN agent...")
    model = DQN(
        policy="MlpPolicy",
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
    logger.info("DQN agent initialized.")

    # Set up callbacks for training
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

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    # Train the agent
    logger.info("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True
    )

    # Save the final trained model
    model_path = "models/wildlife_corridor_policy.zip"
    model.save(model_path)
    logger.info(f"Training complete! Model saved at '{model_path}'.")

if __name__ == "__main__":
    main()
