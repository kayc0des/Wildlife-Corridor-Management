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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger("WildlifeCorridorRL")
    return logger


def setup_directories():
    """
    Create necessary directories for storing models and logs.
    """
    model_dir = "models"
    log_dir = "logs"
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return model_dir, log_dir


def main():
    """
    Train a reinforcement learning agent to navigate the Wildlife Corridor environment.
    """
    logger = setup_logger()
    model_dir, log_dir = setup_directories()

    # Environment parameters
    max_episode_steps = 1000
    total_timesteps = 50000
    env_id = "WildlifeCorridorEnv"

    try:
        logger.info("Initializing environments...")
        
        # Create training and evaluation environments
        env = DummyVecEnv([create_env(env_id, max_episode_steps)])
        eval_env = DummyVecEnv([create_env(env_id, max_episode_steps)])

        # Add monitoring for evaluation environment
        eval_env = VecMonitor(eval_env, filename=f"{log_dir}/eval_monitor", info_keywords=())
        logger.info("Environments initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize environments: {e}")
        return

    try:
        logger.info("Initializing DQN agent...")
        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=5e-4,
            buffer_size=100000,
            learning_starts=2000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            verbose=1,
            tensorboard_log=log_dir
        )
        logger.info("DQN agent initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize DQN agent: {e}")
        return

    # Set up callbacks for training
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=model_dir,
        name_prefix="dqn_wildlife_corridor"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_model",
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    try:
        # Train the agent
        logger.info("Starting training...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True
        )

        # Save the final trained model
        model_path = f"{model_dir}/wildlife_corridor_policy.zip"
        model.save(model_path)
        logger.info(f"Training complete! Model saved at '{model_path}'.")
    except Exception as e:
        logger.error(f"Training failed: {e}")


if __name__ == "__main__":
    main()
