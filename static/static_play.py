import time
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import logging

# Import the custom environment
WildlifeCorridorEnv = __import__('static_custEnv').WildlifeCorridorEnv

def setup_logger():
    """
    Set up a logger for testing progress and issues.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("WildlifeCorridorPlay")
    return logger

def create_env(render_mode="human"):
    """
    Factory function to create the custom environment.
    """
    return WildlifeCorridorEnv(render_mode=render_mode)

def load_model(model_path):
    """
    Load the trained model from the specified path.
    """
    try:
        model = DQN.load(model_path)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Trained model not found at '{model_path}'. Please train the model using train.py.")

def test_model(env, model, n_episodes, logger):
    """
    Test the trained model in the environment for a given number of episodes.
    """
    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        steps = 0

        logger.info(f"\n--- Starting Episode {episode + 1} ---")
        while not done:
            # Predict the action
            action, _ = model.predict(obs, deterministic=True)

            # Take the action in the environment
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]  # Reward is a list in VecEnv
            steps += 1

            # Render the environment
            env.render()
            time.sleep(0.1)  # Add delay for better visualization

        logger.info(f"Episode {episode + 1} completed: Total Reward = {total_reward}, Steps Taken = {steps}")

def main():
    """
    Test the trained reinforcement learning model on the Wildlife Corridor environment.
    """
    logger = setup_logger()

    # Configuration
    model_path = "models/best_model/best_model.zip"
    n_episodes = 5  # Number of episodes to simulate
    render_mode = "human"

    logger.info("Initializing environment...")
    # Create and wrap the environment
    env = DummyVecEnv([lambda: create_env(render_mode)])

    logger.info("Loading trained model...")
    try:
        model = load_model(model_path)
        logger.info("Model loaded successfully!")
    except FileNotFoundError as e:
        logger.error(e)
        env.close()
        return

    logger.info("Starting testing...")
    test_model(env, model, n_episodes, logger)

    env.close()
    logger.info("Testing complete. Environment closed.")

if __name__ == "__main__":
    main()