import time
import gymnasium as gym
from stable_baselines3 import DQN

WildlifeCorridorEnv = __import__('custEnv').WildlifeCorridorEnv


def main():
    """
    Test the trained reinforcement learning model on the Wildlife Corridor environment.
    """
    # Load the custom environment
    env = WildlifeCorridorEnv()
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)

    # Load the trained model
    try:
        model = DQN.load("models/wildlife_corridor_policy.zip")
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: Trained model not found. Make sure to train the model using train.py first.")
        return

    # Test the model
    n_episodes = 5  # Number of episodes to simulate
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        print(f"\n--- Episode {episode + 1} ---")
        while not done:
            # Use the model to predict the best action
            action, _ = model.predict(obs, deterministic=True)

            # Take the action in the environment
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            # Render the environment (visualize the simulation)
            env.render()
            time.sleep(0.1)  # Add delay for better visualization

            # Check if the episode is finished
            done = terminated or truncated

        print(f"Episode {episode + 1} - Total Reward: {total_reward} - Steps Taken: {steps}")

    env.close()


if __name__ == "__main__":
    main()
