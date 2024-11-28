import time
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
WildlifeCorridorEnv = __import__('custEnv').WildlifeCorridorEnv


def main():
    """
    Test the trained reinforcement learning model on the Wildlife Corridor environment.
    """
    # Load the custom environment and wrap it in a DummyVecEnv
    # env = DummyVecEnv([lambda: gym.wrappers.TimeLimit(WildlifeCorridorEnv(), max_episode_steps=100)])

    def make_env():
        return WildlifeCorridorEnv(render_mode="human")

    env = DummyVecEnv([make_env])

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
        obs = env.reset()
        total_reward = 0
        done = False
        steps = 0

        print(f"\n--- Episode {episode + 1} ---")
        while not done:
            # Use the model to predict the best action
            action, _ = model.predict(obs, deterministic=True)

            # Take the action in the environment
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]  # Reward is a list in VecEnv
            steps += 1

            # Render the environment (visualize the simulation)
            env.render()
            time.sleep(0.1)  # Add delay for better visualization

        print(f"Episode {episode + 1} - Total Reward: {total_reward} - Steps Taken: {steps}")

    env.close()


if __name__ == "__main__":
    main()
