import numpy as np

WildlifeCorridorEnv = __import__('custEnv').WildlifeCorridorEnv


def test_environment():
    # Create an instance of the environment
    env = WildlifeCorridorEnv()
    
    print("### Environment Initialized ###")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Grid size: {env.grid_size}")
    print(f"Start position: {env.start_pos}")
    print(f"Goal position: {env.goal_pos}")
    print(f"Obstacles: {env.obstacles}")
    
    # Reset the environment
    obs, info = env.reset()
    print("\n### After Reset ###")
    print(f"Observation: {obs}")
    print(f"Type of observation: {type(obs)}")
    print(f"Shape of observation: {obs.shape if isinstance(obs, np.ndarray) else 'N/A'}")
    print(f"Info: {info}")

    # Take a random step
    random_action = env.action_space.sample()
    print("\n### Random Step ###")
    print(f"Random action taken: {random_action}")
    obs, reward, done, truncated, info = env.step(random_action)
    print(f"New Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")
    print(f"Type of new observation: {type(obs)}")
    print(f"Shape of new observation: {obs.shape if isinstance(obs, np.ndarray) else 'N/A'}")

    # Render the environment
    print("\n### Rendering ###")
    env.render()

    # Close the environment
    env.close()

if __name__ == "__main__":
    test_environment()
