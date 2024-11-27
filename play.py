import time
import gymnasium as gym
from stable_baselines3 import DQN

WildlifeCorridorEnv = __import__('custEnv').WildlifeCorridorEnv


def main():
    # Initialize environment
    env = WildlifeCorridorEnv()

    # Load model (as you've confirmed it's loading properly)
    model = DQN.load("models/wildlife_corridor_policy.zip")

    obs, info = env.reset()  # Reset the environment and get the initial observation
    
    # Ensure obs is a tuple for handling later in your environment logic
    obs = tuple(obs)

    print("Model loaded successfully!")
    
    done = False
    episode = 1
    while not done:
        print(f"--- Episode {episode} ---")
        
        # Get action from your model (ensure that the action is valid)
        action = model.predict(obs)  # Assuming `model.predict` gives the next action
        
        # Check for any errors in action or observation
        try:
            # Take the step in the environment with the action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Ensure the observation is a tuple if it's being passed back into any set or dict
            obs = tuple(obs)  # Convert to tuple if it's numpy.ndarray

            if terminated or truncated:
                done = True
        
        except TypeError as e:
            print(f"Error encountered: {e}")
            # Handle the TypeError by adjusting agent position if necessary
            obs = tuple(obs)  # Convert to tuple to avoid numpy.ndarray issues
            continue
        
        # Optionally, render the environment if needed
        env.render()

        episode += 1
    
    env.close()

if __name__ == "__main__":
    main()