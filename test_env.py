WildlifeCorridorEnv = __import__('custEnv').WildlifeCorridorEnv

if __name__ == "__main__":
    # Test the environment rendering
    env = WildlifeCorridorEnv(grid_size=10, num_obstacles=15, render_mode="human")
    obs, _ = env.reset()

    for _ in range(50):  # Simulate some steps
        action = env.action_space.sample()  # Random action
        obs, reward, done, _, _ = env.step(action)
        env.render()

        if done:
            print("Reached the goal!")
            break

    env.close()