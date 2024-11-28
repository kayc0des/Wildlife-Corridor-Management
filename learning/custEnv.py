import numpy as np
import gymnasium as gym
from gymnasium import spaces

class WildlifeCorridorEnv(gym.Env):
    """
    Custom Environment for Wildlife Corridor Management.
    The agent represents an animal navigating a grid-based landscape.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        """
        Initialize the environment.
        """
        super(WildlifeCorridorEnv, self).__init__()
        
        self.grid_size = 10
        self.start_pos = (0, 0)
        self.goal_pos = (9, 9)
        
        # Define obstacles (e.g., human settlements, predators)
        self.obstacles = [(3, 3), (4, 4), (5, 5), (2, 8), (8, 2)]
        
        # Define action space
        self.action_space = spaces.Discrete(4)
        
        # Observation space: Agent's position on the grid
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32
        )
        
        # Initialize targets
        self.targets = [self.goal_pos]  # Initialize with one goal for now
        
        # Rendering setup
        self.render_mode = render_mode
        self.agent_pos = np.array(self.start_pos)  # Ensure agent_pos is a numpy array

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)
        self.agent_pos = np.array(self.start_pos)  # Use numpy array for agent position
        return self.agent_pos, {}  # Return numpy array as observation

    def step(self, action):
        """
        Execute one step of the environment based on the agent's action.
        """
        moves = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
        }

        # Perform the movement
        new_pos = self.agent_pos + moves[action]

        # Ensure the new position is within bounds of the grid
        new_pos = np.clip(new_pos, 0, self.grid_size - 1)

        # Check for obstacles or target goals
        if tuple(new_pos) in self.obstacles:
            reward = -1  # Penalize for hitting obstacles
        elif tuple(new_pos) == self.goal_pos:
            reward = 10  # Reward for reaching the goal
            self.goal_pos = None  # Goal reached, set to None
        else:
            reward = -0.1  # Small penalty for each move

        # Update agent position
        self.agent_pos = new_pos

        # Check if the goal has been reached
        terminated = self.goal_pos is None  # Terminated if goal reached
        truncated = False  # No truncation logic for now

        # Return observation, reward, terminated, truncated, info
        return self.agent_pos, reward, terminated, truncated, {}

    def render(self):
        """
        Render the environment (simple text-based representation).
        """
        if self.render_mode == "human":
            grid = [[" " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
            
            # Place obstacles
            for obs in self.obstacles:
                grid[obs[0]][obs[1]] = "X"
            
            # Place agent
            grid[self.agent_pos[0]][self.agent_pos[1]] = "A"
            
            # Place goal
            grid[self.goal_pos[0]][self.goal_pos[1]] = "G"
            
            print("\n".join(["".join(row) for row in grid]))
            print("-" * 20)

    def close(self):
        """
        Close the environment.
        """
        pass
