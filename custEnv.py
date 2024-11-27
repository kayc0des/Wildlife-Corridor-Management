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
        
        # Rendering setup
        self.render_mode = render_mode
        self.agent_pos = None

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)
        self.agent_pos = list(self.start_pos)
        return np.array(self.agent_pos, dtype=np.int32), {}

    def step(self, action):
        """
        Take a step in the environment.
        """
        # Map actions to movements
        moves = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1),  # Left
        }
        
        # Calculate new position
        new_pos = [
            self.agent_pos[0] + moves[action][0],
            self.agent_pos[1] + moves[action][1],
        ]
        
        # Keep agent within bounds
        new_pos[0] = np.clip(new_pos[0], 0, self.grid_size - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.grid_size - 1)
        
        # Update agent position if not hitting an obstacle
        if tuple(new_pos) not in self.obstacles:
            self.agent_pos = new_pos
        
        # Check if goal is reached
        done = tuple(self.agent_pos) == self.goal_pos
        reward = 10 if done else -1  # Reward for reaching goal; penalty otherwise
        
        # Return step information
        return np.array(self.agent_pos, dtype=np.int32), reward, done, False, {}

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
