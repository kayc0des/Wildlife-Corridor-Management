import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
import random


class WildlifeCorridorEnv(gym.Env):
    """
    Custom Environment for Wildlife Corridor Management.
    The agent navigates a grid while avoiding obstacles and reaching the goal.
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, grid_size=10, render_mode=None):
        super(WildlifeCorridorEnv, self).__init__()

        self.grid_size = grid_size
        self.render_mode = render_mode

        # Define action space: 0=Up, 1=Right, 2=Down, 3=Left
        self.action_space = spaces.Discrete(4)

        # Define observation space: Grid representation around the agent
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(grid_size, grid_size, 3),  # Grid with layers: agent, obstacles, goal
            dtype=np.float32,
        )

        # Predefined positions
        self.start_pos = (0, 0)  # Agent starts at top-left corner
        self.goal_pos = (grid_size - 1, grid_size - 1)  # Goal is at bottom-right corner
        self.obstacles = {
            (2, 2), (2, 3), (2, 4),
            (5, 5), (5, 6), (5, 7),
            (7, 2), (8, 2), (8, 3),
        }  # Example fixed obstacles

        # Visualization parameters
        self.cell_size = 50  # Size of each grid cell in pixels
        self.window_size = (
            grid_size * self.cell_size,
            grid_size * self.cell_size,
        )  # Window dimensions

        # Tracking agent's behavior
        self.agent_pos = None
        self.recent_positions = []
        self.visited_positions = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Set agent position to the start position
        self.agent_pos = list(self.start_pos)

        # Reset tracking variables
        self.recent_positions = []
        self.visited_positions = {tuple(self.start_pos)}

        return self._get_observation(), {}

    def step(self, action):
        # Define possible moves (Up, Right, Down, Left)
        moves = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1),  # Left
        }

        # Calculate new position based on the action
        new_pos = [
            self.agent_pos[0] + moves[action][0],
            self.agent_pos[1] + moves[action][1],
        ]

        # Ensure the agent stays within the grid bounds
        new_pos[0] = np.clip(new_pos[0], 0, self.grid_size - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.grid_size - 1)

        # Only update the position if it's not an obstacle
        if tuple(new_pos) not in self.obstacles:
            self.agent_pos = new_pos

        # Check if the agent has reached the goal
        done = tuple(self.agent_pos) == self.goal_pos
        distance_to_goal = np.linalg.norm(np.array(self.goal_pos) - np.array(self.agent_pos))

        # Reward based on proximity to the goal
        reward = 0

        # Goal achievement gives a large positive reward
        if done:
            reward = 100  # High reward for reaching the goal
        else:
            # Reward is based on distance to goal (closer = better)
            reward = -distance_to_goal * 0.1

        # Adding penalties to prevent the agent from oscillating (staying in the same place or moving back and forth)
        if self._is_oscillating():
            reward -= 1  # Small penalty for oscillation behavior

        # If the agent moves into a new area (unexplored), reward exploration
        if self._is_new_area():
            reward += 5  # Reward for exploring new areas

        # Return the next state, reward, done flag, and info
        return self._get_observation(), reward, done, False, {}
    
    def _is_oscillating(self):
        # Track recent positions and detect repetitive movement
        self.recent_positions.append(tuple(self.agent_pos))

        # Limit the list size to a fixed number of steps (e.g., 4)
        if len(self.recent_positions) > 4:
            self.recent_positions.pop(0)

        # Check if there are duplicates in recent positions
        return len(set(self.recent_positions)) < len(self.recent_positions)

    def _is_new_area(self):
        # Check if the agent is in a previously unexplored position
        current_position = tuple(self.agent_pos)

        if current_position not in self.visited_positions:
            self.visited_positions.add(current_position)
            return True

        return False
    
    def _get_observation(self):
        """
        Create a grid representation of the environment.
        Layer 1: Agent
        Layer 2: Obstacles
        Layer 3: Goal
        """
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        grid[self.agent_pos[0], self.agent_pos[1], 0] = 1  # Agent layer
        for obs in self.obstacles:
            grid[obs[0], obs[1], 1] = 1  # Obstacle layer
        grid[self.goal_pos[0], self.goal_pos[1], 2] = 1  # Goal layer
        return grid

    def render(self, mode="human"):
        """
        Render the environment using OpenCV.
        """
        # Create a blank frame
        frame = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)

        # Draw grid elements
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                top_left = (y * self.cell_size, x * self.cell_size)
                bottom_right = ((y + 1) * self.cell_size, (x + 1) * self.cell_size)

                if (x, y) in self.obstacles:  # Obstacles
                    color = (0, 255, 0)  # Green
                elif [x, y] == self.agent_pos:  # Agent
                    color = (0, 0, 255)  # Red
                elif (x, y) == self.goal_pos:  # Goal
                    color = (255, 0, 0)  # Blue
                else:  # Empty cell
                    color = (200, 200, 200)  # Light gray

                cv2.rectangle(frame, top_left, bottom_right, color, -1)
                cv2.rectangle(frame, top_left, bottom_right, (50, 50, 50), 1)  # Grid lines

        # Display the frame
        if mode == "human":
            cv2.imshow("Wildlife Corridor Environment", frame)
            cv2.waitKey(1)

    def close(self):
        """
        Cleanup OpenCV windows.
        """
        cv2.destroyAllWindows()

