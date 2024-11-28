import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
import random


class WildlifeCorridorEnv(gym.Env):
    """
    Custom Environment for Wildlife Corridor Management with randomized initialization
    and environment themes (e.g., forest, desert, water).
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, grid_size=10, render_mode=None, obstacles=None, max_steps=100, 
                 random_start_goal=True, theme="forest"):
        super(WildlifeCorridorEnv, self).__init__()

        self.grid_size = grid_size
        self.render_mode = render_mode
        self.max_steps = max_steps  # Terminate the episode after max_steps
        self.steps_taken = 0  # Step counter
        self.random_start_goal = random_start_goal  # Toggle randomized positions
        self.theme = theme  # Theme of the environment

        # Define action space: 0=Up, 1=Right, 2=Down, 3=Left
        self.action_space = spaces.Discrete(4)

        # Define observation space: Grid representation around the agent
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(grid_size, grid_size, 3),  # Grid with layers: agent, obstacles, goal
            dtype=np.float32,
        )

        # Predefined obstacle positions
        self.obstacles = set(obstacles) if obstacles else {
            (2, 2), (2, 3), (2, 4),
            (5, 5), (5, 6), (5, 7),
            (7, 2), (8, 2), (8, 3),
        }

        # Visualization parameters
        self.cell_size = 50  # Size of each grid cell in pixels
        self.window_size = (
            grid_size * self.cell_size,
            grid_size * self.cell_size,
        )  # Window dimensions

        # Tracking agent's behavior
        self.agent_pos = None
        self.goal_pos = None
        self.recent_positions = []
        self.visited_positions = set()

        # Theme effects
        self.zone_effects = self._initialize_theme_effects()

    def _initialize_theme_effects(self):
        """
        Define theme-specific effects on movement and rewards.
        """
        if self.theme == "forest":
            return {"penalty": 0.2, "reward": 5, "color": (34, 139, 34)}  # Forest green
        elif self.theme == "desert":
            return {"penalty": 0.5, "reward": 3, "color": (237, 201, 175)}  # Sandy brown
        elif self.theme == "water":
            return {"penalty": 1.0, "reward": 2, "color": (0, 105, 148)}  # Water blue
        else:
            return {"penalty": 0, "reward": 0, "color": (200, 200, 200)}  # Default gray

    def _get_observation(self):
        """
        Generate a grid-based observation of the environment.
        Layers:
        - Layer 0: Agent's position (1 at agent's location)
        - Layer 1: Obstacles (1 where obstacles are located)
        - Layer 2: Goal position (1 at goal's location)
        """
        observation = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)

        # Agent layer
        observation[self.agent_pos[0], self.agent_pos[1], 0] = 1

        # Obstacles layer
        for obs in self.obstacles:
            observation[obs[0], obs[1], 1] = 1

        # Goal layer
        observation[self.goal_pos[0], self.goal_pos[1], 2] = 1

        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomized start and goal positions
        if self.random_start_goal:
            self.start_pos = self._get_random_position()
            self.goal_pos = self._get_random_position(exclude=self.start_pos)
        else:
            self.start_pos = (0, 0)
            self.goal_pos = (self.grid_size - 1, self.grid_size - 1)

        # Set agent position to the start position
        self.agent_pos = list(self.start_pos)

        # Reset tracking variables
        self.recent_positions = []
        self.visited_positions = {tuple(self.start_pos)}
        self.steps_taken = 0  # Reset step counter

        return self._get_observation(), {}

    def _get_random_position(self, exclude=None):
        """
        Generate a random position on the grid, avoiding obstacles and optional exclusions.
        """
        while True:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos not in self.obstacles and pos != exclude:
                return pos

    def step(self, action):
        self.steps_taken += 1  # Increment step counter

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
        done = tuple(self.agent_pos) == self.goal_pos or self.steps_taken >= self.max_steps
        distance_to_goal = np.linalg.norm(np.array(self.goal_pos) - np.array(self.agent_pos))

        # Reward based on proximity to the goal and theme effects
        reward = 0
        if tuple(self.agent_pos) == self.goal_pos:
            reward = 100  # High reward for reaching the goal
        elif self.steps_taken >= self.max_steps:
            reward = -10  # Penalty for exceeding maximum steps
        else:
            # Distance-based penalty
            reward = -distance_to_goal * 0.1

            # Apply theme-specific penalties and rewards
            reward -= self.zone_effects["penalty"]

        return self._get_observation(), reward, done, False, {}

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
                else:  # Theme background
                    color = self.zone_effects["color"]

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