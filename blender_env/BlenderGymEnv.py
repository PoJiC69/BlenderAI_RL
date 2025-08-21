import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BlenderGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(BlenderGymEnv, self).__init__()

        # Action space: [move_x, move_y, rotate, scale]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation space: [x, y, rotation, scale]
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)

        self.state = np.zeros(4, dtype=np.float32)
        self.steps = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(4, dtype=np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        self.state = self.state + action
        self.steps += 1

        # Reward sederhana: semakin dekat ke target (5,5,0,1)
        target = np.array([5.0, 5.0, 0.0, 1.0], dtype=np.float32)
        dist = np.linalg.norm(self.state - target)
        reward = -dist

        terminated = dist < 0.5
        truncated = self.steps >= self.max_steps

        return self.state, reward, terminated, truncated, {}

    def render(self):
        print(f"State: {self.state}")
