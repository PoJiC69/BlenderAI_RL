import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
from stable_baselines3 import PPO
from blender_env.BlenderGymEnv import BlenderGymEnv

def main():
    env = BlenderGymEnv()

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    model.save("models/blender_ppo")

    obs, _ = env.reset()
    for _ in range(10):
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    main()
