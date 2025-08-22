# BlenderAI Reinforcement Learning

This project integrates a custom Blender environment with Gymnasium and trains it using PPO (Stable-Baselines3).

## Structure
- `blender_env/`: Custom Gym environment for BlenderAI
- `scripts/train_ppo.py`: PPO training script
- `models/`: Saved models
- `requirements.txt`: Dependencies

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Run training: `python scripts/train_ppo.py`
1) blender_worker.py — server Blender (jalankan di Blender)

Simpan sebagai file dan jalankan dengan Blender:
blender -b -P blender_worker.py -- --port 5005 --usd /path/to/scene.usda
