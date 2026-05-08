import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# 1. Setup Environment
# 'continuous=False' gives the AI 5 buttons: 0:Nothing, 1:Left, 2:Right, 3:Gas, 4:Brake
def make_env():
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False)
    return env

env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4) # AI sees 4 frames to understand speed

# 2. Setup Checkpoints (Saves every 10k steps)
checkpoint_callback = CheckpointCallback(
  save_freq=10000,
  save_path="./logs/",
  name_prefix="rl_model"
)

# 3. Create the Brain
# device="mps" uses your MacBook's GPU
model = PPO(
    "CnnPolicy", 
    env, 
    verbose=1, 
    device="mps", 
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64
)

# 4. Start the Learning
# Start with 100,000 steps. It might take 30-60 mins depending on your Mac.
print("TRAINING STARTED. The car will spin and fail for a long time—this is learning.")
model.learn(total_timesteps=500000, callback=checkpoint_callback)

# 5. Save the final version
model.save("ppo_car_racing_final")
print("Training complete!")