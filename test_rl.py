import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# Setup the same env structure as training
env = DummyVecEnv([lambda: gym.make("CarRacing-v3", render_mode="human", continuous=False)])
env = VecFrameStack(env, n_stack=4)

# Load the model (Change name if loading a checkpoint from /logs/)
model = PPO.load("ppo_car_racing_final", device="mps")

obs = env.reset()
print("AI is now driving. Watch its progress!")

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()