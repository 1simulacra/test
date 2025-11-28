import gymnasium as gym
import time
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make("Pendulum-v1", render_mode="human")

# Instantiate the agent
model = DDPG("MlpPolicy", env, verbose=1, learning_rate=1e-3, batch_size=100)
# Train the agent and display a progress bar
model.learn(total_timesteps=int(1e4), progress_bar=True)
# Save the agent
model.save("dqn_lunar")
del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = DDPG.load("dqn_lunar", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
for i in range(2):
    obs = vec_env.reset()
    dones = False
    while not dones:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        time.sleep(0.1)
        vec_env.render("human")