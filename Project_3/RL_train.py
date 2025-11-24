import os
import numpy as np
import time
from inhand_env import CanRotateEnv 
from RL_agent import RlAgent as QLearningAgent

TOTAL_EPISODES = 2000
MAX_STEPS_PER_EPISODE = 300
# not using these currently
LEARNING_RATE = 3e-4
DEVICE = 'cpu' # 'cuda' or 'cpu'

log_dir = "my_agent_logs/"
os.makedirs(log_dir, exist_ok=True)

env = CanRotateEnv(render_mode="headless")
agent = QLearningAgent(n_yaw_bins=24, n_actions=5)

print(agent.q_table.shape)
if agent.q_table.shape != (agent.n_yaw_bins, agent.n_actions):
    raise ValueError(f"Q-table shape mismatch: expected {(agent.n_yaw_bins, agent.n_actions)}, got {agent.q_table.shape}")

for episode in range(TOTAL_EPISODES):
    obs, info = env.reset()
    total_reward = 0.0
    print(f"Episode {episode + 1}/{TOTAL_EPISODES} starting...")

    for step in range(MAX_STEPS_PER_EPISODE):
        # 1. Agent chooses action
        s_idx, a_idx, action_vec = agent.select_action(obs, deterministic=False)

        # 2. Env steps with that action vector
        next_obs, reward, terminated, truncated, info = env.step(action_vec)

        total_reward += reward

        # 3. Discretize next state
        s_next_idx = agent._discrete_state(next_obs)
        # TODO: how is "done" defined in the env.step()?
        done = terminated or truncated

        # 4. Q-learning update
        agent.update_q_table(s_idx, a_idx, reward, s_next_idx, done)

        obs = next_obs

        if done:
            break

    # -- Logging per episode ---
    print(f"Episode {episode + 1} ended with total reward: {total_reward:.2f} in {step + 1} steps.")
    if (episode + 1) % 500 == 0:
        save_path = os.path.join(log_dir, f"q_table_episode_{episode + 1}.npy")
        agent.save(save_path)
        print(f"Saved agent Q-table at episode {episode + 1} to {save_path}")


# --- TODO: Final save and cleanup ---
agent.save(os.path.join(log_dir, "q_table_final.npy"))
env.close()
print("Training finished.")