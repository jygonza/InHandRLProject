"""Simple random-action agent for the In-hand rotation environment.

Usage (PowerShell):
  python .\random_agent.py --episodes 3 --steps 200 --no-render

By default this will create the `CanRotateEnv` from `inhand_env.py`, reset it,
and repeatedly sample random actions from env.action_space and step the env.

Notes:
 - Rendering uses the env's `render_mode='human'` when --no-render is not set.
 - You can seed the action sampling with --seed for reproducibility.
"""

import argparse
import time
import numpy as np

from inhand_env import CanRotateEnv


def run_random(episodes: int, max_steps: int, render: bool, seed: int | None):
    # Create the environment. Use 'human' render mode to show viewer if requested.
    env = CanRotateEnv(render_mode='human' if render else None)

    # Seed RNGs for reproducibility when provided
    if seed is not None:
        np.random.seed(seed)
        try:
            env.action_space.seed(seed)
        except Exception:
            pass

    try:
        for ep in range(episodes):
            obs, info = env.reset()
            total_reward = 0.0

            for step in range(max_steps):
                # Sample a random action from the action space
                action = env.action_space.sample()

                # Step the environment
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)

                # Optionally render. Viewer is handled inside the env when render_mode='human'.
                if render:
                    # small sleep so viewer has time to update; adjust for visual speed
                    time.sleep(0.02)

                if terminated or truncated:
                    break

            print(f"Episode {ep+1}/{episodes} — steps={step+1}, total_reward={total_reward:.3f}")

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--steps', type=int, default=400, help='Max steps per episode')
    parser.add_argument('--no-render', dest='render', action='store_false', help='Disable rendering')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')

    args = parser.parse_args()

    run_random(episodes=args.episodes, max_steps=args.steps, render=args.render, seed=args.seed)
