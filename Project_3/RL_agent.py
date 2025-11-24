'''
RL agent for InHand manipulation task.

The agent sees the current observation, the reward, and whether the episode is done, 
and selects an action to take in the environment.

The agent must pick a discrete action based on the current state.
    I need to discretize the continuous action space into bins.

The agent updates the Q-table.


'''
from scipy.spatial.transform import Rotation as R
import numpy as np

class RlAgent:
    def __init__(self, n_yaw_bins=24, n_actions=5,
                 alpha=0.1, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.05, epsilon_decay_steps=200_000):
        
        # Discretization parameters
        self.n_yaw_bins = n_yaw_bins
        self.n_states = n_yaw_bins  # only discretizing yaw angle
        self.n_actions = n_actions

        # alpha, gamma, epsilon for Q-learning
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

        # Q-table initialization
        self.q_table = np.zeros((self.n_yaw_bins, self.n_actions))  # Key: discrete state, Value: action values
        
        # action values are macro-actions mapped to a 16-D continuous action vector
        step = 0.02 # magnitude of finger movement per step
        self.action_values = self._build_action_values(step)

    def _build_action_values(self, step):
        # Build discrete action values
        actions = []

        # 0: no movement
        actions.append(np.zeros(16, dtype=np.float32))

        # 1: tighten all fingers
        a1 = np.zeros(16, dtype=np.float32)
        a1[:] = step
        actions.append(a1)

        # 2: loosen all fingers
        a2 = np.zeros(16, dtype=np.float32)
        a2[:] = -step
        actions.append(a2)

        # 3: twist clockwise
        a3 = np.zeros(16, dtype=np.float32)
        a3[::2] = step  # even indices
        a3[1::2] = -step  # odd indices
        actions.append(a3)

        # 4: twist counter-clockwise
        a4 = np.zeros(16, dtype=np.float32)
        a4[::2] = -step  # even indices
        a4[1::2] = step  # odd indices
        actions.append(a4)

        # we return a list of 16-D action vectors for each discrete action outlined above
        return np.stack(actions, axis=0)

    def _discrete_state(self, obs):
        # we want to take the yaw angle from the observation and map it to a discrete bin
        # to discretize our continuous state space into discrete states for Q-learning
        yaw_deg = obs[21]  # yaw is at index 21
        yaw_deg = (yaw_deg + 360.0) % 360  # Normalize to [0, 360]

        # 24 bins of 15 degrees each
        bin_size = 360.0 / self.n_yaw_bins # bin size of 15 degrees
        yaw_bin = int(yaw_deg // bin_size)
        # yaw bin should be in [0, n_yaw_bins-1] for table indexing
        yaw_bin = max(0, min(self.n_yaw_bins - 1, yaw_bin))  # Clamp to valid range
        return yaw_bin # discrete state index based on yaw angle

    def select_action(self, obs, deterministic=False):
        # Epsilon-greedy action selection (this is so the agent can explore)
        state_idx = self._discrete_state(obs) # given obs from the env, get discrete state index

        # if deterministic then -> select best action (testing)
        # if not deterministic then -> epsilon-greedy (training)
        if deterministic:
            # Select the best action based on Q-values in the current state 
            action_index = np.argmax(self.q_table[state_idx])
        else:
            # epsilon controls how much we want to explore vs exploit
            if np.random.rand() < self.epsilon:
                action_index = np.random.randint(self.n_actions)
            else:
                action_index = np.argmax(self.q_table[state_idx])

        # based on the action index, return the corresponding continuous action vector built with _build_action_values
        action_vec = self.action_values[action_index]

        # we want to return state and the action index for the Q-table update later
        return state_idx, action_index, action_vec


    def update_q_table(self, state, action_index, reward, next_state_idx, done):
        # In the training loop, after taking an action on an observation,
        # we get back the reward and next observation in the `env.step(action)` call.
        # We use that info to update the Q-table.

        # get the best next Q value for the next observation 
        # self._discrete_state(next_obs) gives us an entire row of Q-values
        best_next_q = 0 if done else np.max(self.q_table[next_state_idx])
            # best_next_q is what we think the future is worth from the next state

        # target is the reward plus the discounted best next Q value
        # this tells us what the Q-value should be
        target = reward + self.gamma * best_next_q

        # what is our current Q value for the current state and action (estimate)?
        old_value = self.q_table[state, action_index]

        # Q-learning update rule (equation in slides), alpha is the learning rate
        # target - old value is the TD error (how wrong we were)
        # if the TD error is positive, then we increase the Q-value -> better than we expected
        # if the TD error is negative, then we decrease the Q-value -> worse than we expected
        self.q_table[state, action_index] = old_value + self.alpha * (target - old_value)

        self._update_epsilon()

    def _update_epsilon(self):
        self.total_steps += 1
        # represents how far we are through the decay process
        frac = min(1.0, self.total_steps / self.epsilon_decay_steps)

        # updates our epsilon during learning so that early on we explore more, later on we exploit more        
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)


    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)
        self.n_states, self.n_actions = self.q_table.shape