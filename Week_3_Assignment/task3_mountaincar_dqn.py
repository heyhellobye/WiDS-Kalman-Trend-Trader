import gymnasium as gym
import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# ============================
# Q-Network
# ============================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ============================
# Environment
# ============================
env = gym.make("MountainCar-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# ============================
# Hyperparameters
# ============================
gamma = 0.99
lr = 0.001
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
batch_size = 64
memory_size = 50000
episodes = 500

# ============================
# Setup
# ============================
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
criterion = nn.MSELoss()

memory = deque(maxlen=memory_size)

# ============================
# Replay function
# ============================
def replay():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    q_vals = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
    next_q_vals = target_net(next_states).max(1)[0]
    target = rewards + gamma * next_q_vals * (1 - dones)

    loss = criterion(q_vals, target.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ============================
# Training Loop
# ============================
for ep in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = torch.argmax(
                    policy_net(torch.FloatTensor(state))
                ).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        replay()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {ep+1}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

env.close()
