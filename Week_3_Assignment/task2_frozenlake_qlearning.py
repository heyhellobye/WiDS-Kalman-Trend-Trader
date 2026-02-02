import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1")

n_states = env.observation_space.n
n_actions = env.action_space.n

Q = np.zeros((n_states, n_actions))

# ---------------- PARAMETERS ----------------
alpha = 0.1          # learning rate
gamma = 0.99         # discount factor
epsilon = 1.0        # initial exploration
epsilon_min = 0.01
epsilon_decay = 0.999
episodes = 20000
# --------------------------------------------

success = []

for ep in range(episodes):
    state, _ = env.reset()
    done = False
    ep_success = 0

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state
        ep_success = reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    success.append(ep_success)

env.close()

print("Training finished")
print("Final epsilon:", epsilon)
print("Average success rate:", np.mean(success[-1000:]))
