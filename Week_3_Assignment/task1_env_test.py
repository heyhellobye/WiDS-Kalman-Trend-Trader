import gymnasium as gym

env = gym.make("FrozenLake-v1")


state, info = env.reset()
print("Initial state:", state)
print("State space:", env.observation_space)
print("Action space:", env.action_space)

done = False
while not done:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
