import gym

# Create the environment
env = gym.make("HalfCheetah-v4")

# Reset the environment
state = env.reset()
done = False

# Interact with the environment
while not done:
    action = env.action_space.sample()  # Take a random action
    obs, reward, terminated, truncated, info = env.step(action)  # Handle 5 return values
    done = terminated or truncated  # Combine terminated and truncated flags

print("Environment is working correctly!")
