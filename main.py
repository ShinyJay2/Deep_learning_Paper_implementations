# main.py
import gymnasium as gym
import numpy as np
import torch
from agent import DQNAgent
from replay_memory import ReplayMemory
import time

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500  # Adjusted for step-based decay
TARGET_UPDATE = 1000  # Update target network every 1000 steps
MEMORY_CAPACITY = 10000
NUM_EPISODES = 500
MAX_REWARD = 500  # Early stopping threshold (optional)

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

# Environment setup with render_mode specified to avoid warnings
# Use separate environments for training and rendering to prevent conflicts
train_env = gym.make('CartPole-v1', render_mode=None)  # No rendering during training
render_env = gym.make('CartPole-v1', render_mode='human')  # Separate environment for rendering

n_actions = train_env.action_space.n
state_shape = train_env.observation_space.shape

# Initialize agent and replay memory
agent = DQNAgent(state_shape, n_actions).to(device)  # Move agent to device
replay_buffer = ReplayMemory(MEMORY_CAPACITY)

def render_agent(agent, env, episodes=1, sleep_time=0.02, epsilon=0.0):
    """
    Render the agent's performance in the environment.

    Args:
        agent (DQNAgent): The trained DQN agent.
        env (gym.Env): The Gymnasium environment for rendering.
        episodes (int): Number of episodes to render.
        sleep_time (float): Time to sleep between steps for rendering.
        epsilon (float): Epsilon value for action selection (exploration rate).
    """
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # In 'human' mode, rendering is handled automatically during env.step()
            # Hence, no need to call env.render()

            # Select action using the agent with the specified epsilon
            action = agent.select_action(state, epsilon_threshold=epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            time.sleep(sleep_time)  # Control the speed of rendering

        print(f"Render Episode {episode + 1}: Total Reward: {total_reward}")
    # Do not close the rendering environment here to allow further rendering if needed
    # env.close()  # Removed to prevent closing the environment used for rendering

def main():
    """
    Main function to train the DQN agent and render its performance.
    """
    # 1. Render initial performance before training
    print("Rendering initial performance before training...")
    render_agent(agent, render_env, episodes=1, sleep_time=0.02, epsilon=1.0)  # epsilon=1.0 for random actions

    total_steps = 0  # Initialize total steps
    episode_rewards = []  # To track rewards for optional logging or early stopping

    # Training loop
    for episode in range(NUM_EPISODES):
        state, _ = train_env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Calculate epsilon for the current step (step-based decay)
            epsilon_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * total_steps / EPS_DECAY)
            action = agent.select_action(state, epsilon_threshold)
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated

            # Adjust reward for ending the episode early
            if done and episode_reward < 199:
                reward = -1.0

            # Verify the shape of the next state
            next_state_shape = next_state.shape
            if next_state_shape != train_env.observation_space.shape:
                raise ValueError(f"Expected state shape {train_env.observation_space.shape}, but got {next_state_shape}")

            # Push the transition to replay memory
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            # Optimize the model
            agent.optimize_model(replay_buffer, BATCH_SIZE, GAMMA)

            total_steps += 1  # Increment total steps

            # Update the target network periodically based on total steps
            if total_steps % TARGET_UPDATE == 0:
                agent.update_target_network()

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward: {episode_reward:.2f} | Epsilon: {epsilon_threshold:.4f}")

        # Optional: Early stopping if the agent has learned sufficiently
        if episode_reward >= MAX_REWARD:
            print(f"\nEnvironment solved in {episode + 1} episodes with a reward of {episode_reward}!")
            break

        # 2. Render during training at specified intervals
        if (episode + 1) % 100 == 0:
            print(f"\nRendering performance after {episode + 1} episodes of training...")
            render_agent(agent, render_env, episodes=1, sleep_time=0.02, epsilon=0.05)  # Small epsilon for minimal exploration

    # Save the trained model
    torch.save(agent.policy_net.state_dict(), 'dqn_cartpole_model.pth')
    print("\nModel saved as 'dqn_cartpole_model.pth'.")

    # Load the model for evaluation (optional)
    agent.policy_net.load_state_dict(torch.load('dqn_cartpole_model.pth', map_location=device))
    agent.policy_net.eval()

    # 3. Render final performance after training
    print("\nRendering final performance after training...")
    render_agent(agent, render_env, episodes=1, sleep_time=0.02, epsilon=0.0)  # epsilon=0.0 for pure exploitation

    # Close environments
    train_env.close()
    render_env.close()
    print("Training and rendering completed.")

if __name__ == "__main__":
    main()
