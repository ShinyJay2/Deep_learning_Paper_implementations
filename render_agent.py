# render_agent.py
import gymnasium as gym
import numpy as np
import torch
from agent import DQNAgent
import time

def render_agent(agent, env, episodes=5, sleep_time=0.02, epsilon=0.0):
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
        try:
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                # Select action using the agent with the specified epsilon
                action = agent.select_action(state, epsilon_threshold=epsilon)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward
                time.sleep(sleep_time)  # Control the speed of rendering

            print(f"Render Episode {episode + 1}: Total Reward: {total_reward}")
        except Exception as e:
            print(f"Rendering encountered an error: {e}")
            break
    # Do not close the rendering environment here to allow further rendering if needed

def main():
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
    render_env = gym.make('CartPole-v1', render_mode='human')  # 'human' mode for rendering

    n_actions = render_env.action_space.n
    state_shape = render_env.observation_space.shape

    # Initialize agent
    agent = DQNAgent(state_shape, n_actions).to(device)  # Move agent to device

    # Load the trained model
    model_path = 'dqn_cartpole_model.pth'  # Ensure this path is correct
    try:
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
        agent.policy_net.eval()
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file '{model_path}' not found. Please train the agent first.")
        return

    # Render episodes
    print("Rendering trained agent...")
    render_agent(agent, render_env, episodes=190, sleep_time=0.02, epsilon=0.0)  # epsilon=0.0 for pure exploitation

    # Close the rendering environment
    render_env.close()
    print("Rendering completed.")

if __name__ == "__main__":
    main()
