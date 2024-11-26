import gym
import torch
import numpy as np
from ppo import ActorCritic  # Ensure this is correctly imported based on your project structure
import time

# Hyperparameters (should match those used during training)
ENV_NAME = "HalfCheetah-v4"
MODEL_PATH = "ppo_half_cheetah_final.pt"  # Path to your saved model
SEED = 42

# Device configuration
DEVICE = torch.device("cuda") if torch.cuda.is_available() else (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)

class NormalizedEnv(gym.ObservationWrapper):
    """
    Normalizes observations to have zero mean and unit variance using running estimates.
    This implementation updates mean and std based on incoming observations.
    """
    def __init__(self, env):
        super().__init__(env)
        self.mean = np.zeros(env.observation_space.shape)
        self.var = np.ones(env.observation_space.shape)
        self.count = 1e-4  # To avoid division by zero

    def observation(self, observation):
        # Update running estimates of mean and variance
        self.count += 1
        last_mean = self.mean.copy()
        self.mean += (observation - self.mean) / self.count
        self.var += (observation - last_mean) * (observation - self.mean)
        std = np.sqrt(self.var / self.count)
        std = np.where(std < 1e-8, 1.0, std)  # Prevent division by zero
        return (observation - self.mean) / std

    @property
    def sim(self):
        """Expose the underlying MuJoCo simulation."""
        return self.env.sim

def load_model(model_path, state_dim, action_dim, hidden_dim=512):
    """
    Loads the Actor-Critic model from the specified path.

    Args:
        model_path (str): Path to the saved model.
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        hidden_dim (int): Number of hidden units.

    Returns:
        ActorCritic: Loaded Actor-Critic model.
    """
    model = ActorCritic(state_dim, action_dim, hidden_dim=hidden_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def visualize(model, env, num_episodes=5):
    """
    Runs the environment with the trained model and renders the agent's behavior.

    Args:
        model (ActorCritic): Trained Actor-Critic model.
        env (gym.Env): Gym environment.
        num_episodes (int): Number of episodes to visualize.
    """
    for episode in range(num_episodes):
        state, info = env.reset(seed=SEED + episode)
        done = False
        total_reward = 0
        while not done:
            env.render()  # Render the current state
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action, _, _ = model.get_action(state_tensor, deterministic=True)
            action = action.cpu().numpy().squeeze()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            time.sleep(1/30)  # Adjust sleep time to control rendering speed

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    # Initialize the environment with rendering enabled
    env = NormalizedEnv(gym.make(ENV_NAME, render_mode="human"))  # Use "human" for real-time rendering

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Load the trained model
    model = load_model(MODEL_PATH, state_dim, action_dim)

    try:
        visualize(model, env, num_episodes=5)
    except KeyboardInterrupt:
        print("Visualization interrupted by user.")
    except Exception as e:
        print(f"An error occurred during visualization: {e}")
    finally:
        env.close()
