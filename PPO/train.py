import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from ppo import ActorCritic, ppo_update, compute_returns, compute_advantages
import random
import cv2
import time
import os

# Hyperparameters
ENV_NAME = "HalfCheetah-v4"
NUM_EPOCHS = 500
STEPS_PER_EPOCH = 4096
GAMMA = 0.99
PPO_EPOCHS = 3
EPSILON = 0.2
ENTROPY_COEF = 0.01
LEARNING_RATE = 1e-4  # Reduced learning rate for stability
MAX_GRAD_NORM = 0.5
HIDDEN_DIM = 512
EVAL_EVERY = 50
BATCH_SIZE = 64  # For PPO update minibatching
SEED = 42  # For reproducibility

# Render settings
RENDER_INTERVAL = 100  # Steps between renderings
RENDER_DURATION = 5  # Duration to display rendering in seconds

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

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


def collect_trajectories(env, actor_critic, num_steps, gamma, render=False):
    """
    Collects trajectories for training PPO.

    Args:
        env (gym.Env): The environment to interact with.
        actor_critic (ActorCritic): The Actor-Critic network.
        num_steps (int): Number of steps to collect.
        gamma (float): Discount factor.
        render (bool): Whether to render the environment.

    Returns:
        tuple: (states, actions, old_log_probs, returns, total_reward)
    """
    states, actions, rewards, log_probs, dones = [], [], [], [], []
    total_reward = 0
    state, info = env.reset(seed=SEED)  # Set seed for reproducibility
    done = False
    for step in range(num_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        action, log_prob, _ = actor_critic.get_action(state_tensor)  # No deterministic flag here.

        # Ensure correct action shape
        action = action.detach().cpu().numpy().squeeze()

        # Step the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # **Adjust the Camera to Follow the Agent**
        try:
            # Access the agent's x-position from the state
            agent_x_pos = state[0]  # Assuming the first element is the x-position

            # Update the camera's lookat position
            # You may need to adjust 'cam_pos' and 'cam_lookat' based on your environment's camera configuration
            env.sim.model.cam_pos[0] = agent_x_pos + 5.0  # Offset the camera 5 units ahead on the x-axis
            env.sim.model.cam_lookat[0] = agent_x_pos  # Center the camera on the agent's x-position

            # Optional: Adjust camera height and distance if needed
            # env.sim.model.cam_pos[1] = desired_y_position
            # env.sim.model.cam_pos[2] = desired_z_position
        except AttributeError:
            # If the environment does not expose sim, skip camera adjustment
            pass

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob.item())
        dones.append(done)
        total_reward += reward

        # Render the environment if required
        if render and (step % RENDER_INTERVAL == 0):
            frame = env.render()
            if frame is not None:
                # Convert frame to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('HalfCheetah', frame)
                # Display the frame for a short duration
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # Exit if 'q' is pressed

        # Reset the environment if done
        state = next_state if not done else env.reset(seed=SEED)[0]

    # Convert lists to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    log_probs = np.array(log_probs)
    dones = np.array(dones)

    # Compute value estimate for the last state
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, last_value = actor_critic(state_tensor)
        last_value = last_value.item()

    # Compute discounted returns without normalization
    returns = compute_returns(rewards, dones, gamma, last_value, normalize=False)

    return (
        torch.tensor(states, dtype=torch.float32).to(DEVICE),
        torch.tensor(actions, dtype=torch.float32).to(DEVICE),
        torch.tensor(log_probs, dtype=torch.float32).unsqueeze(1).to(DEVICE),
        returns.to(DEVICE),
        total_reward  # Return total reward for logging
    )


def evaluate_policy_deterministic(env, actor_critic):
    """
    Evaluates the deterministic policy (without exploration noise).

    Args:
        env (gym.Env): The environment to evaluate in.
        actor_critic (ActorCritic): The Actor-Critic network.

    Returns:
        float: Total reward obtained during evaluation.
    """
    state, info = env.reset(seed=SEED + 999)  # Different seed for evaluation
    done = False
    total_reward = 0
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        action, _, _ = actor_critic.get_action(state_tensor, deterministic=True)  # Deterministic action
        action = action.detach().cpu().numpy().squeeze()

        # Step the environment
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    return total_reward


def train():
    """
    Main training loop for PPO.
    """
    # Initialize the environment with normalization and render_mode="rgb_array"
    env = NormalizedEnv(gym.make(ENV_NAME, render_mode="rgb_array"))  # Use "rgb_array" for programmatic rendering
    # No need to call env.seed(SEED) since seed is passed to reset()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize Actor-Critic network
    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=LEARNING_RATE)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - (epoch / NUM_EPOCHS))

    train_losses = []
    eval_rewards = []
    deterministic_rewards = []
    avg_rewards = []  # To track average rewards per epoch

    # Initialize variables for rendering control
    render_active = False
    render_start_time = None

    for epoch in range(NUM_EPOCHS):
        # Determine if rendering should be active
        if (epoch + 1) % EVAL_EVERY == 0:
            render_active = True
            render_start_time = time.time()

        # Collect trajectories with rendering if active
        states, actions, old_log_probs, returns, total_reward = collect_trajectories(
            env, actor_critic, STEPS_PER_EPOCH, GAMMA, render=render_active
        )

        # Deactivate rendering after the specified duration
        if render_active and (time.time() - render_start_time >= RENDER_DURATION):
            render_active = False
            cv2.destroyAllWindows()

        # Get value estimates from the Critic
        with torch.no_grad():
            values = actor_critic.critic(states)

        # Compute advantages without normalization
        advantages = compute_advantages(returns, values, normalize=False)

        # PPO Update and Track Loss
        total_loss = ppo_update(
            PPO_EPOCHS,
            states,
            actions,
            old_log_probs,
            returns,
            advantages,
            actor_critic,
            optimizer,
            epsilon=EPSILON,
            entropy_coef=ENTROPY_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            batch_size=BATCH_SIZE
        )
        train_losses.append(total_loss)  # Track training loss

        # Update learning rate
        scheduler.step()

        # Calculate average reward for this epoch
        avg_reward = total_reward / STEPS_PER_EPOCH
        avg_rewards.append(avg_reward)

        # Log total and average reward collected in this epoch
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: Total Reward Collected = {total_reward:.2f}, Average Reward = {avg_reward:.4f}")

        # Evaluate policy every EVAL_EVERY epochs
        if (epoch + 1) % EVAL_EVERY == 0:
            # Create a separate evaluation environment with render_mode="rgb_array"
            eval_env = NormalizedEnv(gym.make(ENV_NAME, render_mode="rgb_array"))  # Use "rgb_array" for programmatic rendering
            eval_reward = evaluate_policy_deterministic(eval_env, actor_critic)
            eval_rewards.append(eval_reward)
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Deterministic Eval Reward: {eval_reward:.2f}")

            # Save model checkpoint
            torch.save(actor_critic.state_dict(), os.path.expanduser(f"~/Desktop/Model_implementation/PPO/ppo_half_cheetah_epoch_{epoch + 1}.pt"))
            print(f"Model checkpoint saved at epoch {epoch + 1}.")

            # Test deterministic policy in training environment
            deterministic_reward = evaluate_policy_deterministic(env, actor_critic)
            deterministic_rewards.append(deterministic_reward)
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Deterministic Training Reward: {deterministic_reward:.2f}")

            eval_env.close()

    # Save final model
    torch.save(actor_critic.state_dict(), os.path.expanduser("~/Desktop/Model_implementation/PPO/ppo_half_cheetah_final.pt"))
    print("Final model saved.")

    # Close any remaining OpenCV windows
    cv2.destroyAllWindows()

    # Plot training and evaluation results
    plt.figure(figsize=(18, 6))

    # Plot Training Loss
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()

    # Plot Average Rewards
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(avg_rewards) + 1), avg_rewards, label="Average Reward")
    plt.xlabel("Epoch")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Per Epoch")
    plt.legend()

    # Plot Evaluation Rewards
    plt.subplot(1, 3, 3)
    eval_epochs = list(range(EVAL_EVERY, NUM_EPOCHS + 1, EVAL_EVERY))
    plt.plot(eval_epochs, eval_rewards, label="Eval Rewards")
    plt.xlabel("Epoch")
    plt.ylabel("Eval Reward")
    plt.title("Evaluation Rewards Over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.expanduser("~/Desktop/Model_implementation/PPO/ppo_half_cheetah_final.pt"))
    plt.show()


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        cv2.destroyAllWindows()
