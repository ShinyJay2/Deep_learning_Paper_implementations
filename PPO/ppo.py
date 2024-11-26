import torch
import torch.nn as nn
import numpy as np


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic Network without Tanh activation on the actor.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()

        # Define the Actor network (outputs action mean for continuous control tasks).
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
            # Removed Tanh activation
        )

        # Define the Critic network (outputs scalar value estimates).
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single output for state value V(s).
        )

        # Learnable log standard deviation for action distribution.
        # Initialized to a small value to start with low exploration noise.
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)  # Initialize log_std to -0.5 (~exp(-0.5) ≈ 0.6065)

    def forward(self, state):
        """
        Forward pass through Actor-Critic network.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            tuple: (action mean, state value).
        """
        action_mean = self.actor(state)  # Actor network output.
        state_value = self.critic(state)  # Critic network output (value estimate).
        return action_mean, state_value

    def get_action(self, state, deterministic=False):
        """
        Sample an action from the policy distribution or return the mean action deterministically.

        Args:
            state (torch.Tensor): The current state.
            deterministic (bool, optional): Whether to return a deterministic action. Defaults to False.

        Returns:
            tuple: (action, log probability, entropy).
        """
        action_mean, _ = self.forward(state)  # Forward pass to get action mean.

        if deterministic:
            # If deterministic, use the mean action without sampling.
            action = action_mean
            action_log_prob = torch.zeros((state.size(0), 1), device=state.device)  # No log prob for deterministic actions.
            entropy = torch.zeros((state.size(0), 1), device=state.device)  # No entropy for deterministic actions.
        else:
            # Use learnable standard deviation for exploration noise.
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(action_mean, std)  # Gaussian policy.

            # Sample an action from the distribution.
            action = dist.sample()

            # Compute log probability and entropy for the action.
            action_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)  # Sum over action dimensions.
            entropy = dist.entropy().sum(dim=-1, keepdim=True)  # Encourages exploration.

        return action, action_log_prob, entropy


def compute_returns(rewards, dones, gamma, last_value, normalize=False):
    """
    Compute discounted returns for a trajectory, handling episode terminations.

    Args:
        rewards (list or np.ndarray): List of rewards collected during the trajectory.
        dones (list or np.ndarray): List of done flags indicating episode terminations.
        gamma (float): Discount factor.
        last_value (float): The value estimate of the last state (for bootstrapping).
        normalize (bool, optional): Whether to normalize returns for stability. Defaults to False.

    Returns:
        torch.Tensor: Discounted returns.
    """
    returns = []
    R = last_value
    # Compute discounted rewards in reverse order, resetting R when done=True.
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            R = 0  # Reset the return at the end of an episode.
        R = reward + gamma * R
        returns.insert(0, R)  # Add the return at the beginning of the list.
    returns = torch.tensor(returns, dtype=torch.float32)
    if normalize:
        # Normalize returns to have mean 0 and std 1 for numerical stability.
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns.unsqueeze(1)  # Add dimension for compatibility.


def compute_advantages(returns, values, normalize=False):
    """
    Compute advantage estimates.

    Args:
        returns (torch.Tensor): Discounted returns for each state.
        values (torch.Tensor): Value estimates from the Critic.
        normalize (bool, optional): Whether to normalize advantages. Defaults to False.

    Returns:
        torch.Tensor: Advantage estimates.
    """
    # Advantage is the difference between return and value estimate.
    advantages = returns - values  # A_t = G_t - V(s_t)
    if normalize:
        # Normalize advantages to have mean 0 and std 1.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages


def ppo_update(ppo_epochs, states, actions, old_log_probs, returns, advantages, actor_critic, optimizer,
               epsilon=0.2, entropy_coef=0.01, max_grad_norm=0.5, batch_size=64):
    """
    Perform PPO updates on the Actor-Critic network with minibatching.

    Args:
        ppo_epochs (int): Number of optimization epochs for PPO.
        states (torch.Tensor): Tensor of states from trajectories.
        actions (torch.Tensor): Tensor of actions taken.
        old_log_probs (torch.Tensor): Log probabilities of actions under the old policy.
        returns (torch.Tensor): Discounted returns.
        advantages (torch.Tensor): Advantage estimates.
        actor_critic (ActorCritic): The Actor-Critic network.
        optimizer (torch.optim.Optimizer): Optimizer for network parameters.
        epsilon (float, optional): Clipping parameter for PPO. Defaults to 0.2.
        entropy_coef (float, optional): Coefficient for entropy bonus. Defaults to 0.01.
        max_grad_norm (float, optional): Maximum gradient norm for clipping. Defaults to 0.5.
        batch_size (int, optional): Size of minibatches for updates. Defaults to 64.

    Returns:
        float: The average total loss over all minibatches in the last epoch.
    """
    total_loss_value = 0.0  # Variable to accumulate loss over minibatches

    for epoch in range(ppo_epochs):
        # Shuffle the indices for minibatching
        indices = torch.randperm(states.size(0))
        for start in range(0, states.size(0), batch_size):
            end = start + batch_size
            mb_indices = indices[start:end]

            mb_states = states[mb_indices]
            mb_actions = actions[mb_indices]
            mb_old_log_probs = old_log_probs[mb_indices]
            mb_returns = returns[mb_indices]
            mb_advantages = advantages[mb_indices]

            # Forward pass through the Actor-Critic network.
            action_means, state_values = actor_critic(mb_states)
            std = torch.exp(actor_critic.log_std)
            dist = torch.distributions.Normal(action_means, std)

            # Compute new log probabilities for actions.
            new_log_probs = dist.log_prob(mb_actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)

            # Compute probability ratios: r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t).
            ratios = torch.exp(new_log_probs - mb_old_log_probs.detach())

            # Compute surrogate objective components.
            surr1 = ratios * mb_advantages.detach()  # Unclipped objective.
            surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * mb_advantages.detach()  # Clipped objective.

            # Actor loss: Maximize the surrogate objective.
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss: Minimize the mean squared error between returns and value estimates.
            critic_loss = nn.functional.mse_loss(state_values, mb_returns.detach())

            # Entropy bonus: Encourage exploration by maximizing entropy.
            entropy_loss = entropy.mean()

            # Total loss: Combine actor, critic, and entropy losses.
            total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy_loss

            # Backpropagation and gradient clipping.
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
            optimizer.step()

            # Accumulate loss
            total_loss_value += total_loss.item()

    # Compute average loss over all updates in the last epoch
    avg_loss = total_loss_value / (ppo_epochs * (states.size(0) // batch_size + 1))
    return avg_loss
