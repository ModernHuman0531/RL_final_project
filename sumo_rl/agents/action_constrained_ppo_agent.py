"""
Action-Constrained PPO Agent for Traffic Signal Control.

Core idea:
- Policy network proposes actions from pi(a|s).
- Actions are sampled only from feasible set C(s) via action masking.
- Unfeasible actions are masked out (set logits to -1e9) before sampling.
- Sampled action is guaranteed to be feasible.
- GAE estimates advantages with TD errors.
- PPO updates the policy on clipped surrogate objective.

Constraints enforced:
- Minimum green time
- Maximum green time
- Yellow phase transitions
- Pedestrian safety (inherited from env.get_valid_actions())

Guarantees: 0% constraint violations (action masking ensures only feasible actions are sampled).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque


class ActorNetwork(nn.Module):
    """
    Actor network: maps state -> action logits over discrete actions.
    
    State shape: (batch, state_dim)
    Output shape: (batch, num_actions)
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        """
        Args:
            state: Tensor of shape (batch, state_dim) or (state_dim,)
        Returns:
            logits: Tensor of shape (batch, action_dim) or (action_dim,)
        """
        return self.net(state)


class CriticNetwork(nn.Module):
    """
    Critic network: maps state -> value estimate.
    
    State shape: (batch, state_dim)
    Output shape: (batch, 1)
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """
        Args:
            state: Tensor of shape (batch, state_dim) or (state_dim,)
        Returns:
            value: Tensor of shape (batch, 1) or (1,)
        """
        return self.net(state)


class ActionConstrainedPPOAgent:
    """
    PPO agent with action masking for constraint satisfaction.
    
    Key features:
    - Action masking: infeasible actions have logits set to -1e9.
    - Sampling: only feasible actions are sampled.
    - GAE: advantage estimation with TD errors and exponential smoothing.
    - PPO: clipped surrogate objective for stable policy updates.
    
    Args:
        state_dim: Dimension of the state space.
        action_dim: Number of discrete actions (per intersection).
        num_intersections: Number of intersections to control.
        learning_rate: Learning rate for optimizer.
        gamma: Discount factor.
        gae_lambda: GAE smoothing parameter.
        ppo_clip_epsilon: PPO clipping range.
        entropy_coef: Coefficient for entropy regularization.
        value_coef: Coefficient for value loss.
        max_grad_norm: Gradient clipping norm.
        hidden_dim: Hidden dimension of networks.
        device: torch device (cpu or cuda).
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_intersections: int = 1,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_intersections = num_intersections
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip_epsilon = ppo_clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.reset_buffer()
        
        # Tracking
        self.num_actions_sampled = 0
        self.num_actions_rejected = 0  # Not used with action masking, but kept for logging
    
    def reset_buffer(self):
        """Reset the experience buffer for a new trajectory."""
        self.states = deque()
        self.actions = deque()
        self.rewards = deque()
        self.values = deque()
        self.log_probs = deque()
        self.dones = deque()
    
    def select_action(self, state: np.ndarray, feasible_actions: list) -> int:
        """
        Select an action using the policy with action masking.
        
        Process:
        1. Compute policy logits.
        2. Mask infeasible actions (set logits to -1e9).
        3. Sample from masked distribution.
        4. Action is guaranteed to be feasible.
        
        Args:
            state: Current state (numpy array).
            feasible_actions: List of feasible action indices for this step.
                             e.g., [0, 1] or [1] if only action 1 is feasible.
        
        Returns:
            action: Selected action index (guaranteed feasible).
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            # Get logits and value
            logits = self.actor(state_tensor)  # Shape: (action_dim,)
            value = self.critic(state_tensor)   # Shape: (1,)
        
        # Apply action masking: set infeasible actions to -1e9
        mask = torch.zeros(self.action_dim, device=self.device)
        for action_idx in feasible_actions:
            mask[action_idx] = 1.0
        
        masked_logits = logits.clone()
        masked_logits[mask == 0] = -1e9
        
        # Create distribution and sample
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Store experience for batch training."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_advantages(self, next_value: float) -> tuple:
        """
        Compute GAE advantages.
        
        GAE formula:
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
            R_t = A_t + V(s_t)  (return estimate)
        
        Args:
            next_value: Value of the final next state.
        
        Returns:
            advantages: List of advantage estimates.
            returns: List of return estimates (target for value network).
        """
        advantages = []
        returns = []
        
        values = list(self.values) + [next_value]
        gae = 0.0
        
        for t in reversed(range(len(self.rewards))):
            done = self.dones[t]
            reward = self.rewards[t]
            
            # TD error (delta)
            delta = reward + self.gamma * values[t + 1] * (1.0 - done) - values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1.0 - done) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        # Normalize advantages for stability
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, batch_size: int = 64, num_epochs: int = 3):
        """
        Update actor and critic using PPO objective.
        
        Args:
            batch_size: Batch size for mini-batch updates.
            num_epochs: Number of passes through the data.
        
        Returns:
            dict with training metrics.
        """
        if len(self.states) == 0:
            return {}
        
        # Compute returns and advantages
        with torch.no_grad():
            last_state = torch.FloatTensor(np.array(list(self.states)[-1])).to(self.device)
            next_value = self.critic(last_state).item()
        
        advantages, returns = self.compute_advantages(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(list(self.states))).to(self.device)
        actions = torch.LongTensor(list(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(list(self.log_probs)).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Training loop
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl_divergence": 0.0,
        }
        
        num_batches = max(1, len(self.states) // batch_size)
        
        for epoch in range(num_epochs):
            # Shuffle indices
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            
            for batch_start in range(0, len(self.states), batch_size):
                batch_indices = indices[batch_start : batch_start + batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                
                # Forward pass
                logits = self.actor(batch_states)  # (batch, action_dim)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                values = self.critic(batch_states).squeeze(-1)  # (batch,)
                
                # PPO policy loss (clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Combined loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                # Optimizer step
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Track metrics
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - new_log_probs).mean().item()
                
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["kl_divergence"] += kl_div
        
        # Average metrics
        num_updates = num_epochs * num_batches
        for key in metrics:
            metrics[key] /= num_updates
        
        # Reset buffer for next episode
        self.reset_buffer()
        
        return metrics
    
    def save(self, path: str):
        """Save actor and critic networks."""
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load actor and critic networks."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
    
    def reset(self):
        """Reset for new episode (clears internal state if any)."""
        pass
