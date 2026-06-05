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
        value_clip_epsilon: float = None,
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
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.value_clip_epsilon = (
            ppo_clip_epsilon if value_clip_epsilon is None else value_clip_epsilon
        )
        self.device = torch.device(device)
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.reset_buffer()
        self._last_action_mask = None
        
        # Tracking
        self.num_actions_sampled = 0
        self.num_actions_rejected = 0  # Not used with action masking, but kept for logging
    
    def reset_buffer(self):
        """Reset the experience buffer for a new trajectory."""
        self.states = deque()
        self.actions = deque()
        self.action_masks = deque()
        self.rewards = deque()
        self.values = deque()
        self.log_probs = deque()
        self.dones = deque()

    def _build_action_mask(self, feasible_actions: list) -> np.ndarray:
        """
        Build a binary mask over the action space.

        The same mask used during sampling must also be used during PPO updates,
        otherwise old and new log probabilities come from different
        distributions.
        """
        mask = np.zeros(self.action_dim, dtype=np.float32)
        for action_idx in feasible_actions:
            if 0 <= action_idx < self.action_dim:
                mask[action_idx] = 1.0

        if mask.sum() == 0:
            raise ValueError("No feasible actions were provided to the PPO agent.")

        return mask
    
    def select_action(
        self,
        state: np.ndarray,
        feasible_actions: list,
        deterministic: bool = False,
    ) -> int:
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
            deterministic: If True, choose the highest-probability feasible
                           action instead of sampling. Use for evaluation.
        
        Returns:
            action: Selected action index (guaranteed feasible).
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            # Get logits and value
            logits = self.actor(state_tensor)  # Shape: (action_dim,)
            value = self.critic(state_tensor)   # Shape: (1,)
        
        # Apply action masking: set infeasible actions to -1e9
        mask_np = self._build_action_mask(feasible_actions)
        mask = torch.FloatTensor(mask_np).to(self.device)
        
        masked_logits = logits.clone()
        masked_logits[mask == 0] = -1e9
        
        # Create distribution and sample
        dist = Categorical(logits=masked_logits)
        if deterministic:
            action = torch.argmax(masked_logits)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        # Keep the exact mask so store_experience() can save it with the step.
        self._last_action_mask = mask_np.copy()
        self.num_actions_sampled += 1
        
        return action.item(), log_prob.item(), value.item()
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        feasible_actions: list = None,
        action_mask: np.ndarray = None,
    ):
        """Store experience for batch training."""
        if action_mask is None:
            if feasible_actions is not None:
                action_mask = self._build_action_mask(feasible_actions)
            elif self._last_action_mask is not None:
                action_mask = self._last_action_mask
            else:
                action_mask = np.ones(self.action_dim, dtype=np.float32)

        action_mask = np.asarray(action_mask, dtype=np.float32)
        if action_mask.shape != (self.action_dim,):
            raise ValueError(
                f"action_mask must have shape ({self.action_dim},), got {action_mask.shape}"
            )
        if action_mask.sum() == 0:
            raise ValueError("Stored action_mask has no valid actions.")

        self.states.append(state)
        self.actions.append(action)
        self.action_masks.append(action_mask.copy())
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
        action_masks = torch.FloatTensor(np.array(list(self.action_masks))).to(self.device)
        old_values = torch.FloatTensor(list(self.values)).to(self.device)
        old_log_probs = torch.FloatTensor(list(self.log_probs)).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Training loop
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl_divergence": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "value_clip_fraction": 0.0,
            "explained_variance": 0.0,
            "mean_return": returns_tensor.mean().item(),
            "mean_advantage": advantages_tensor.mean().item(),
            "std_advantage": advantages_tensor.std(unbiased=False).item(),
            "valid_action_count": action_masks.sum(dim=1).mean().item(),
            "valid_action_fraction": action_masks.mean().item(),
        }
        
        num_updates = 0
        
        for epoch in range(num_epochs):
            # Shuffle indices
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            
            for batch_start in range(0, len(self.states), batch_size):
                batch_indices = indices[batch_start : batch_start + batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_action_masks = action_masks[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                
                # Forward pass
                logits = self.actor(batch_states)  # (batch, action_dim)
                masked_logits = logits.masked_fill(batch_action_masks == 0, -1e9)
                dist = Categorical(logits=masked_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                values = self.critic(batch_states).squeeze(-1)  # (batch,)
                
                # PPO policy loss (clipped objective)
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_epsilon, 1.0 + self.ppo_clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Clipped value loss keeps critic updates from jumping too far
                # away from the value estimates used to compute advantages.
                value_pred_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values,
                    -self.value_clip_epsilon,
                    self.value_clip_epsilon,
                )
                value_loss_unclipped = (values - batch_returns).pow(2)
                value_loss_clipped = (value_pred_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(
                    value_loss_unclipped,
                    value_loss_clipped,
                ).mean()
                
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
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.ppo_clip_epsilon)
                        .float()
                        .mean()
                        .item()
                    )
                    value_clip_fraction = (
                        (torch.abs(values - batch_old_values) > self.value_clip_epsilon)
                        .float()
                        .mean()
                        .item()
                    )
                
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["kl_divergence"] += kl_div
                metrics["approx_kl"] += approx_kl
                metrics["clip_fraction"] += clip_fraction
                metrics["value_clip_fraction"] += value_clip_fraction
                num_updates += 1
        
        # Average metrics
        for key in (
            "policy_loss",
            "value_loss",
            "entropy",
            "kl_divergence",
            "approx_kl",
            "clip_fraction",
            "value_clip_fraction",
        ):
            metrics[key] /= max(num_updates, 1)

        with torch.no_grad():
            final_values = self.critic(states).squeeze(-1)
            returns_variance = torch.var(returns_tensor, unbiased=False)
            if returns_variance.item() > 1e-8:
                explained_variance = (
                    1.0
                    - torch.var(returns_tensor - final_values, unbiased=False)
                    / returns_variance
                )
                metrics["explained_variance"] = explained_variance.item()
        
        # Reset buffer for next episode
        self.reset_buffer()
        
        return metrics
    
    def get_config(self):
        """Return enough metadata to recreate this agent."""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "num_intersections": self.num_intersections,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ppo_clip_epsilon": self.ppo_clip_epsilon,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "hidden_dim": self.hidden_dim,
            "value_clip_epsilon": self.value_clip_epsilon,
            "device": str(self.device),
        }

    def save(self, path: str, episode: int = None, extra: dict = None):
        """Save actor, critic, optimizer state, and training metadata."""
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "config": self.get_config(),
            "episode": episode,
            "num_actions_sampled": self.num_actions_sampled,
            "num_actions_rejected": self.num_actions_rejected,
            "extra": extra or {},
        }, path)
    
    def load(self, path: str, load_optimizers: bool = True):
        """Load actor and critic networks."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        if load_optimizers and "actor_optimizer_state_dict" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if load_optimizers and "critic_optimizer_state_dict" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.num_actions_sampled = checkpoint.get("num_actions_sampled", self.num_actions_sampled)
        self.num_actions_rejected = checkpoint.get("num_actions_rejected", self.num_actions_rejected)
        return checkpoint
    
    def reset(self):
        """Reset for new episode (clears internal state if any)."""
        self._last_action_mask = None
