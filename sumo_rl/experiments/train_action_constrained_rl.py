"""
Training script for Action-Constrained PPO agent on traffic signal control.

Features:
- Single-agent control (one PPO agent per intersection, independent)
- Action masking ensures 0% constraint violations
- Logs metrics: episode reward, waiting times, constraint violations
- Saves trained models
- Can evaluate on test episodes

Usage:
    python train_action_constrained_rl.py \
        --sumo-cfg-file path/to/config.sumocfg \
        --num-episodes 100 \
        --learning-rate 3e-4 \
        --gamma 0.99 \
        --output-dir ./results
"""

import os
import sys
import argparse
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# Ensure package imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sumo_rl.environement.sumo_env import SUMOEnvironment
from sumo_rl.agents.action_constrained_ppo_agent import ActionConstrainedPPOAgent


def train_episode(env, agent, alpha_vehicle=1.0, beta_pedestrian=1.0):
    """
    Run one training episode and update the agent.
    
    Args:
        env: SUMOEnvironment instance.
        agent: ActionConstrainedPPOAgent instance.
        alpha_vehicle: Weight for vehicle waiting time in reward.
        beta_pedestrian: Weight for pedestrian waiting time in reward.
    
    Returns:
        dict with episode metrics.
    """
    state, _ = env.reset()
    agent.reset()
    
    episode_reward = 0.0
    episode_vehicle_wait = 0.0
    episode_ped_wait = 0.0
    episode_queue = 0.0
    constraint_violations = 0
    steps = 0
    
    done = False
    while not done:
        # Get feasible actions for all intersections
        feasible_actions_list = env.get_feasible_actions()
        
        # For single intersection (assume index 0)
        # If multiple intersections, this logic extends naturally
        feasible_actions = feasible_actions_list[0] if feasible_actions_list else [0, 1]
        
        # Select action using masked PPO policy
        action_value, log_prob, value_estimate = agent.select_action(state, feasible_actions)
        action = np.array([action_value])  # PPO returns int, wrap in array
        
        # Check if action is actually feasible (sanity check)
        if action_value not in feasible_actions:
            constraint_violations += 1
        
        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Compute detailed reward breakdown for logging
        vehicle_wait = sum(s.get_vehicle_waiting_time() for s in env.traffic_signals.values())
        ped_wait = sum(s.get_pedestrian_waiting_time() for s in env.traffic_signals.values())
        queue_len = sum(sum(s.get_vehicle_queue()) for s in env.traffic_signals.values())
        
        # Standard reward for training
        detailed_reward = -(alpha_vehicle * vehicle_wait + beta_pedestrian * ped_wait)
        
        # Store experience (use computed detailed reward)
        agent.store_experience(
            state=state,
            action=action_value,
            reward=detailed_reward,
            value=value_estimate,
            log_prob=log_prob,
            done=done,
        )
        
        # Update running totals
        episode_reward += detailed_reward
        episode_vehicle_wait += vehicle_wait
        episode_ped_wait += ped_wait
        episode_queue += queue_len
        steps += 1
        
        state = next_state
    
    # Update agent at end of episode
    update_metrics = agent.update(batch_size=64, num_epochs=3)
    
    metrics = {
        "episode_reward": episode_reward,
        "avg_vehicle_waiting_time": episode_vehicle_wait / max(steps, 1),
        "avg_pedestrian_waiting_time": episode_ped_wait / max(steps, 1),
        "avg_queue_length": episode_queue / max(steps, 1),
        "constraint_violations": constraint_violations,
        "constraint_violation_rate": constraint_violations / max(steps, 1),
        "steps": steps,
    }
    
    # Add training metrics
    metrics.update(update_metrics)
    
    return metrics


def evaluate_episode(env, agent, num_episodes=5, alpha_vehicle=1.0, beta_pedestrian=1.0):
    """
    Run evaluation episodes without training updates.
    
    Args:
        env: SUMOEnvironment instance.
        agent: ActionConstrainedPPOAgent instance.
        num_episodes: Number of evaluation episodes.
        alpha_vehicle: Weight for vehicle waiting time.
        beta_pedestrian: Weight for pedestrian waiting time.
    
    Returns:
        dict with aggregated metrics over evaluation episodes.
    """
    all_metrics = {
        "episode_rewards": [],
        "avg_vehicle_waiting_times": [],
        "avg_pedestrian_waiting_times": [],
        "avg_queue_lengths": [],
        "constraint_violation_rates": [],
    }
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        agent.reset()
        
        episode_reward = 0.0
        episode_vehicle_wait = 0.0
        episode_ped_wait = 0.0
        episode_queue = 0.0
        violations = 0
        steps = 0
        
        done = False
        while not done:
            # Get feasible actions
            feasible_actions_list = env.get_feasible_actions()
            feasible_actions = feasible_actions_list[0] if feasible_actions_list else [0, 1]
            
            # Select action (no gradient)
            with torch.no_grad():
                action_value, _, _ = agent.select_action(state, feasible_actions)
            action = np.array([action_value])
            
            if action_value not in feasible_actions:
                violations += 1
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            vehicle_wait = sum(s.get_vehicle_waiting_time() for s in env.traffic_signals.values())
            ped_wait = sum(s.get_pedestrian_waiting_time() for s in env.traffic_signals.values())
            queue_len = sum(sum(s.get_vehicle_queue()) for s in env.traffic_signals.values())
            
            detailed_reward = -(alpha_vehicle * vehicle_wait + beta_pedestrian * ped_wait)
            
            episode_reward += detailed_reward
            episode_vehicle_wait += vehicle_wait
            episode_ped_wait += ped_wait
            episode_queue += queue_len
            steps += 1
            
            state = next_state
        
        all_metrics["episode_rewards"].append(episode_reward)
        all_metrics["avg_vehicle_waiting_times"].append(episode_vehicle_wait / max(steps, 1))
        all_metrics["avg_pedestrian_waiting_times"].append(episode_ped_wait / max(steps, 1))
        all_metrics["avg_queue_lengths"].append(episode_queue / max(steps, 1))
        all_metrics["constraint_violation_rates"].append(violations / max(steps, 1))
    
    # Aggregate
    aggregated = {
        "eval_avg_reward": float(np.mean(all_metrics["episode_rewards"])),
        "eval_std_reward": float(np.std(all_metrics["episode_rewards"])),
        "eval_avg_vehicle_waiting_time": float(np.mean(all_metrics["avg_vehicle_waiting_times"])),
        "eval_avg_pedestrian_waiting_time": float(np.mean(all_metrics["avg_pedestrian_waiting_times"])),
        "eval_avg_queue_length": float(np.mean(all_metrics["avg_queue_lengths"])),
        "eval_avg_constraint_violation_rate": float(np.mean(all_metrics["constraint_violation_rates"])),
    }
    
    return aggregated


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train Action-Constrained PPO for traffic signal control.")
    parser.add_argument(
        "--sumo-cfg-file",
        type=str,
        default="sumo_rl/nets/single-intersection/single_intersection.sumocfg",
        help="Path to SUMO configuration file.",
    )
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of training episodes.")
    parser.add_argument("--num-eval-episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluate every N episodes.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.2, help="PPO clip epsilon.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy regularization coefficient.")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient.")
    parser.add_argument("--alpha-vehicle", type=float, default=1.0, help="Weight for vehicle waiting time.")
    parser.add_argument("--beta-pedestrian", type=float, default=1.0, help="Weight for pedestrian waiting time.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory for outputs.")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda).")
    parser.add_argument("--delta-time", type=int, default=1, help="SUMO delta time.")
    parser.add_argument("--yellow-time", type=int, default=5, help="Yellow phase duration.")
    parser.add_argument("--min-green-time", type=int, default=10, help="Minimum green time.")
    parser.add_argument("--max-green-time", type=int, default=60, help="Maximum green time.")
    parser.add_argument("--sim-end-time", type=int, default=3600, help="Simulation end time (seconds).")
    
    args = parser.parse_args()
    
    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Config saved to {output_dir / 'config.json'}")
    
    # Initialize environment
    print(f"[INFO] Initializing environment: {args.sumo_cfg_file}")
    env = SUMOEnvironment(
        sumo_cfg_file=args.sumo_cfg_file,
        delta_time=args.delta_time,
        yellow_time=args.yellow_time,
        min_green_time=args.min_green_time,
        max_green_time=args.max_green_time,
        end_time=args.sim_end_time,
        use_gui=False,
    )
    
    # Get state and action dimensions (from first reset)
    state, _ = env.reset()
    state_dim = len(state)
    action_dim = 2  # Single intersection has 2 actions (NS green or EW green)
    print(f"[INFO] State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Initialize agent
    print(f"[INFO] Initializing ActionConstrainedPPOAgent")
    agent = ActionConstrainedPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_intersections=1,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ppo_clip_epsilon=args.ppo_clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        device=args.device,
    )
    
    # Training loop
    print(f"[INFO] Starting training for {args.num_episodes} episodes")
    
    train_history = []
    eval_history = []
    
    for episode in range(args.num_episodes):
        # Train
        metrics = train_episode(
            env,
            agent,
            alpha_vehicle=args.alpha_vehicle,
            beta_pedestrian=args.beta_pedestrian,
        )
        train_history.append(metrics)
        
        # Log
        if (episode + 1) % 10 == 0:
            print(
                f"[Episode {episode + 1}/{args.num_episodes}] "
                f"Reward: {metrics['episode_reward']:.2f}, "
                f"Avg Wait (V): {metrics['avg_vehicle_waiting_time']:.2f}, "
                f"Avg Wait (P): {metrics['avg_pedestrian_waiting_time']:.2f}, "
                f"Violations: {metrics['constraint_violations']}"
            )
        
        # Evaluate
        if (episode + 1) % args.eval_interval == 0:
            print(f"[INFO] Running evaluation ({args.num_eval_episodes} episodes)...")
            eval_metrics = evaluate_episode(
                env,
                agent,
                num_episodes=args.num_eval_episodes,
                alpha_vehicle=args.alpha_vehicle,
                beta_pedestrian=args.beta_pedestrian,
            )
            eval_metrics["episode"] = episode + 1
            eval_history.append(eval_metrics)
            
            print(
                f"[Eval at Episode {episode + 1}] "
                f"Avg Reward: {eval_metrics['eval_avg_reward']:.2f}, "
                f"Avg Wait (V): {eval_metrics['eval_avg_vehicle_waiting_time']:.2f}, "
                f"Violation Rate: {eval_metrics['eval_avg_constraint_violation_rate']:.4f}"
            )
    
    # Save results
    with open(output_dir / "train_history.json", "w") as f:
        json.dump(train_history, f, indent=2)
    
    with open(output_dir / "eval_history.json", "w") as f:
        json.dump(eval_history, f, indent=2)
    
    # Save model
    model_path = output_dir / "agent_final.pt"
    agent.save(str(model_path))
    print(f"[INFO] Model saved to {model_path}")
    
    # Final summary
    print(f"\n[INFO] Training completed. Results saved to {output_dir}")
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    import torch
    main()
