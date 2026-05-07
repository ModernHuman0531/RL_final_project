"""
Baseline evaluation script — Single Intersection.

Runs Fixed-Time, Max Pressure, SOTL, and Action-Constrained PPO agents for one episode each 
and prints a comparison table with the metrics from the project proposal:
  - Average vehicle waiting time  (lower is better)
  - Average queue length          (lower is better)
  - Constraint violation rate     (target: 0%)
  - Total episode reward
  - Pedestrian waiting time       (lower is better)

Usage:
    # Evaluate baselines only
    python eval_baselines.py
    
    # Evaluate baselines + trained ActionConstrainedPPO model
    python eval_baselines.py --trained-model ./results/agent_final.pt --device cuda
"""
import os
import sys
import argparse
import numpy as np

# Allow running directly from any working directory inside the project.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sumo_rl.environement.sumo_env import SUMOEnvironment
from sumo_rl.agents.fixed_time_agent import FixedTimeAgent
from sumo_rl.agents.max_pressure_agent import MaxPressureAgent
from sumo_rl.agents.sotl_agent import SOTLAgent
from sumo_rl.agents.action_constrained_ppo_agent import ActionConstrainedPPOAgent

CFG_FILE = os.path.join(
    os.path.dirname(__file__), "..", "nets", "single-intersection", "single_intersection.sumocfg"
)


def run_episode(env, agent, get_action):
    """
    Run one full episode and return a dict of metrics.

    Args:
        env:        SUMOEnvironment instance.
        agent:      Agent object (must implement reset()).
        get_action: Callable(state, signals) -> np.ndarray.
                    Decouples the different agent interfaces from this loop.
    Returns:
        dict with keys: avg_waiting_time, avg_queue_length,
                        violation_rate, total_reward, steps,
                        avg_pedestrian_waiting_time.
    """
    state, _ = env.reset()
    agent.reset()

    total_reward = 0.0
    waiting_times = []
    pedestrian_waiting_times = []
    queue_lengths = []
    violations = 0
    steps = 0

    done = False
    while not done:
        signals = list(env.traffic_signals.values())
        action = get_action(state, signals)

        # --- Count constraint violations BEFORE the env corrects the action ---
        for i, signal in enumerate(signals):
            valid = signal.get_valid_actions()
            if valid[action[i]] == 0:
                violations += 1

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # --- Collect per-step metrics from the signals' current state ---
        step_wait = sum(s.get_vehicle_waiting_time() for s in signals)
        step_ped_wait = sum(s.get_pedestrian_waiting_time() for s in signals)
        step_queue = sum(sum(s.get_vehicle_queue()) for s in signals)

        total_reward += reward
        waiting_times.append(step_wait)
        pedestrian_waiting_times.append(step_ped_wait)
        queue_lengths.append(step_queue)
        steps += 1

    return {
        "avg_waiting_time": float(np.mean(waiting_times)),
        "avg_pedestrian_waiting_time": float(np.mean(pedestrian_waiting_times)),
        "avg_queue_length":  float(np.mean(queue_lengths)),
        "violation_rate":    violations / steps,
        "total_reward":      total_reward,
        "steps":             steps,
    }


def run_episode_ppo(env, agent, feasible_actions_fn, device="cpu"):
    """
    Run one full episode with ActionConstrainedPPOAgent.
    
    Args:
        env: SUMOEnvironment instance.
        agent: ActionConstrainedPPOAgent instance.
        feasible_actions_fn: Function to get feasible actions from env.
        device: torch device.
    
    Returns:
        dict with metrics.
    """
    import torch
    
    state, _ = env.reset()
    agent.reset()
    
    total_reward = 0.0
    waiting_times = []
    pedestrian_waiting_times = []
    queue_lengths = []
    violations = 0
    steps = 0
    
    done = False
    while not done:
        signals = list(env.traffic_signals.values())
        
        # Get feasible actions
        feasible_actions_list = env.get_feasible_actions()
        feasible_actions = feasible_actions_list[0] if feasible_actions_list else [0, 1]
        
        # Select action (no gradients)
        with torch.no_grad():
            action_value, _, _ = agent.select_action(state, feasible_actions)
        action = np.array([action_value])
        
        # Count violations
        if action_value not in feasible_actions:
            violations += 1
        
        # Check against environment's constraint (should always pass)
        for i, signal in enumerate(signals):
            valid = signal.get_valid_actions()
            if valid[action[i]] == 0:
                violations += 1
        
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        step_wait = sum(s.get_vehicle_waiting_time() for s in signals)
        step_ped_wait = sum(s.get_pedestrian_waiting_time() for s in signals)
        step_queue = sum(sum(s.get_vehicle_queue()) for s in signals)
        
        total_reward += reward
        waiting_times.append(step_wait)
        pedestrian_waiting_times.append(step_ped_wait)
        queue_lengths.append(step_queue)
        steps += 1
    
    return {
        "avg_waiting_time": float(np.mean(waiting_times)),
        "avg_pedestrian_waiting_time": float(np.mean(pedestrian_waiting_times)),
        "avg_queue_length": float(np.mean(queue_lengths)),
        "violation_rate": violations / steps,
        "total_reward": total_reward,
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate traffic signal control baselines.")
    parser.add_argument(
        "--sumo-cfg-file",
        type=str,
        default="sumo_rl/nets/single-intersection/single_intersection.sumocfg",
        help="Path to SUMO configuration file.",
    )
    parser.add_argument(
        "--trained-model",
        type=str,
        default=None,
        help="Path to trained ActionConstrainedPPOAgent model (.pt file). If provided, evaluates the trained agent.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for PPO agent (cpu or cuda).",
    )
    
    args = parser.parse_args()
    
    CFG_FILE = args.sumo_cfg_file
    TRAINED_MODEL_PATH = args.trained_model
    DEVICE = args.device
    
    env = SUMOEnvironment(
        sumo_cfg_file=CFG_FILE,
        delta_time=1,
        yellow_time=5,
        min_green_time=10,
        max_green_time=60,
        use_gui=False,
    )

    ft_agent   = FixedTimeAgent(cycle_time=30, num_intersections=1)
    mp_agent   = MaxPressureAgent(num_intersections=1)
    sotl_agent = SOTLAgent(kappa=5, num_intersections=1)

    # Each entry: (display name, agent, get_action callable)
    # Lambdas give all agents a uniform (state, signals) interface.
    experiments = [
        (
            "Fixed-Time (30s cycle)",
            ft_agent,
            lambda state, signals: ft_agent.select_action(state),
        ),
        (
            "Max Pressure",
            mp_agent,
            lambda state, signals: mp_agent.select_action(signals),
        ),
        (
            "SOTL (kappa=5)",
            sotl_agent,
            lambda state, signals: sotl_agent.select_action(signals),
        ),
    ]
    
    # Add trained PPO agent if model path is provided
    if TRAINED_MODEL_PATH is not None:
        print(f"[INFO] Loading trained PPO model from {TRAINED_MODEL_PATH}")
        import torch
        
        # Determine state and action dimensions from environment
        state, _ = env.reset()
        state_dim = len(state)
        action_dim = 2
        env.close()  # Close for re-initialization
        
        # Create and load agent
        ppo_agent = ActionConstrainedPPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            num_intersections=1,
            device=DEVICE,
        )
        ppo_agent.load(TRAINED_MODEL_PATH)
        
        experiments.append(
            (
                "Action-Constrained PPO",
                ppo_agent,
                None,  # Special handling for PPO
            )
        )

    print("\n=== Baseline & Method Evaluation: Single Intersection (1 episode = 3600 steps) ===\n")

    col = "{:<28} {:>14} {:>16} {:>14} {:>17} {:>14}"
    
    # Collect results from all experiments (run silently)
    results_list = []
    
    for experiment in experiments:
        if len(experiment) == 3:
            name, agent, get_action = experiment
            
            if get_action is None:
                # PPO agent special case
                result = run_episode_ppo(env, agent, lambda: env.get_feasible_actions(), device=DEVICE)
            else:
                result = run_episode(env, agent, get_action)
            
            results_list.append((name, result))
    
    # Print header
    print(col.format(
        "Agent",
        "Avg Wait (s)",
        "Avg Ped Wait (s)",
        "Avg Queue",
        "Violation Rate",
        "Total Reward"
    ))
    print("-" * 115)
    
    # Print all results at once (clean output)
    for name, result in results_list:
        print(col.format(
            name,
            f"{result['avg_waiting_time']:.2f}",
            f"{result['avg_pedestrian_waiting_time']:.2f}",
            f"{result['avg_queue_length']:.2f}",
            f"{result['violation_rate']:.4f}",
            f"{result['total_reward']:.1f}",
        ))

    print()
    env.close()


if __name__ == "__main__":
    import torch
    main()
