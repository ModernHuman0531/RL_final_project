"""
Baseline evaluation script — Single Intersection.

Runs Fixed-Time, Max Pressure, SOTL, and (optionally) DQN-AR / SPre+ agents
for one episode each and prints a comparison table:
  - Average vehicle waiting time  (lower is better)
  - Average queue length          (lower is better)
  - Constraint violation rate     (target: 0%)
  - Total episode reward

DQN-AR and SPre+ require a trained checkpoint.  Pass --model <path> to include
them; omit it to run only the rule-based baselines.
"""
import argparse
import os
import sys
import numpy as np

# Allow running directly from any working directory inside the project.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sumo_rl.environement.sumo_env import SUMOEnvironment
from sumo_rl.agents.fixed_time_agent import FixedTimeAgent
from sumo_rl.agents.max_pressure_agent import MaxPressureAgent
from sumo_rl.agents.sotl_agent import SOTLAgent

CFG_FILE = os.path.join(
    os.path.dirname(__file__), "..", "nets", "single-intersection", "single_intersection.sumocfg"
)

STATE_DIM_PER_INTERSECTION = 11


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
                        violation_rate, total_reward, steps.
    """
    state, _ = env.reset()
    agent.reset()

    total_reward = 0.0
    waiting_times = []
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
        step_queue = sum(sum(s.get_vehicle_queue()) for s in signals)

        total_reward += reward
        waiting_times.append(step_wait)
        queue_lengths.append(step_queue)
        steps += 1

    return {
        "avg_waiting_time": float(np.mean(waiting_times)),
        "avg_queue_length":  float(np.mean(queue_lengths)),
        "violation_rate":    violations / steps,
        "total_reward":      total_reward,
        "steps":             steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained DQN-AR checkpoint (.pt). "
                             "If provided, DQN-AR and SPre+ are also evaluated.")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

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

    # Add DQN-AR and SPre+ only when a trained checkpoint is available.
    if args.model is not None:
        from sumo_rl.agents.base_agent import DQNAgent
        from sumo_rl.agents.spre_plus_agent import SPRePlusAgent

        dqn_agent = DQNAgent(
            state_dim=STATE_DIM_PER_INTERSECTION,
            action_dim=2,
            num_intersections=1,
            device=args.device,
        )
        dqn_agent.load(args.model)
        dqn_agent.epsilon = 0.0  # greedy evaluation

        spre_agent = SPRePlusAgent(action_dim=2, num_intersections=1, use_scipy=True,
                                   state_dim=STATE_DIM_PER_INTERSECTION)
        spre_agent.set_policy(dqn_agent.get_q_fn())

        experiments += [
            (
                "DQN-AR",
                dqn_agent,
                lambda state, signals: dqn_agent.select_action(state, signals),
            ),
            (
                "SPre+ (DQN policy)",
                spre_agent,
                lambda state, signals: spre_agent.select_action(state, signals),
            ),
        ]

    print("\n=== Baseline Evaluation: Single Intersection (1 episode = 3600 steps) ===\n")

    col = "{:<24} {:>14} {:>13} {:>17} {:>14}"
    print(col.format("Agent", "Avg Wait (s)", "Avg Queue", "Violation Rate", "Total Reward"))
    print("-" * 86)

    for name, agent, get_action in experiments:
        print(f"  Running {name}...", end="", flush=True)
        result = run_episode(env, agent, get_action)
        print(f"\r" + col.format(
            name,
            f"{result['avg_waiting_time']:.2f}",
            f"{result['avg_queue_length']:.2f}",
            f"{result['violation_rate']:.4f}",
            f"{result['total_reward']:.1f}",
        ))

    print()


if __name__ == "__main__":
    main()
