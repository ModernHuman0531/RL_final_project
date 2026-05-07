"""
Training script — DQN-AR agent on Single Intersection.

Usage (inside Docker or with SUMO_HOME set):
    python -m sumo_rl.experiments.train
    python -m sumo_rl.experiments.train --episodes 50 --save models/dqn_ar.pt

Checkpoints are saved after each episode to the path given by --save.
A CSV log is written to the same directory as the save path.
"""
import os
import sys
import argparse
import csv
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sumo_rl.environement.sumo_env import SUMOEnvironment
from sumo_rl.agents.base_agent import DQNAgent

CFG_FILE = os.path.join(
    os.path.dirname(__file__), "..", "nets", "single-intersection", "single_intersection.sumocfg"
)

# State dimension for one intersection: 2 (phase_one_hot) + 1 (min_green) + 4 (vehicle_queue) + 4 (ped_queue) = 11
STATE_DIM_PER_INTERSECTION = 11


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",   type=int,   default=30,             help="Number of training episodes")
    p.add_argument("--save",       type=str,   default="models/dqn_ar.pt", help="Path to save model checkpoint")
    p.add_argument("--gui",        action="store_true",                 help="Show SUMO GUI (slow)")
    p.add_argument("--device",     type=str,   default="cpu",           help="torch device")
    return p.parse_args()


def run_episode(env, agent):
    state, _ = env.reset()
    agent.reset()

    total_reward = 0.0
    waiting_times = []
    queue_lengths = []
    violations = 0
    losses = []
    steps = 0
    done = False

    while not done:
        signals = list(env.traffic_signals.values())

        # Count violations before env corrects the action
        action = agent.select_action(state, signals)
        for i, signal in enumerate(signals):
            if signal.get_valid_actions()[action[i]] == 0:
                violations += 1

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store(state, action[0] if len(action) == 1 else action, reward, next_state, float(done))
        loss = agent.update()
        if loss is not None:
            losses.append(loss)

        step_wait  = sum(s.get_vehicle_waiting_time() for s in signals)
        step_queue = sum(sum(s.get_vehicle_queue())    for s in signals)
        waiting_times.append(step_wait)
        queue_lengths.append(step_queue)
        total_reward += reward
        state = next_state
        steps += 1

    return {
        "avg_waiting_time": float(np.mean(waiting_times)),
        "avg_queue_length":  float(np.mean(queue_lengths)),
        "violation_rate":    violations / max(steps, 1),
        "total_reward":      total_reward,
        "avg_loss":          float(np.mean(losses)) if losses else 0.0,
        "epsilon":           agent.epsilon,
        "steps":             steps,
    }


def main():
    args = parse_args()

    # Create output directory if needed
    save_dir = os.path.dirname(args.save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir or ".", "train_log.csv") if save_dir else "train_log.csv"

    env = SUMOEnvironment(
        sumo_cfg_file=CFG_FILE,
        delta_time=1,
        yellow_time=5,
        min_green_time=10,
        max_green_time=60,
        use_gui=args.gui,
    )

    agent = DQNAgent(
        state_dim=STATE_DIM_PER_INTERSECTION,
        action_dim=2,
        num_intersections=1,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=args.episodes * 3600 // 2,  # decay over first half of training
        batch_size=64,
        replay_capacity=50_000,
        target_update_freq=500,
        device=args.device,
    )

    col = "{:<6} {:>13} {:>12} {:>15} {:>14} {:>10} {:>8}"
    header = col.format("Ep", "Avg Wait (s)", "Avg Queue", "Violation Rate", "Total Reward", "Avg Loss", "Eps")
    print("\n=== DQN-AR Training: Single Intersection ===\n")
    print(header)
    print("-" * len(header))

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "episode", "avg_waiting_time", "avg_queue_length",
            "violation_rate", "total_reward", "avg_loss", "epsilon", "steps"
        ])
        writer.writeheader()

        for ep in range(1, args.episodes + 1):
            print(f"  Episode {ep}/{args.episodes}...", end="", flush=True)
            result = run_episode(env, agent)
            result["episode"] = ep
            writer.writerow(result)
            f.flush()

            print("\r" + col.format(
                ep,
                f"{result['avg_waiting_time']:.2f}",
                f"{result['avg_queue_length']:.2f}",
                f"{result['violation_rate']:.4f}",
                f"{result['total_reward']:.1f}",
                f"{result['avg_loss']:.4f}",
                f"{result['epsilon']:.3f}",
            ))

            agent.save(args.save)

    print(f"\nTraining complete. Model saved to {args.save}")
    print(f"Log saved to {log_path}\n")


if __name__ == "__main__":
    main()
