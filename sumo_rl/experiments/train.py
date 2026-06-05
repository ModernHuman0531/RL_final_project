"""
Train DQN-AR on a SUMO traffic-light environment.

The script discovers the current observation/action dimensions from SUMO, so
it works with the current 1x1 four-action setup and center-controlled 3x3
experiments. Each run writes to a unique folder by default.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sumo_rl.agents.base_agent import DQNAgent
from sumo_rl.environement.sumo_env import SUMOEnvironment


DEFAULT_CFG_FILE = "sumo_rl/nets/single-intersection/single_intersection.sumocfg"


def parse_controlled_tls(value):
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def make_run_dir(base_dir: str, run_name: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = run_name or f"dqn_ar_{timestamp}"
    run_dir = os.path.join(base_dir, name)
    counter = 1
    unique_dir = run_dir
    while os.path.exists(unique_dir):
        unique_dir = f"{run_dir}_{counter}"
        counter += 1
    os.makedirs(os.path.join(unique_dir, "checkpoints"), exist_ok=True)
    return unique_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN-AR for SUMO traffic signal control.")
    parser.add_argument("--sumo-cfg-file", type=str, default=DEFAULT_CFG_FILE)
    parser.add_argument("--controlled-tls", type=str, default=None, help="Comma-separated TLS ids, for example B1.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--sim-end-time", type=int, default=3600)
    parser.add_argument("--delta-time", type=int, default=1)
    parser.add_argument("--yellow-time", type=int, default=5)
    parser.add_argument("--min-green-time", type=int, default=10)
    parser.add_argument("--max-green-time", type=int, default=60)
    parser.add_argument("--reward-mode", choices=["waiting_time", "queue_delta", "hybrid"], default="waiting_time")
    parser.add_argument("--queue-reward-weight", type=float, default=1.0)
    parser.add_argument("--vehicle-wait-weight", type=float, default=1.0)
    parser.add_argument("--pedestrian-wait-weight", type=float, default=1.0)
    parser.add_argument("--violation-penalty", type=float, default=50.0)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save", type=str, default=None, help="Optional extra legacy checkpoint path.")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--gui", action="store_true", help="Show SUMO GUI. This is slow.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=None)
    return parser.parse_args()


def make_env(args):
    return SUMOEnvironment(
        sumo_cfg_file=args.sumo_cfg_file,
        delta_time=args.delta_time,
        yellow_time=args.yellow_time,
        min_green_time=args.min_green_time,
        max_green_time=args.max_green_time,
        end_time=args.sim_end_time,
        controlled_tls=parse_controlled_tls(args.controlled_tls),
        use_gui=args.gui,
        reward_mode=args.reward_mode,
        queue_reward_weight=args.queue_reward_weight,
        vehicle_wait_weight=args.vehicle_wait_weight,
        pedestrian_wait_weight=args.pedestrian_wait_weight,
        violation_penalty=args.violation_penalty,
    )


def probe_dimensions(env):
    state, _ = env.reset()
    signals = list(env.traffic_signals.values())
    if not signals:
        raise RuntimeError("No controlled traffic signals were found.")

    state_dims = [signal.state_size for signal in signals]
    action_dims = [len(signal.green_phases) for signal in signals]
    if len(set(state_dims)) != 1 or len(set(action_dims)) != 1:
        raise RuntimeError(
            "DQNAgent currently shares one network across intersections, so all controlled "
            "signals must have the same state/action dimensions. Use --controlled-tls B1 "
            "for a single 3x3 intersection."
        )

    return {
        "state_dim": int(state_dims[0]),
        "action_dim": int(action_dims[0]),
        "num_intersections": len(signals),
        "joint_state_dim": int(len(state)),
        "traffic_signal_ids": list(env.traffic_signals.keys()),
        "green_phases": {sid: signal.green_phases for sid, signal in env.traffic_signals.items()},
    }


def run_episode(env, agent):
    state, _ = env.reset()
    agent.reset()

    total_reward = 0.0
    waiting_times = []
    pedestrian_waiting_times = []
    queue_lengths = []
    violations = 0
    losses = []
    steps = 0
    done = False

    while not done:
        signals = list(env.traffic_signals.values())
        action = agent.select_action(state, signals)

        for idx, signal in enumerate(signals):
            valid = signal.get_valid_actions()
            proposed = int(action[idx])
            if proposed < 0 or proposed >= len(valid) or valid[proposed] == 0:
                violations += 1

        next_state, reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated

        agent.store(state, action[0] if len(action) == 1 else action, reward, next_state, float(done))
        loss = agent.update()
        if loss is not None:
            losses.append(loss)

        waiting_times.append(env.get_total_vehicle_waiting_time())
        pedestrian_waiting_times.append(env.get_total_pedestrian_waiting_time())
        queue_lengths.append(env.get_total_vehicle_queue())
        total_reward += float(reward)
        state = next_state
        steps += 1

    return {
        "avg_waiting_time": float(np.mean(waiting_times)) if waiting_times else 0.0,
        "avg_pedestrian_waiting_time": float(np.mean(pedestrian_waiting_times)) if pedestrian_waiting_times else 0.0,
        "avg_queue_length": float(np.mean(queue_lengths)) if queue_lengths else 0.0,
        "violation_rate": float(violations / max(steps, 1)),
        "total_reward": float(total_reward),
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "epsilon": float(agent.epsilon),
        "steps": int(steps),
    }


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def main():
    args = parse_args()
    run_dir = make_run_dir(args.output_dir, args.run_name)

    env = make_env(args)
    dims = probe_dimensions(env)
    env.close()

    epsilon_decay_steps = args.epsilon_decay_steps
    if epsilon_decay_steps is None:
        epsilon_decay_steps = max(args.episodes * args.sim_end_time // 2, 1)

    config = vars(args).copy()
    config.update(dims)
    config["run_dir"] = run_dir
    config["epsilon_decay_steps"] = epsilon_decay_steps
    write_json(os.path.join(run_dir, "config.json"), config)

    env = make_env(args)
    agent = DQNAgent(
        state_dim=dims["state_dim"],
        action_dim=dims["action_dim"],
        num_intersections=dims["num_intersections"],
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        target_update_freq=args.target_update_freq,
        device=args.device,
    )

    history = []
    jsonl_path = os.path.join(run_dir, "train_metrics.jsonl")
    csv_path = os.path.join(run_dir, "train_history.csv")
    final_path = os.path.join(run_dir, "dqn_ar_final.pt")

    print("\n=== DQN-AR Training ===")
    print(f"Run dir: {run_dir}")
    print(f"State dim: {dims['state_dim']} | Action dim: {dims['action_dim']} | Signals: {dims['traffic_signal_ids']}\n")

    col = "{:<6} {:>13} {:>15} {:>12} {:>16} {:>12} {:>9}"
    print(col.format("Ep", "Avg Wait", "Avg Ped Wait", "Avg Queue", "Violation Rate", "Reward", "Epsilon"))
    print("-" * 94)

    fieldnames = [
        "episode",
        "avg_waiting_time",
        "avg_pedestrian_waiting_time",
        "avg_queue_length",
        "violation_rate",
        "total_reward",
        "avg_loss",
        "epsilon",
        "steps",
    ]

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file, open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for episode in range(1, args.episodes + 1):
            result = run_episode(env, agent)
            result["episode"] = episode
            history.append(result)

            jsonl_file.write(json.dumps(result) + "\n")
            jsonl_file.flush()
            writer.writerow(result)
            csv_file.flush()

            if args.checkpoint_interval > 0 and episode % args.checkpoint_interval == 0:
                ckpt_path = os.path.join(run_dir, "checkpoints", f"dqn_ar_ep_{episode:04d}.pt")
                agent.save(ckpt_path, episode=episode, extra={"metrics": result})

            print(col.format(
                episode,
                f"{result['avg_waiting_time']:.2f}",
                f"{result['avg_pedestrian_waiting_time']:.2f}",
                f"{result['avg_queue_length']:.2f}",
                f"{result['violation_rate']:.4f}",
                f"{result['total_reward']:.1f}",
                f"{result['epsilon']:.3f}",
            ))

    agent.save(final_path, episode=args.episodes, extra={"history_path": "train_history.json"})
    if args.save:
        agent.save(args.save, episode=args.episodes, extra={"source_run_dir": run_dir})

    write_json(os.path.join(run_dir, "train_history.json"), history)
    write_json(
        os.path.join(run_dir, "run_summary.json"),
        {
            "run_dir": run_dir,
            "final_model": final_path,
            "legacy_save": args.save,
            "episodes": args.episodes,
            "final_metrics": history[-1] if history else {},
        },
    )
    env.close()

    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Metrics saved in: {run_dir}\n")


if __name__ == "__main__":
    main()
