"""
Training script for the 1x1 Action-Constrained PPO traffic-signal agent.

Each invocation writes to a unique run directory:
    results/ppo_1x1_YYYYMMDD_HHMMSS_micro/

The run directory contains:
    config.json
    train_history.json
    eval_history.json
    train_metrics.jsonl
    eval_metrics.jsonl
    run_summary.json
    agent_final.pt
    best_agent.pt
    checkpoints/agent_ep_XXXX.pt
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Ensure package imports work when run as a script or module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sumo_rl.agents.action_constrained_ppo_agent import ActionConstrainedPPOAgent
from sumo_rl.environement.sumo_env import SUMOEnvironment


def to_jsonable(value):
    """Convert numpy/torch-ish values to plain JSON-compatible objects."""
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload):
    with open(path, "w") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def append_jsonl(path: Path, payload):
    with open(path, "a") as f:
        f.write(json.dumps(to_jsonable(payload)) + "\n")


def create_run_dir(output_root: str, run_name: str = None) -> Path:
    root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = run_name or f"ppo_1x1_{timestamp}"
    run_dir = root / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{base_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    return run_dir


def parse_controlled_tls(raw_value):
    if raw_value is None or raw_value.strip() == "":
        return None
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def collect_step_metrics(env):
    return {
        "vehicle_wait": env.get_total_vehicle_waiting_time(),
        "pedestrian_wait": env.get_total_pedestrian_waiting_time(),
        "queue": env.get_total_vehicle_queue(),
        "pedestrian_crossing_occupancy": env.get_total_pedestrian_crossing_occupancy(),
    }


def train_episode(env, agent, batch_size=64, update_epochs=3):
    """Run one training episode and update PPO once at the end."""
    state, _ = env.reset()
    agent.reset()

    episode_reward = 0.0
    episode_vehicle_wait = 0.0
    episode_ped_wait = 0.0
    episode_queue = 0.0
    episode_queue_delta = 0.0
    episode_invalid_repairs = 0
    episode_ped_crossing_steps = 0.0
    constraint_violations = 0
    valid_action_counts = []
    steps = 0

    done = False
    while not done:
        feasible_actions_list = env.get_feasible_actions()
        feasible_actions = feasible_actions_list[0] if feasible_actions_list else [0]
        valid_action_counts.append(len(feasible_actions))

        action_value, log_prob, value_estimate = agent.select_action(
            state,
            feasible_actions,
            deterministic=False,
        )
        action = np.array([action_value], dtype=np.int32)

        if action_value not in feasible_actions:
            constraint_violations += 1

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        step_metrics = collect_step_metrics(env)
        reward_components = info.get("reward_components", {})

        agent.store_experience(
            state=state,
            action=action_value,
            reward=reward,
            value=value_estimate,
            log_prob=log_prob,
            done=done,
            feasible_actions=feasible_actions,
        )

        episode_reward += reward
        episode_vehicle_wait += step_metrics["vehicle_wait"]
        episode_ped_wait += step_metrics["pedestrian_wait"]
        episode_queue += step_metrics["queue"]
        episode_queue_delta += reward_components.get("queue_delta", 0.0)
        episode_invalid_repairs += info.get("invalid_actions", 0)
        episode_ped_crossing_steps += step_metrics["pedestrian_crossing_occupancy"]
        steps += 1
        state = next_state

    update_metrics = agent.update(batch_size=batch_size, num_epochs=update_epochs)

    metrics = {
        "episode_reward": episode_reward,
        "avg_vehicle_waiting_time": episode_vehicle_wait / max(steps, 1),
        "avg_pedestrian_waiting_time": episode_ped_wait / max(steps, 1),
        "avg_queue_length": episode_queue / max(steps, 1),
        "avg_queue_delta": episode_queue_delta / max(steps, 1),
        "constraint_violations": constraint_violations,
        "constraint_violation_rate": constraint_violations / max(steps, 1),
        "invalid_action_repairs": episode_invalid_repairs,
        "avg_valid_action_count": float(np.mean(valid_action_counts)) if valid_action_counts else 0.0,
        "pedestrian_crossing_step_fraction": episode_ped_crossing_steps / max(steps, 1),
        "steps": steps,
    }
    metrics.update(update_metrics)
    return metrics


def evaluate_episode(env, agent, num_episodes=5):
    """Run deterministic evaluation episodes without PPO updates."""
    all_metrics = {
        "episode_rewards": [],
        "avg_vehicle_waiting_times": [],
        "avg_pedestrian_waiting_times": [],
        "avg_queue_lengths": [],
        "avg_queue_deltas": [],
        "constraint_violation_rates": [],
        "invalid_action_repairs": [],
        "pedestrian_crossing_step_fractions": [],
    }

    for _ in range(num_episodes):
        state, _ = env.reset()
        agent.reset()

        episode_reward = 0.0
        episode_vehicle_wait = 0.0
        episode_ped_wait = 0.0
        episode_queue = 0.0
        episode_queue_delta = 0.0
        episode_invalid_repairs = 0
        episode_ped_crossing_steps = 0.0
        violations = 0
        steps = 0

        done = False
        while not done:
            feasible_actions_list = env.get_feasible_actions()
            feasible_actions = feasible_actions_list[0] if feasible_actions_list else [0]

            action_value, _, _ = agent.select_action(
                state,
                feasible_actions,
                deterministic=True,
            )
            action = np.array([action_value], dtype=np.int32)

            if action_value not in feasible_actions:
                violations += 1

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            step_metrics = collect_step_metrics(env)
            reward_components = info.get("reward_components", {})

            episode_reward += reward
            episode_vehicle_wait += step_metrics["vehicle_wait"]
            episode_ped_wait += step_metrics["pedestrian_wait"]
            episode_queue += step_metrics["queue"]
            episode_queue_delta += reward_components.get("queue_delta", 0.0)
            episode_invalid_repairs += info.get("invalid_actions", 0)
            episode_ped_crossing_steps += step_metrics["pedestrian_crossing_occupancy"]
            steps += 1
            state = next_state

        all_metrics["episode_rewards"].append(episode_reward)
        all_metrics["avg_vehicle_waiting_times"].append(episode_vehicle_wait / max(steps, 1))
        all_metrics["avg_pedestrian_waiting_times"].append(episode_ped_wait / max(steps, 1))
        all_metrics["avg_queue_lengths"].append(episode_queue / max(steps, 1))
        all_metrics["avg_queue_deltas"].append(episode_queue_delta / max(steps, 1))
        all_metrics["constraint_violation_rates"].append(violations / max(steps, 1))
        all_metrics["invalid_action_repairs"].append(episode_invalid_repairs)
        all_metrics["pedestrian_crossing_step_fractions"].append(
            episode_ped_crossing_steps / max(steps, 1)
        )

    return {
        "eval_avg_reward": float(np.mean(all_metrics["episode_rewards"])),
        "eval_std_reward": float(np.std(all_metrics["episode_rewards"])),
        "eval_avg_vehicle_waiting_time": float(np.mean(all_metrics["avg_vehicle_waiting_times"])),
        "eval_avg_pedestrian_waiting_time": float(np.mean(all_metrics["avg_pedestrian_waiting_times"])),
        "eval_avg_queue_length": float(np.mean(all_metrics["avg_queue_lengths"])),
        "eval_avg_queue_delta": float(np.mean(all_metrics["avg_queue_deltas"])),
        "eval_avg_constraint_violation_rate": float(np.mean(all_metrics["constraint_violation_rates"])),
        "eval_avg_invalid_action_repairs": float(np.mean(all_metrics["invalid_action_repairs"])),
        "eval_pedestrian_crossing_step_fraction": float(
            np.mean(all_metrics["pedestrian_crossing_step_fractions"])
        ),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train 1x1 Action-Constrained PPO for traffic signal control."
    )
    parser.add_argument(
        "--sumo-cfg-file",
        type=str,
        default="sumo_rl/nets/single-intersection/single_intersection.sumocfg",
        help="Path to SUMO configuration file.",
    )
    parser.add_argument(
        "--controlled-tls",
        type=str,
        default=None,
        help="Comma-separated traffic light ids to control. Leave empty for all TLS.",
    )
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--num-eval-episodes", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-clip-epsilon", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--reward-mode",
        type=str,
        choices=["waiting_time", "queue_delta", "hybrid"],
        default="hybrid",
        help="Training reward used by the SUMO environment.",
    )
    parser.add_argument("--queue-reward-weight", type=float, default=1.0)
    parser.add_argument("--vehicle-wait-weight", type=float, default=0.01)
    parser.add_argument("--pedestrian-wait-weight", type=float, default=0.02)
    parser.add_argument("--violation-penalty", type=float, default=50.0)
    parser.add_argument(
        "--alpha-vehicle",
        type=float,
        default=None,
        help="Deprecated alias for --vehicle-wait-weight.",
    )
    parser.add_argument(
        "--beta-pedestrian",
        type=float,
        default=None,
        help="Deprecated alias for --pedestrian-wait-weight.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--delta-time", type=int, default=1)
    parser.add_argument("--yellow-time", type=int, default=5)
    parser.add_argument("--min-green-time", type=int, default=10)
    parser.add_argument("--max-green-time", type=int, default=60)
    parser.add_argument("--sim-end-time", type=int, default=3600)
    parser.add_argument("--use-gui", action="store_true", help="Show SUMO GUI during training.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.alpha_vehicle is not None:
        args.vehicle_wait_weight = args.alpha_vehicle
    if args.beta_pedestrian is not None:
        args.pedestrian_wait_weight = args.beta_pedestrian

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = create_run_dir(args.output_dir, args.run_name)
    checkpoint_dir = run_dir / "checkpoints"
    train_history_path = run_dir / "train_history.json"
    eval_history_path = run_dir / "eval_history.json"
    train_jsonl_path = run_dir / "train_metrics.jsonl"
    eval_jsonl_path = run_dir / "eval_metrics.jsonl"

    config = vars(args).copy()
    config["run_dir"] = str(run_dir)
    write_json(run_dir / "config.json", config)

    print(f"[INFO] Run directory: {run_dir}")
    print(f"[INFO] Config saved to {run_dir / 'config.json'}")

    env = SUMOEnvironment(
        sumo_cfg_file=args.sumo_cfg_file,
        delta_time=args.delta_time,
        yellow_time=args.yellow_time,
        min_green_time=args.min_green_time,
        max_green_time=args.max_green_time,
        end_time=args.sim_end_time,
        controlled_tls=parse_controlled_tls(args.controlled_tls),
        use_gui=args.use_gui,
        reward_mode=args.reward_mode,
        queue_reward_weight=args.queue_reward_weight,
        vehicle_wait_weight=args.vehicle_wait_weight,
        pedestrian_wait_weight=args.pedestrian_wait_weight,
        violation_penalty=args.violation_penalty,
    )

    state, _ = env.reset()
    state_dim = len(state)
    num_intersections = len(env.traffic_signals)
    if num_intersections != 1:
        env.close()
        raise NotImplementedError(
            "This PPO training script is currently for one controlled traffic light. "
            "Use --controlled-tls to choose one TLS, such as --controlled-tls B1 for the 3x3 center."
        )
    action_dim = int(env.action_space.nvec[0])
    env.close()

    print(f"[INFO] State dimension: {state_dim}")
    print(f"[INFO] Action dimension: {action_dim}")
    print(f"[INFO] Reward mode: {args.reward_mode}")

    agent = ActionConstrainedPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_intersections=num_intersections,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ppo_clip_epsilon=args.ppo_clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_dim=args.hidden_dim,
        value_clip_epsilon=args.value_clip_epsilon,
        device=args.device,
    )

    train_history = []
    eval_history = []
    best_eval_reward = -float("inf")
    best_model_path = run_dir / "best_agent.pt"

    print(f"[INFO] Starting PPO training for {args.num_episodes} episodes")

    try:
        for episode in range(1, args.num_episodes + 1):
            metrics = train_episode(
                env,
                agent,
                batch_size=args.batch_size,
                update_epochs=args.update_epochs,
            )
            metrics["episode"] = episode
            metrics["timestamp"] = datetime.now().isoformat()
            train_history.append(metrics)
            append_jsonl(train_jsonl_path, metrics)
            write_json(train_history_path, train_history)

            print(
                f"[Episode {episode}/{args.num_episodes}] "
                f"Reward: {metrics['episode_reward']:.2f}, "
                f"Avg Wait V/P: {metrics['avg_vehicle_waiting_time']:.2f}/"
                f"{metrics['avg_pedestrian_waiting_time']:.2f}, "
                f"Avg Queue: {metrics['avg_queue_length']:.2f}, "
                f"Violations: {metrics['constraint_violations']}"
            )

            if args.checkpoint_interval > 0 and episode % args.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"agent_ep_{episode:04d}.pt"
                agent.save(
                    str(checkpoint_path),
                    episode=episode,
                    extra={"train_metrics": metrics, "run_dir": str(run_dir)},
                )

            if args.eval_interval > 0 and episode % args.eval_interval == 0:
                print(f"[INFO] Evaluating deterministically for {args.num_eval_episodes} episodes...")
                eval_metrics = evaluate_episode(
                    env,
                    agent,
                    num_episodes=args.num_eval_episodes,
                )
                eval_metrics["episode"] = episode
                eval_metrics["timestamp"] = datetime.now().isoformat()
                eval_history.append(eval_metrics)
                append_jsonl(eval_jsonl_path, eval_metrics)
                write_json(eval_history_path, eval_history)

                if eval_metrics["eval_avg_reward"] > best_eval_reward:
                    best_eval_reward = eval_metrics["eval_avg_reward"]
                    agent.save(
                        str(best_model_path),
                        episode=episode,
                        extra={"eval_metrics": eval_metrics, "run_dir": str(run_dir)},
                    )

                print(
                    f"[Eval {episode}] "
                    f"Avg Reward: {eval_metrics['eval_avg_reward']:.2f}, "
                    f"Avg Wait V/P: {eval_metrics['eval_avg_vehicle_waiting_time']:.2f}/"
                    f"{eval_metrics['eval_avg_pedestrian_waiting_time']:.2f}, "
                    f"Avg Queue: {eval_metrics['eval_avg_queue_length']:.2f}, "
                    f"Violation Rate: {eval_metrics['eval_avg_constraint_violation_rate']:.4f}"
                )

        final_model_path = run_dir / "agent_final.pt"
        agent.save(
            str(final_model_path),
            episode=args.num_episodes,
            extra={"run_dir": str(run_dir), "final_train_metrics": train_history[-1] if train_history else {}},
        )

        summary = {
            "run_dir": str(run_dir),
            "final_model": str(final_model_path),
            "best_model": str(best_model_path) if best_model_path.exists() else None,
            "num_train_episodes": len(train_history),
            "num_eval_points": len(eval_history),
            "best_eval_reward": best_eval_reward if eval_history else None,
            "final_train_metrics": train_history[-1] if train_history else {},
            "final_eval_metrics": eval_history[-1] if eval_history else {},
        }
        write_json(run_dir / "run_summary.json", summary)

        print(f"\n[INFO] Training complete.")
        print(f"[INFO] Final model: {final_model_path}")
        print(f"[INFO] Run records: {run_dir}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
