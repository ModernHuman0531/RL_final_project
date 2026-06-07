"""
Evaluate traffic-signal baselines and learned agents.

By default, evaluates:
  - Fixed-Time
  - Max Pressure
  - SOTL

Optionally evaluates:
  - DQN-AR with --dqn-model
  - SPRe+ with --dqn-model
  - Action-Constrained PPO with --ppo-model or --trained-model

Use --agent to evaluate or visualize one agent at a time.
Results are printed and saved to a unique folder.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sumo_rl.agents.action_constrained_ppo_agent import ActionConstrainedPPOAgent
from sumo_rl.agents.base_agent import DQNAgent
from sumo_rl.agents.fixed_time_agent import FixedTimeAgent
from sumo_rl.agents.max_pressure_agent import MaxPressureAgent
from sumo_rl.agents.sotl_agent import SOTLAgent
from sumo_rl.agents.spre_plus_agent import SPRePlusAgent
from sumo_rl.environement.sumo_env import SUMOEnvironment


DEFAULT_CFG_FILE = "sumo_rl/nets/single-intersection/single_intersection.sumocfg"
AGENT_CHOICES = ["all", "fixed-time", "max-pressure", "sotl", "dqn-ar", "spre-plus", "ppo"]


def parse_controlled_tls(value):
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def make_run_dir(base_dir: str, run_name: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = run_name or f"baseline_compare_{timestamp}"
    run_dir = os.path.join(base_dir, name)
    counter = 1
    unique_dir = run_dir
    while os.path.exists(unique_dir):
        unique_dir = f"{run_dir}_{counter}"
        counter += 1
    os.makedirs(unique_dir, exist_ok=True)
    return unique_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SUMO-RL baselines and learned agents.")
    parser.add_argument("--sumo-cfg-file", type=str, default=DEFAULT_CFG_FILE)
    parser.add_argument("--controlled-tls", type=str, default=None, help="Comma-separated TLS ids, for example B1.")
    parser.add_argument("--num-episodes", type=int, default=1)
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
    parser.add_argument("--fixed-cycle", type=int, default=30)
    parser.add_argument("--sotl-kappa", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--use-gui", action="store_true")
    parser.add_argument(
        "--agent",
        choices=AGENT_CHOICES,
        default="all",
        help="Evaluate all agents or exactly one selected agent.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dqn-model", type=str, default=None, help="Path to trained DQN-AR checkpoint.")
    parser.add_argument("--include-untrained-dqn", action="store_true", help="Evaluate untrained DQN/SPRe+ if --dqn-model is absent.")
    parser.add_argument(
        "--ppo-model",
        "--trained-model",
        dest="ppo_model",
        type=str,
        default=None,
        help="Path to trained Action-Constrained PPO checkpoint.",
    )
    parser.add_argument("--spre-use-scipy", action="store_true", help="Use scipy SLSQP projection for SPRe+ if scipy is installed.")

    args = parser.parse_args()
    if args.agent == "ppo" and not args.ppo_model:
        parser.error("--agent ppo requires --ppo-model PATH.")
    if args.agent in {"dqn-ar", "spre-plus"} and not args.dqn_model and not args.include_untrained_dqn:
        parser.error(f"--agent {args.agent} requires --dqn-model PATH, or --include-untrained-dqn for a smoke test.")
    return args


def make_env(args, use_gui=None):
    if use_gui is None:
        use_gui = args.use_gui
    return SUMOEnvironment(
        sumo_cfg_file=args.sumo_cfg_file,
        delta_time=args.delta_time,
        yellow_time=args.yellow_time,
        min_green_time=args.min_green_time,
        max_green_time=args.max_green_time,
        end_time=args.sim_end_time,
        controlled_tls=parse_controlled_tls(args.controlled_tls),
        use_gui=use_gui,
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
            "DQN/SPRe+/PPO helpers assume one shared state/action size. Use "
            "--controlled-tls B1 for a single 3x3 intersection."
        )

    return {
        "state_dim": int(state_dims[0]),
        "joint_state_dim": int(len(state)),
        "action_dim": int(action_dims[0]),
        "num_intersections": len(signals),
        "traffic_signal_ids": list(env.traffic_signals.keys()),
        "green_phases": {sid: signal.green_phases for sid, signal in env.traffic_signals.items()},
    }


def build_experiments(args, dims):
    experiments = []

    def wants(agent_name):
        return args.agent in {"all", agent_name}

    if wants("fixed-time"):
        fixed = FixedTimeAgent(
            cycle_time=args.fixed_cycle,
            num_intersections=dims["num_intersections"],
            action_dims=[dims["action_dim"]] * dims["num_intersections"],
        )
        experiments.append((
            f"Fixed-Time ({args.fixed_cycle}s cycle)",
            fixed,
            lambda state, signals, agent=fixed: agent.select_action(traffic_signals=signals),
        ))

    if wants("max-pressure"):
        max_pressure = MaxPressureAgent(num_intersections=dims["num_intersections"])
        experiments.append((
            "Max Pressure",
            max_pressure,
            lambda state, signals, agent=max_pressure: agent.select_action(signals),
        ))

    if wants("sotl"):
        sotl = SOTLAgent(kappa=args.sotl_kappa, num_intersections=dims["num_intersections"])
        experiments.append((
            f"SOTL (kappa={args.sotl_kappa})",
            sotl,
            lambda state, signals, agent=sotl: agent.select_action(signals),
        ))

    dqn_agent = None
    dqn_family_requested = (
        args.agent in {"dqn-ar", "spre-plus"}
        or (args.agent == "all" and (args.dqn_model or args.include_untrained_dqn))
    )
    if dqn_family_requested:
        dqn_agent = DQNAgent(
            state_dim=dims["state_dim"],
            action_dim=dims["action_dim"],
            num_intersections=dims["num_intersections"],
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_steps=1,
            device=args.device,
        )
        if args.dqn_model:
            print(f"[INFO] Loading DQN-AR model from {args.dqn_model}")
            dqn_agent.load(args.dqn_model, load_optimizer=False)
        else:
            print("[WARN] Evaluating untrained DQN-AR because --include-untrained-dqn was set.")
        dqn_agent.epsilon = 0.0
        dqn_agent.epsilon_end = 0.0

        if wants("dqn-ar"):
            experiments.append((
                "DQN-AR",
                dqn_agent,
                lambda state, signals, agent=dqn_agent: agent.select_action(state, signals),
            ))

        if wants("spre-plus"):
            spre = SPRePlusAgent(
                action_dim=dims["action_dim"],
                num_intersections=dims["num_intersections"],
                use_scipy=args.spre_use_scipy,
                state_dim=dims["state_dim"],
            )
            spre.set_policy(dqn_agent.get_q_fn())
            experiments.append((
                "SPRe+ (DQN policy)",
                spre,
                lambda state, signals, agent=spre: agent.select_action(state, signals),
            ))
    elif args.agent == "all":
        print("[INFO] Skipping DQN-AR and SPRe+. Pass --dqn-model PATH to evaluate them.")

    if wants("ppo") and args.ppo_model:
        if dims["num_intersections"] != 1:
            raise RuntimeError("The current PPO evaluation path supports one controlled traffic signal. Use --controlled-tls B1 for 3x3.")
        print(f"[INFO] Loading PPO model from {args.ppo_model}")
        ppo = ActionConstrainedPPOAgent(
            state_dim=dims["joint_state_dim"],
            action_dim=dims["action_dim"],
            num_intersections=dims["num_intersections"],
            device=args.device,
        )
        ppo.load(args.ppo_model, load_optimizers=False)
        experiments.append((
            "Action-Constrained PPO",
            ppo,
            lambda state, signals, agent=ppo: ppo_action_fn(state, signals, agent),
        ))

    if not experiments:
        raise RuntimeError(f"No experiments were built for --agent {args.agent}. Check the requested model paths.")

    return experiments


def ppo_action_fn(state, signals, agent):
    feasible = signals[0].get_valid_actions()
    feasible_actions = [idx for idx, valid in enumerate(feasible) if valid == 1]
    action, _log_prob, _value = agent.select_action(
        state,
        feasible_actions,
        deterministic=True,
    )
    return np.array([action], dtype=np.int32)


def run_episode(env, agent, action_fn, episode_idx: int):
    state, _ = env.reset()
    agent.reset()

    total_reward = 0.0
    waiting_times = []
    pedestrian_waiting_times = []
    queue_lengths = []
    crossing_occupancies = []
    violations = 0
    steps = 0
    done = False

    while not done:
        signals = list(env.traffic_signals.values())
        action = np.asarray(action_fn(state, signals), dtype=np.int32)
        if action.ndim == 0:
            action = action.reshape(1)

        for idx, signal in enumerate(signals):
            proposed = int(action[idx])
            valid = signal.get_valid_actions()
            if proposed < 0 or proposed >= len(valid) or valid[proposed] == 0:
                violations += 1

        state, reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated

        waiting_times.append(env.get_total_vehicle_waiting_time())
        pedestrian_waiting_times.append(env.get_total_pedestrian_waiting_time())
        queue_lengths.append(env.get_total_vehicle_queue())
        crossing_occupancies.append(env.get_total_pedestrian_crossing_occupancy())
        total_reward += float(reward)
        steps += 1

    return {
        "episode": int(episode_idx),
        "avg_waiting_time": float(np.mean(waiting_times)) if waiting_times else 0.0,
        "avg_pedestrian_waiting_time": float(np.mean(pedestrian_waiting_times)) if pedestrian_waiting_times else 0.0,
        "avg_queue_length": float(np.mean(queue_lengths)) if queue_lengths else 0.0,
        "avg_pedestrian_crossing_occupancy": float(np.mean(crossing_occupancies)) if crossing_occupancies else 0.0,
        "violation_rate": float(violations / max(steps, 1)),
        "violations": int(violations),
        "total_reward": float(total_reward),
        "steps": int(steps),
    }


def aggregate_results(agent_name, episode_results):
    metric_keys = [
        "avg_waiting_time",
        "avg_pedestrian_waiting_time",
        "avg_queue_length",
        "avg_pedestrian_crossing_occupancy",
        "violation_rate",
        "total_reward",
        "steps",
    ]
    row = {"agent": agent_name, "episodes": len(episode_results)}
    for key in metric_keys:
        values = [result[key] for result in episode_results]
        row[key] = float(np.mean(values)) if values else 0.0
    row["violations"] = int(sum(result["violations"] for result in episode_results))
    return row


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def write_csv(path, rows):
    fieldnames = [
        "agent",
        "episodes",
        "avg_waiting_time",
        "avg_pedestrian_waiting_time",
        "avg_queue_length",
        "avg_pedestrian_crossing_occupancy",
        "violation_rate",
        "violations",
        "total_reward",
        "steps",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    run_dir = make_run_dir(args.output_dir, args.run_name)

    # Probe dimensions headlessly so --use-gui opens one visible SUMO window only
    # for the actual evaluation episode.
    probe_env = make_env(args, use_gui=False)
    dims = probe_dimensions(probe_env)
    probe_env.close()

    config = vars(args).copy()
    config.update(dims)
    config["run_dir"] = run_dir
    write_json(os.path.join(run_dir, "config.json"), config)

    experiments = build_experiments(args, dims)
    env = make_env(args)

    all_episode_rows = []
    comparison_rows = []

    print("\n=== Baseline & Method Evaluation ===")
    print(f"Run dir: {run_dir}")
    print(f"Signals: {dims['traffic_signal_ids']} | State dim: {dims['joint_state_dim']} | Action dim: {dims['action_dim']}\n")

    for agent_name, agent, action_fn in experiments:
        episode_results = []
        for episode in range(1, args.num_episodes + 1):
            result = run_episode(env, agent, action_fn, episode)
            result["agent"] = agent_name
            episode_results.append(result)
            all_episode_rows.append(result)
        comparison_rows.append(aggregate_results(agent_name, episode_results))

    env.close()

    write_json(os.path.join(run_dir, "comparison.json"), comparison_rows)
    write_json(os.path.join(run_dir, "episodes.json"), all_episode_rows)
    write_csv(os.path.join(run_dir, "comparison.csv"), comparison_rows)

    with open(os.path.join(run_dir, "episodes.jsonl"), "w", encoding="utf-8") as file:
        for row in all_episode_rows:
            file.write(json.dumps(row) + "\n")

    col = "{:<30} {:>13} {:>15} {:>12} {:>16} {:>12}"
    print(col.format("Agent", "Avg Wait", "Avg Ped Wait", "Avg Queue", "Violation Rate", "Reward"))
    print("-" * 104)
    for row in comparison_rows:
        print(col.format(
            row["agent"],
            f"{row['avg_waiting_time']:.2f}",
            f"{row['avg_pedestrian_waiting_time']:.2f}",
            f"{row['avg_queue_length']:.2f}",
            f"{row['violation_rate']:.4f}",
            f"{row['total_reward']:.1f}",
        ))

    print(f"\nSaved comparison files in: {run_dir}\n")


if __name__ == "__main__":
    main()
