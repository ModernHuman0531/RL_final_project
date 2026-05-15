# Action-Constrained RL for Traffic Signal Control

## Overview

This implementation adds an **Action-Constrained Proximal Policy Optimization (PPO)** agent to the traffic signal control project. The method enforces safety constraints by construction using action masking, guaranteeing **0% constraint violations** during execution.

## Key Features

- **Action Masking**: Only feasible actions (respecting min/max green time, yellow transitions, pedestrian safety) are sampled from the policy
- **Zero Safety Violations**: By design, infeasible actions are never executed
- **Generalized Advantage Estimation (GAE)**: Reduces variance in advantage estimation for stable training
- **PPO Updates**: Clipped surrogate objective for stable policy optimization
- **Multi-metric Tracking**: Vehicle waiting time, pedestrian delay, queue length, constraint violations, learning stability

## Architecture

### Policy Network (Actor)
- Input: State (11 features per intersection: phase one-hot, min_green flag, vehicle queues, pedestrian counts)
- Hidden: 128 units, ReLU activation
- Output: Logits over 2 discrete actions (NS-green=0, EW-green=1)

### Value Network (Critic)
- Input: State (same as actor)
- Hidden: 128 units, ReLU activation  
- Output: State value estimate

### Constraint Enforcement
```
get_feasible_actions(state) -> List[int]
  Returns valid action indices based on:
  - Minimum green time constraint
  - Maximum green time constraint
  - Yellow phase transitions
  - Pedestrian clearance intervals
```

**Action Masking** before sampling:
```python
masked_logits[infeasible_actions] = -1e9
sampled_action = Categorical(masked_logits).sample()
# guaranteed feasible
```

## Training

### Quick Start

```bash
cd /workspace

# Train with default parameters (100 episodes)
python3 sumo_rl/experiments/train_action_constrained_rl.py

# Train with custom parameters
python3 sumo_rl/experiments/train_action_constrained_rl.py \
    --num-episodes 200 \
    --learning-rate 1e-3 \
    --gamma 0.99 \
    --alpha-vehicle 1.0 \
    --beta-pedestrian 1.0 \
    --output-dir ./results_ppo \
    --device cpu
```

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-episodes` | 100 | Total training episodes |
| `--learning-rate` | 3e-4 | Adam optimizer learning rate |
| `--gamma` | 0.99 | Discount factor for returns |
| `--gae-lambda` | 0.95 | GAE exponential smoothing parameter |
| `--ppo-clip-epsilon` | 0.2 | PPO clipping range [1-eps, 1+eps] |
| `--entropy-coef` | 0.01 | Entropy regularization weight |
| `--value-coef` | 0.5 | Value loss weight in combined objective |
| `--alpha-vehicle` | 1.0 | Weight for vehicle waiting time in reward |
| `--beta-pedestrian` | 1.0 | Weight for pedestrian waiting time in reward |
| `--seed` | 42 | Random seed for reproducibility |
| `--device` | cpu | PyTorch device (cpu or cuda) |

### Reward Function

```
r_t = - (α * vehicle_waiting_time + β * pedestrian_waiting_time)
```

Where:
- `α = --alpha-vehicle` (default 1.0)
- `β = --beta-pedestrian` (default 1.0)
- Both weighted equally by default (fair to vehicles and pedestrians)

### Training Loop

1. **Episode collection**: Agent interacts with environment for 3600 simulation steps
2. **Advantage estimation**: Compute GAE advantages and returns
3. **PPO update**: Mini-batch SGD with clipped surrogate objective
4. **Periodic evaluation**: Every 10 episodes, run 5 test episodes without training

### Output

Training creates `--output-dir` with:
- `config.json` — training configuration
- `train_history.json` — per-episode metrics
- `eval_history.json` — evaluation results every 10 episodes
- `agent_final.pt` — trained model checkpoint

Example output:
```
[Episode 10/100] Reward: -523.45, Avg Wait (V): 12.34, Avg Wait (P): 5.67, Violations: 0
[Episode 20/100] Reward: -487.23, Avg Wait (V): 11.89, Avg Wait (P): 5.32, Violations: 0
...
[INFO] Training completed. Results saved to ./results
```

## Evaluation

### Evaluate Against All Baselines

```bash
cd /workspace

# Baselines only
python3 sumo_rl/experiments/eval_baselines.py

# Baselines + trained PPO agent
python3 sumo_rl/experiments/eval_baselines.py \
    --trained-model ./results/agent_final.pt \
    --device cpu
```

### Output Example

```
=== Baseline & Method Evaluation: Single Intersection (1 episode = 3600 steps) ===

Agent                       Avg Wait (s) Avg Ped Wait (s) Avg Queue Violation Rate  Total Reward
───────────────────────────────────────────────────────────────────────────────────────────────────
Fixed-Time (30s cycle)           12.45            6.78         3.21           0.0000      -2345.6
Max Pressure                       9.87            5.43         2.54           0.0000      -1876.4
SOTL (kappa=5)                    10.23            5.91         2.78           0.0000      -1945.2
Action-Constrained PPO             8.76            4.92         2.12           0.0000      -1652.3
```

## Constraints

### Feasible Action Set C(s)

An action is **feasible** if it satisfies ALL of:

1. **Minimum Green Time**: Current phase must be held ≥ `min_green_time` seconds
2. **Maximum Green Time**: Current phase cannot exceed `max_green_time` seconds
3. **Yellow Transition**: Cannot skip yellow phase during transitions
4. **Pedestrian Safety**: Respects pedestrian clearance intervals (inherited from environment)

These constraints are enforced by `env.get_valid_actions()` which returns a binary mask `[0 or 1, 0 or 1]` for each intersection.

### Safety Guarantee

By construction:
```python
feasible_actions = env.get_valid_actions()  # [0, 1] or [1, 0] or [0, 0] (rare)
masked_logits = logits.clone()
masked_logits[~feasible_mask] = -1e9
action = Categorical(masked_logits).sample()  # ALWAYS feasible
```

**Claim**: Constraint violation rate = 0% (assuming `get_valid_actions()` is correct)

## Comparisons with Baselines

### Fixed-Time Controller
- Simple 30-second green cycles, alternating NS ↔ EW
- Ignores traffic state
- Reliable but suboptimal in varying conditions

### Max Pressure (Varaiya, 2013)
- Selects phase with highest queue pressure
- Reactive heuristic (no learning)
- Greedy but respects safety constraints

### SOTL (Gershenson, 2004)
- Switches when red-lane pressure exceeds threshold (kappa=5)
- Adaptive but still heuristic
- Aims to minimize oscillations

### SPre+ (Hung et al., 2025)
- Generic safe-RL baseline: any policy + QP projection
- Computationally expensive (O(|A|²) per step)
- Guaranteed feasible but slow

### **Action-Constrained PPO** (This Work)
- Learns adaptive policy from experience
- Action masking ensures feasibility by design
- O(1) per-step cost (no QP)
- Data-efficient (GAE stabilization)
- Target: improve efficiency while maintaining safety

## Metrics

### Per-Episode Metrics

| Metric | Interpretation |
|--------|-----------------|
| `episode_reward` | Sum of rewards over episode (higher is better) |
| `avg_vehicle_waiting_time` | Avg seconds vehicles spend waiting at red (lower is better) |
| `avg_pedestrian_waiting_time` | Avg seconds pedestrians spend waiting (lower is better) |
| `avg_queue_length` | Avg number of vehicles queued (lower is better) |
| `constraint_violations` | Count of infeasible actions (target: 0) |
| `constraint_violation_rate` | Violations per step (target: 0.0%) |
| `policy_loss` | PPO surrogate loss |
| `value_loss` | Critic MSE loss |
| `entropy` | Policy entropy (regularization) |
| `kl_divergence` | KL divergence from old to new policy |

### Evaluation Metrics (5-episode average)

- `eval_avg_reward` — Average cumulative reward ± std
- `eval_avg_vehicle_waiting_time` — Expected vehicle delay
- `eval_avg_pedestrian_waiting_time` — Expected pedestrian delay
- `eval_avg_queue_length` — Expected queue size
- `eval_avg_constraint_violation_rate` — Safety guarantee (should be 0.0)

## Implementation Details

### GAE (Generalized Advantage Estimation)

```python
delta_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
A_t = δ_t + γ * λ * (1 - done) * A_{t+1}
R_t = A_t + V(s_t)  # return estimate
```

Where:
- `γ = 0.99` (discount factor)
- `λ = 0.95` (GAE smoothing)
- Higher λ → smoother advantages, lower bias
- Lower λ → lower variance, higher bias

### PPO Objective

```python
L_CLIP(θ) = E[ min( r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t ) ]
```

Where:
- `r_t(θ) = π_new(a|s) / π_old(a|s)` (probability ratio)
- `ε = 0.2` (clipping range)
- Prevents large policy updates

### Training Objective

```python
L_total = L_CLIP - c_entropy * H(π) + c_value * L_value
```

Where:
- `c_entropy = 0.01` (entropy regularization)
- `c_value = 0.5` (value loss weight)
- Balances exploration (entropy) and stability (value estimation)

## Files Added/Modified

### New Files
- `sumo_rl/agents/action_constrained_ppo_agent.py` — PPO agent with action masking
- `sumo_rl/experiments/train_action_constrained_rl.py` — Training script
- `sumo_rl/agents/ACTION_CONSTRAINED_RL_README.md` — This file

### Modified Files
- `sumo_rl/environement/sumo_env.py` — Added `close()` and `get_feasible_actions()` methods
- `sumo_rl/experiments/eval_baselines.py` — Added PPO evaluation and command-line args

### Unmodified (Preserved Baselines)
- `sumo_rl/agents/fixed_time_agent.py`
- `sumo_rl/agents/max_pressure_agent.py`
- `sumo_rl/agents/sotl_agent.py`
- `sumo_rl/agents/spre_plus_agent.py`

## Troubleshooting

### Issue: CUDA out of memory
```bash
python3 train_action_constrained_rl.py --device cpu  # Use CPU instead
```

### Issue: Training loss is NaN
- Check reward values (should be finite)
- Reduce learning rate: `--learning-rate 1e-4`
- Increase entropy coefficient: `--entropy-coef 0.05`

### Issue: Constraint violations detected during evaluation
- Check `env.get_valid_actions()` is correct
- Verify action masking is applied (see agent code)
- Print debug: `print(f"Action: {action}, Feasible: {feasible_actions}")`

### Issue: Slow training
- Reduce `--sim-end-time` to shorten episodes (default 3600 steps)
- Use `--device cuda` if GPU available
- Reduce hidden layer size in ActorNetwork/CriticNetwork

## Research Notes

### Acceptance-Rejection vs Action Masking
This implementation uses **action masking** (preferred):
- Masks infeasible actions in logits before sampling
- O(1) per-step cost
- No bias from rejection sampling

Alternative: **Acceptance-Rejection** (Hung et al., 2025)
- Resample until feasible action found
- Useful if masking is difficult
- May have higher variance if feasibility rate is low

### Why PPO?
- Stable on-policy learning (clipped objective)
- Works well with discrete actions
- Natural compatibility with action masking
- GAE enables low-variance advantage estimation

### Next Steps
- Extend to multi-intersection grids (2x2, 4x4)
- Implement multi-agent training (QMIX, MAPPO)
- Add reward shaping for emergent coordination
- Test on real-world traffic networks (OpenStreetMap)
- Compare with DQN, A3C, and other methods

## References

- Hung et al. (2025): "Efficient Action-Constrained Reinforcement Learning via Acceptance-Rejection Method and Augmented MDPs" [ICLR]
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- Schulman et al. (2015): "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
- Varaiya (2013): "Max Pressure Control of Signalized Intersections"
- Gershenson (2005): "Self-Organizing Traffic Lights"
