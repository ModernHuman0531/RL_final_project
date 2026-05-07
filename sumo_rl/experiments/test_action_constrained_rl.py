"""
Quick validation script for ActionConstrainedPPOAgent.

Tests:
1. Environment initialization and stepping
2. Feasible action generation
3. Policy network forward pass
4. Action masking
5. Single episode execution
6. Model save/load

Usage:
    python3 sumo_rl/experiments/test_action_constrained_rl.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sumo_rl.environement.sumo_env import SUMOEnvironment
from sumo_rl.agents.action_constrained_ppo_agent import ActionConstrainedPPOAgent


def test_environment():
    """Test environment initialization and methods."""
    print("\n[TEST 1] Environment Initialization")
    print("-" * 50)
    
    env = SUMOEnvironment(
        sumo_cfg_file="sumo_rl/nets/single-intersection/single_intersection.sumocfg",
        delta_time=1,
        yellow_time=5,
        min_green_time=10,
        max_green_time=60,
        end_time=3600,
        use_gui=False,
    )
    
    state, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  State shape: {state.shape}")
    print(f"  State dtype: {state.dtype}")
    print(f"  State range: [{state.min():.3f}, {state.max():.3f}]")
    
    # Test feasible actions
    feasible_actions = env.get_feasible_actions()
    print(f"✓ Feasible actions retrieved: {feasible_actions}")
    
    # Test step
    action = np.array([0])  # Action 0 = NS green
    next_state, reward, done, truncated, info = env.step(action)
    print(f"✓ Environment step successful")
    print(f"  Reward: {reward:.4f}")
    print(f"  Done: {done}, Truncated: {truncated}")
    
    env.close()
    print(f"✓ Environment closed successfully")
    return True


def test_agent_network():
    """Test policy and value networks."""
    print("\n[TEST 2] Agent Networks")
    print("-" * 50)
    
    import torch
    
    state_dim = 11
    action_dim = 2
    
    agent = ActionConstrainedPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_intersections=1,
        device="cpu",
    )
    
    # Test forward pass
    dummy_state = torch.randn(state_dim)
    with torch.no_grad():
        logits = agent.actor(dummy_state)
        value = agent.critic(dummy_state)
    
    print(f"✓ Actor forward pass: logits shape {logits.shape}")
    print(f"✓ Critic forward pass: value shape {value.shape}")
    
    return True


def test_action_masking():
    """Test action masking mechanism."""
    print("\n[TEST 3] Action Masking")
    print("-" * 50)
    
    state_dim = 11
    action_dim = 2
    
    agent = ActionConstrainedPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_intersections=1,
        device="cpu",
    )
    
    # Create a dummy state
    dummy_state = np.random.randn(state_dim).astype(np.float32)
    
    # Test with both actions feasible
    feasible_actions_1 = [0, 1]
    action_1, log_prob_1, value_1 = agent.select_action(dummy_state, feasible_actions_1)
    print(f"✓ Both actions feasible: action={action_1}, log_prob={log_prob_1:.4f}")
    assert action_1 in feasible_actions_1, f"Action {action_1} not in feasible set {feasible_actions_1}"
    
    # Test with only action 0 feasible
    feasible_actions_2 = [0]
    action_2, log_prob_2, value_2 = agent.select_action(dummy_state, feasible_actions_2)
    print(f"✓ Only action 0 feasible: action={action_2}, log_prob={log_prob_2:.4f}")
    assert action_2 == 0, f"Action {action_2} should be 0 (only feasible action)"
    
    # Test with only action 1 feasible
    feasible_actions_3 = [1]
    action_3, log_prob_3, value_3 = agent.select_action(dummy_state, feasible_actions_3)
    print(f"✓ Only action 1 feasible: action={action_3}, log_prob={log_prob_3:.4f}")
    assert action_3 == 1, f"Action {action_3} should be 1 (only feasible action)"
    
    print(f"✓ Action masking works correctly!")
    return True


def test_gae_and_update():
    """Test GAE computation and PPO update."""
    print("\n[TEST 4] GAE & PPO Update")
    print("-" * 50)
    
    state_dim = 11
    action_dim = 2
    
    agent = ActionConstrainedPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_intersections=1,
        device="cpu",
    )
    
    # Simulate a short trajectory
    trajectory_length = 10
    dummy_state = np.random.randn(state_dim).astype(np.float32)
    
    for t in range(trajectory_length):
        feasible_actions = [0, 1]
        action, log_prob, value = agent.select_action(dummy_state, feasible_actions)
        reward = np.random.randn()  # Random reward
        done = (t == trajectory_length - 1)
        
        agent.store_experience(
            state=dummy_state,
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=done,
        )
    
    print(f"✓ Stored {trajectory_length} experiences")
    
    # Update agent
    metrics = agent.update(batch_size=4, num_epochs=2)
    print(f"✓ PPO update completed")
    print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
    print(f"  Value Loss: {metrics['value_loss']:.4f}")
    print(f"  Entropy: {metrics['entropy']:.4f}")
    print(f"  KL Divergence: {metrics['kl_divergence']:.4f}")
    
    # Check that metrics are finite
    for key, value in metrics.items():
        assert np.isfinite(value), f"Metric {key} is not finite: {value}"
    
    print(f"✓ All metrics are finite!")
    return True


def test_model_save_load():
    """Test model checkpoint save/load."""
    print("\n[TEST 5] Model Save/Load")
    print("-" * 50)
    
    import torch
    import tempfile
    
    state_dim = 11
    action_dim = 2
    
    agent1 = ActionConstrainedPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_intersections=1,
        device="cpu",
    )
    
    # Get a reference output
    dummy_state = torch.randn(state_dim)
    with torch.no_grad():
        output1 = agent1.actor(dummy_state).clone()
    
    # Save model
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model_path = f.name
    
    agent1.save(model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Load into new agent
    agent2 = ActionConstrainedPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_intersections=1,
        device="cpu",
    )
    
    agent2.load(model_path)
    print(f"✓ Model loaded from {model_path}")
    
    # Compare outputs
    with torch.no_grad():
        output2 = agent2.actor(dummy_state)
    
    diff = (output1 - output2).abs().max().item()
    print(f"✓ Output difference: {diff:.6f} (should be ~0)")
    assert diff < 1e-5, f"Loaded model differs from original: diff={diff}"
    
    # Cleanup
    os.remove(model_path)
    print(f"✓ Model save/load verified!")
    
    return True


def test_full_episode():
    """Test a single training episode."""
    print("\n[TEST 6] Full Episode Execution")
    print("-" * 50)
    
    env = SUMOEnvironment(
        sumo_cfg_file="sumo_rl/nets/single-intersection/single_intersection.sumocfg",
        delta_time=1,
        yellow_time=5,
        min_green_time=10,
        max_green_time=60,
        end_time=600,  # Short episode for testing
        use_gui=False,
    )
    
    state, _ = env.reset()
    state_dim = len(state)
    action_dim = 2
    
    agent = ActionConstrainedPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_intersections=1,
        device="cpu",
    )
    
    agent.reset()
    
    episode_reward = 0.0
    constraint_violations = 0
    steps = 0
    
    done = False
    while not done:
        feasible_actions = env.get_feasible_actions()[0]
        action_value, log_prob, value = agent.select_action(state, feasible_actions)
        action = np.array([action_value])
        
        # Check constraint
        if action_value not in feasible_actions:
            constraint_violations += 1
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.store_experience(
            state=state,
            action=action_value,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=done,
        )
        
        episode_reward += reward
        steps += 1
        state = next_state
    
    print(f"✓ Episode completed with {steps} steps")
    print(f"  Episode Reward: {episode_reward:.2f}")
    print(f"  Constraint Violations: {constraint_violations}")
    print(f"  Violation Rate: {constraint_violations / steps:.4f}")
    
    # Update agent
    metrics = agent.update(batch_size=32, num_epochs=2)
    print(f"✓ Agent update completed")
    print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
    
    assert constraint_violations == 0, f"Constraint violations detected: {constraint_violations}"
    assert np.isfinite(episode_reward), f"Episode reward is not finite: {episode_reward}"
    
    env.close()
    print(f"✓ Full episode test passed!")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("ACTION-CONSTRAINED RL VALIDATION TESTS")
    print("=" * 50)
    
    tests = [
        test_environment,
        test_agent_network,
        test_action_masking,
        test_gae_and_update,
        test_model_save_load,
        test_full_episode,
    ]
    
    failed = []
    
    for test in tests:
        try:
            if test():
                print(f"✓ {test.__name__} PASSED")
            else:
                print(f"✗ {test.__name__} FAILED")
                failed.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} FAILED with exception:")
            print(f"  {type(e).__name__}: {str(e)}")
            failed.append(test.__name__)
    
    print("\n" + "=" * 50)
    if not failed:
        print("ALL TESTS PASSED ✓")
        return 0
    else:
        print(f"FAILED TESTS ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")
        return 1


if __name__ == "__main__":
    exit(main())
