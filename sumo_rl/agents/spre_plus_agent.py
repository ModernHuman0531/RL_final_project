"""
SPre+ — Action-Constrained RL via QP Projection (Hung et al., 2025).

HOW IT WORKS
------------
When a policy proposes a probability distribution p over actions, SPre+
checks whether the proposed action is in the feasible set C(s). If not,
it solves the following Quadratic Programme to find the nearest valid
distribution q:

    min   ||q - p||²
    s.t.  q_a = 0   for all a ∉ C(s)
          Σ q_a = 1
          q_a ≥ 0

The action with the highest probability in q is then executed. This
guarantees safety by construction but is computationally expensive: the QP
must be solved at every step, and cost grows with the action-space size.

WHY THIS IS A BASELINE (not the proposed method)
-------------------------------------------------
This is the existing SOTA safe-RL method that the project's AR approach
improves upon. The AR method avoids QP entirely by resampling from the policy
until a valid action is drawn — O(1) per step vs O(|A|²) for QP.

INTEGRATION NOTE
----------------
This class requires a trained policy to produce meaningful results.
Without one it falls back to random (uniform) preferences, which makes it
equivalent to a random-safe agent. Once DQN is implemented, plug in the
trained Q-network via set_policy():

    agent = SPRePlusAgent()
    agent.set_policy(lambda state: dqn.q_net(torch.FloatTensor(state)).numpy())
"""

import numpy as np

try:
    from scipy.optimize import minimize
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


class SPRePlusAgent:
    """
    SPre+ baseline: any policy + QP projection onto C(s).

    Args:
        action_dim:        Number of discrete actions per intersection.
        num_intersections: Number of intersections to control.
        use_scipy:         If True, call scipy SLSQP (shows real QP overhead).
                           If False, use the closed-form solution (equivalent
                           result for small action spaces, much faster).
    """

    def __init__(
        self,
        action_dim: int = 2,
        num_intersections: int = 1,
        use_scipy: bool = True,
        state_dim: int = None,
    ):
        self.action_dim = action_dim
        self.num_intersections = num_intersections
        self.use_scipy = use_scipy and _SCIPY_AVAILABLE
        self.state_dim = state_dim  # if set, slices exactly this many features per signal
        self._policy_fn = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_policy(self, policy_fn):
        """
        Plug in a trained policy.

        Args:
            policy_fn: Callable(state: np.ndarray) -> np.ndarray of Q-values
                       with shape (action_dim,). Typically a wrapper around
                       the trained DQN Q-network.
        """
        self._policy_fn = policy_fn

    def select_action(self, state: np.ndarray, traffic_signals: list) -> np.ndarray:
        """
        Args:
            state:           Current flattened environment state.
            traffic_signals: List of TrafficSignalEnv objects.
        Returns:
            np.ndarray of shape (num_intersections,) with safe actions.
        """
        actions = []
        state_per_signal = self.state_dim if self.state_dim is not None \
                           else len(state) // max(len(traffic_signals), 1)

        for i, signal in enumerate(traffic_signals):
            signal_state = state[i * state_per_signal: (i + 1) * state_per_signal]
            valid_actions = signal.get_valid_actions()
            preferences = self._get_preferences(signal_state)
            actions.append(self._qp_project(preferences, valid_actions))

        return np.array(actions, dtype=np.int32)

    def reset(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_preferences(self, state: np.ndarray) -> np.ndarray:
        """Softmax over Q-values from the policy (or uniform if no policy)."""
        if self._policy_fn is not None:
            q = self._policy_fn(state)
        else:
            # Placeholder: uniform random Q-values.
            # Replace with trained DQN via set_policy() for real results.
            q = np.random.uniform(0.0, 1.0, self.action_dim)

        q_shifted = q - q.max()
        exp_q = np.exp(q_shifted)
        return exp_q / exp_q.sum()

    def _qp_project(self, preferences: np.ndarray, valid_actions: list) -> int:
        """Project preferences onto the feasible set via QP."""
        valid = np.array(valid_actions, dtype=float)

        if valid.sum() == 0:
            return 0                        # fallback (should not occur)
        if valid.sum() == 1:
            return int(np.argmax(valid))    # trivial: only one valid action

        if self.use_scipy:
            return self._qp_scipy(preferences, valid)
        return self._qp_analytical(preferences, valid)

    def _qp_scipy(self, preferences: np.ndarray, valid: np.ndarray) -> int:
        """
        General QP via scipy SLSQP.
        Included to demonstrate the per-step computational overhead of SPre+
        compared to the O(1) acceptance-rejection method.
        """
        x0 = valid / valid.sum()
        bounds = [(0.0, 1.0) if v == 1 else (0.0, 0.0) for v in valid]
        constraints = {"type": "eq", "fun": lambda q: np.sum(q) - 1.0}

        result = minimize(
            fun=lambda q: 0.5 * float(np.sum((q - preferences) ** 2)),
            x0=x0,
            method="SLSQP",
            jac=lambda q: (q - preferences).astype(float),
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 100},
        )
        return int(np.argmax(result.x))

    def _qp_analytical(self, preferences: np.ndarray, valid: np.ndarray) -> int:
        """
        Closed-form QP solution (exact for convex problems with box constraints):
        zero out invalid actions and renormalise.
        """
        projected = preferences * valid
        if projected.sum() > 0:
            projected = projected / projected.sum()
        else:
            projected = valid / valid.sum()
        return int(np.argmax(projected))
