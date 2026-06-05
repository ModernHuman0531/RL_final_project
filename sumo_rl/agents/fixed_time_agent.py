import numpy as np


class FixedTimeAgent:
    """
    Fixed-time traffic signal controller baseline.

    The controller is state-blind: every cycle_time decision steps it advances
    to the next available green action. The environment still enforces yellow,
    min-green, max-green, and pedestrian-safety constraints.

    Args:
        cycle_time: Number of decision steps each proposed green action is held.
        num_intersections: Number of intersections to control.
        action_dims: Optional fallback action count per intersection. If
            traffic signals are passed to select_action(), their dynamic
            green-phase counts are used instead.
    """

    def __init__(
        self,
        cycle_time: int = 30,
        num_intersections: int = 1,
        action_dims=None,
    ):
        self.cycle_time = int(cycle_time)
        self.num_intersections = int(num_intersections)
        self.action_dims = self._normalize_action_dims(action_dims)
        self._step_counters = [0] * self.num_intersections
        self._current_actions = [0] * self.num_intersections

    def _normalize_action_dims(self, action_dims):
        if action_dims is None:
            return [2] * self.num_intersections
        if isinstance(action_dims, int):
            return [int(action_dims)] * self.num_intersections
        dims = [int(dim) for dim in action_dims]
        if len(dims) < self.num_intersections:
            dims.extend([dims[-1]] * (self.num_intersections - len(dims)))
        return dims[: self.num_intersections]

    def _looks_like_signals(self, value) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) > 0
            and hasattr(value[0], "green_phases")
        )

    def _action_dim_for(self, idx: int, traffic_signals=None) -> int:
        if traffic_signals is not None and idx < len(traffic_signals):
            return max(len(traffic_signals[idx].green_phases), 1)
        return max(self.action_dims[idx], 1)

    def select_action(self, state=None, traffic_signals=None) -> np.ndarray:
        """
        Return one proposed action per intersection.

        state is accepted for compatibility but ignored. For dynamic SUMO
        networks, pass traffic_signals=list(env.traffic_signals.values()).
        """
        if traffic_signals is None and self._looks_like_signals(state):
            traffic_signals = state

        actions = []
        for i in range(self.num_intersections):
            action_dim = self._action_dim_for(i, traffic_signals)
            self._current_actions[i] %= action_dim

            self._step_counters[i] += 1
            if self._step_counters[i] >= self.cycle_time:
                self._step_counters[i] = 0
                self._current_actions[i] = (self._current_actions[i] + 1) % action_dim

            actions.append(self._current_actions[i])

        return np.array(actions, dtype=np.int32)

    def reset(self):
        """Reset timers and proposed phases at the start of an episode."""
        self._step_counters = [0] * self.num_intersections
        self._current_actions = [0] * self.num_intersections
