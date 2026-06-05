import numpy as np


class MaxPressureAgent:
    """
    Dynamic max-pressure traffic signal controller.

    For each intersection, every available green action is scored using the
    TrafficSignalEnv phase-pressure helper. The proposed action is the phase
    with the highest pressure. On ties, the current green action is kept.

    The environment still performs the actual constraint repair, so the
    violation metric can show when this unconstrained baseline wanted an
    action that min-green/yellow/pedestrian constraints did not allow.
    """

    def __init__(self, num_intersections: int = 1):
        self.num_intersections = int(num_intersections)
        self._current_actions = [0] * self.num_intersections

    def select_action(self, traffic_signals: list) -> np.ndarray:
        actions = []
        for i, signal in enumerate(traffic_signals):
            current = self._current_actions[i] if i < len(self._current_actions) else 0
            action = self._select_for_signal(signal, current)
            if i < len(self._current_actions):
                self._current_actions[i] = action
            actions.append(action)
        return np.array(actions, dtype=np.int32)

    def _select_for_signal(self, signal, current_action: int) -> int:
        action_dim = len(signal.green_phases)
        current_action = int(np.clip(current_action, 0, max(action_dim - 1, 0)))

        if signal.is_transitioning:
            return signal.current_green_phase_idx

        best_action = current_action
        best_pressure = float("-inf")

        for action in range(action_dim):
            if hasattr(signal, "get_phase_pressure"):
                pressure = signal.get_phase_pressure(action)
            else:
                pressure = self._fallback_pressure(signal, action)

            if pressure > best_pressure:
                best_pressure = pressure
                best_action = action
            elif pressure == best_pressure and action == current_action:
                best_action = current_action

        return int(best_action)

    def _fallback_pressure(self, signal, action: int) -> float:
        """Queue-only fallback for older TrafficSignalEnv objects."""
        served = []
        if hasattr(signal, "get_served_lanes_for_action"):
            served = signal.get_served_lanes_for_action(action)

        if served:
            pressure = 0.0
            for lane in served:
                try:
                    pressure += float(signal.sumo_traci.lane.getLastStepHaltingNumber(lane))
                except Exception:
                    pass
            return pressure

        queues = signal.get_vehicle_queue()
        if not queues:
            return 0.0
        return float(sum(queues))

    def reset(self):
        self._current_actions = [0] * self.num_intersections
