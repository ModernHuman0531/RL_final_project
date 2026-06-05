import numpy as np


class SOTLAgent:
    """
    Self-Organizing Traffic Lights baseline.

    The controller keeps the current green action until queued vehicles on
    currently unserved lanes reach kappa and the current green has satisfied
    min_green_time. When switching, it selects the non-current green action
    serving the largest queued demand.
    """

    def __init__(self, kappa: int = 5, num_intersections: int = 1):
        self.kappa = int(kappa)
        self.num_intersections = int(num_intersections)

    def select_action(self, traffic_signals: list) -> np.ndarray:
        return np.array(
            [self._select_for_signal(signal) for signal in traffic_signals],
            dtype=np.int32,
        )

    def _select_for_signal(self, signal) -> int:
        current_action = signal.current_green_phase_idx

        if signal.is_transitioning:
            return current_action

        if signal.phase_timer < signal.min_green_time:
            return current_action

        red_pressure = self._red_pressure(signal, current_action)
        if red_pressure < self.kappa:
            return current_action

        return self._best_non_current_action(signal, current_action)

    def _red_pressure(self, signal, current_action: int) -> float:
        served = set()
        if hasattr(signal, "get_served_lanes_for_action"):
            served = set(signal.get_served_lanes_for_action(current_action))

        red_pressure = 0.0
        queues = signal.get_vehicle_queue()
        for lane, queue in zip(signal.lanes, queues):
            if lane not in served:
                red_pressure += queue
        return float(red_pressure)

    def _best_non_current_action(self, signal, current_action: int) -> int:
        best_action = current_action
        best_score = float("-inf")

        for action in range(len(signal.green_phases)):
            if action == current_action:
                continue
            if hasattr(signal, "get_phase_pressure"):
                score = signal.get_phase_pressure(action)
            else:
                score = self._served_queue(signal, action)

            if score > best_score:
                best_score = score
                best_action = action

        return int(best_action)

    def _served_queue(self, signal, action: int) -> float:
        if not hasattr(signal, "get_served_lanes_for_action"):
            return float(sum(signal.get_vehicle_queue()))

        served = set(signal.get_served_lanes_for_action(action))
        queues = signal.get_vehicle_queue()
        return float(sum(q for lane, q in zip(signal.lanes, queues) if lane in served))

    def reset(self):
        pass
