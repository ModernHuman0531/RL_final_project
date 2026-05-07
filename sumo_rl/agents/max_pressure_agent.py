import numpy as np


class MaxPressureAgent:
    """
    Max Pressure traffic signal controller (Varaiya, 2013).

    At each decision step, selects the green phase whose total incoming queue
    pressure is highest:
        pressure_NS = queue(n_in) + queue(s_in)
        pressure_EW = queue(e_in) + queue(w_in)

    On ties, keeps the current phase to avoid unnecessary switches.
    The agent never consults or modifies the environment's constraint layer —
    yellow-time and min/max-green enforcement still happen inside env.step().

    Args:
        num_intersections: Number of intersections to control.
    """

    def __init__(self, num_intersections: int = 1):
        self.num_intersections = num_intersections
        self._current_phases = [0] * num_intersections  # 0=NS-green, 1=EW-green

    def select_action(self, traffic_signals: list) -> np.ndarray:
        """
        Args:
            traffic_signals: List of TrafficSignalEnv objects, one per intersection.
                             Obtained from list(env.traffic_signals.values()).
        Returns:
            np.ndarray of shape (num_intersections,) with 0 (NS) or 1 (EW).
        """
        actions = []
        for i, signal in enumerate(traffic_signals):
            action = self._select_for_signal(signal, self._current_phases[i])
            self._current_phases[i] = action
            actions.append(action)
        return np.array(actions, dtype=np.int32)

    def _select_for_signal(self, signal, current_phase: int) -> int:
        queues = signal.get_vehicle_queue()   # one entry per lane in signal.lanes

        ns_pressure = 0.0
        ew_pressure = 0.0

        for lane, q in zip(signal.lanes, queues):
            # Lane names: "n_in_1", "e_in_1", ... Internal crossing lanes start
            # with ":" and won't match edge_to_direction — they are skipped.
            edge = lane.rsplit("_", 1)[0]          # "n_in_1" → "n_in"
            direction = signal.edge_to_direction.get(edge)
            if direction in ("N", "S"):
                ns_pressure += q
            elif direction in ("E", "W"):
                ew_pressure += q

        if ns_pressure > ew_pressure:
            return 0   # NS green
        elif ew_pressure > ns_pressure:
            return 1   # EW green
        else:
            return current_phase   # tie → keep current phase

    def reset(self):
        """Reset tracked phases — call at the start of each episode."""
        self._current_phases = [0] * self.num_intersections
