import numpy as np


class SOTLAgent:
    """
    Self-Organizing Traffic Lights — Gershenson (2004).

    Switches the green phase when the number of vehicles queued on the
    currently RED lanes reaches or exceeds a threshold kappa, provided the
    current phase has been active long enough (read directly from the signal's
    internal phase_timer so it is always in sync with the environment).

    The rule is purely reactive — no learning, no model:
        switch  if  red_pressure >= kappa  AND  phase_timer >= min_green_time
        keep    otherwise

    The environment's constraint layer (yellow lock, min/max green) still
    applies on top, so the agent never produces physically illegal transitions.

    Args:
        kappa: Vehicle count threshold on red lanes that triggers a switch.
               Gershenson (2004) uses values in the range 4–10.
        num_intersections: Number of intersections to control.
    """

    def __init__(self, kappa: int = 5, num_intersections: int = 1):
        self.kappa = kappa
        self.num_intersections = num_intersections

    def select_action(self, traffic_signals: list) -> np.ndarray:
        """
        Args:
            traffic_signals: List of TrafficSignalEnv objects, one per
                             intersection. Obtained from
                             list(env.traffic_signals.values()).
        Returns:
            np.ndarray of shape (num_intersections,) with 0 (NS) or 1 (EW).
        """
        return np.array(
            [self._select_for_signal(s) for s in traffic_signals],
            dtype=np.int32,
        )

    def _select_for_signal(self, signal) -> int:
        # During yellow transition keep the target green phase — switching now
        # would conflict with the ongoing transition.
        if signal.is_transitioning:
            return signal.green_phases.index(signal.current_green_phase)

        current_action = signal.green_phases.index(signal.current_green_phase)

        # Sum halting vehicles on the lanes that are currently RED.
        red_pressure = 0
        queues = signal.get_vehicle_queue()
        for lane, q in zip(signal.lanes, queues):
            edge = lane.rsplit("_", 1)[0]          # "n_in_1" → "n_in"
            direction = signal.edge_to_direction.get(edge)
            if direction is None:
                continue                            # skip crossing / internal lanes
            if current_action == 0 and direction in ("E", "W"):   # NS green → EW red
                red_pressure += q
            elif current_action == 1 and direction in ("N", "S"): # EW green → NS red
                red_pressure += q

        # Core SOTL rule: switch when red-side pressure is high enough
        # and the current green phase has been held for at least min_green_time.
        if red_pressure >= self.kappa and signal.phase_timer >= signal.min_green_time:
            return 1 - current_action   # switch to other phase
        return current_action           # keep current phase

    def reset(self):
        """Stateless agent — nothing to reset between episodes."""
        pass
