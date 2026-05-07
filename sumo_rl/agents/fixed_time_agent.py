import numpy as np


class FixedTimeAgent:
    """
    Fixed-time traffic signal controller (baseline).

    Switches between NS-green and EW-green on a fixed cycle regardless of
    traffic state. This is the standard real-world reference controller that
    learned methods must outperform.

    Each call to select_action() counts as one simulation step. The agent
    proposes a phase switch every `cycle_time` steps; the environment's
    constraint layer (yellow time, min/max green) still applies underneath.

    Args:
        cycle_time: Number of steps each green phase is held before switching.
                    Default 30 gives a 60-second full cycle at delta_time=1.
        num_intersections: Number of intersections to control (one action per).
    """

    def __init__(self, cycle_time: int = 30, num_intersections: int = 1):
        self.cycle_time = cycle_time
        self.num_intersections = num_intersections
        self._step_counters = [0] * num_intersections
        self._current_phases = [0] * num_intersections  # 0=NS-green, 1=EW-green

    def select_action(self, state=None) -> np.ndarray:
        """
        Return one action per intersection based purely on the internal timer.
        `state` is accepted but ignored — this agent is state-blind.
        """
        actions = []
        for i in range(self.num_intersections):
            self._step_counters[i] += 1
            if self._step_counters[i] >= self.cycle_time:
                self._step_counters[i] = 0
                self._current_phases[i] = 1 - self._current_phases[i]  # flip phase
            actions.append(self._current_phases[i])
        return np.array(actions, dtype=np.int32)

    def reset(self):
        """Reset timers and phases — call at the start of each episode."""
        self._step_counters = [0] * self.num_intersections
        self._current_phases = [0] * self.num_intersections
