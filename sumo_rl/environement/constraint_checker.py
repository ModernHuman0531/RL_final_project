"""

ConstraintChecker - A class that regulate all the traffic light switching constraints.
Output valid_actions mask for TrafficSignalEnv.

--------------------------------------------------------------
The original code i directly write get_valid_actions() function in TrafficSignalEnv, 
it's okay when we only have traffic light time swithcing constraint, but when we add
pedestrian crossing constraint, we don't want two different logics mixed together, so
we put all the constraint checking logic in this class, and TrafficSignalEnv just call this class to get the valid_actions mask.

Traffic light time switching constraint:
    - When in Yellow phase, the traffic light cannot switch.
    - Didn't reached the minimum green time, the traffic light cannot switch.
    - Exceed the maximum green time, the traffic light must switch.

Pedestrian crossing constraint:
    - When pedestrian crossing is active, the traffic light cannot switch to a phase that will endanger pedestrians.

In the original design of SUMO, already ensure to let the pedestrian crossing phase and the conflicting traffic light phase never appear at the same time, 
so we only need to check if the pedestrian crossing is active, if it's active, then we cannot switch to the conflicting traffic light phase.
Because the pedestrian is still crossing, but the traffic light already swithced and
the vehicles start to move, which will endanger the pedestrians.

Solution: Use traci.person.getRoadID() to check if there is any pedestrian on the road, 
And then compare the green phase and the pedestrian crossing phase, if they are conflicting, 
then this green phase is not a valid action, we cannot switch to this green phase until the pedestrian crossing is no longer active.
"""
import os
import sys
from typing import TYPE_CHECKING, Dict, List, Set

if "SUMO_HOME" in os.environ:
    tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools_path)

if TYPE_CHECKING:
    from .traffic_signal import TrafficSignalEnv

class ConstraintChecker:
    """
    Regulate one intersection's traffic light switching constraints, and output valid_actions mask.

    Usage (In TrafficSignalEnv):
        self.constraint_checker = ConstraintChecker(self, enable_pedestrian_safety=True)
        ...
        valid_actions = self.constraint_checker.get_valid_actions()
        Args:
            signal: TrafficSignalEnv object.
            enable_pedestrian_safety: Whether to enable pedestrian crossing constraint.
    """
    def __init__(self, signal: "TrafficSignalEnv", enable_pedestrian_safety: bool = True):
        self.signal = signal
        self.enable_pedestrian_safety = enable_pedestrian_safety

        # Build the mapping of green phase and the conflicting pedestrian crossing phase.
        self.conflict_map: Dict[int, Set[str]] = {}
        self._build_conflict_map()
    
    def _build_conflict_map(self):
        """
        Analyze all green phase's state string, find out every green phase's vehicle 
        moving direction corresponding to the pedestrian crossing phase.

        How to implement:
            1. Get TLS controlled links, seperating vehicle's link and pedestrian's link.
            2. For each green phase, check which vehicle link is green('G')
            3. Find which direction is the vehicle comes from.
            4. Confilcting pedestrian direction = prependicular direction of the vehicle direction.
            (NS green conflict with EW pedestrian, EW green conflict with NS pedestrian)
        """
        traci = self.signal.sumo_traci
        tls_id = self.signal.intersection_id

        # Get all controlled links
        # links: list[list[(from_lane, to_lane, via_link)]]
        # Each list's index corresponds to phase state string's index.
        links = traci.trafficlight.getControlledLinks(tls_id)

        # Categorize vehicle link and pedestrian link
        # from_lane is pedestrian crossing if lane's type is 'c'

        crossing_link_indicies: Set[int] = set()
        for i, link_group in enumerate(links):
            # Crossing link's from_lane start from ":", use this feature to identify.
            for (from_lane, to_lane, via_link) in link_group:
                from_edge = from_lane.rsplit("_", 1)[0]
                if from_edge.startswith(":"):
                    crossing_link_indicies.add(i)
                    break
        
        # Get phase state string
        logics = traci.trafficlight.getAllProgramLogics(tls_id)
        phases = logics[0].phases

        for green_idx in self.signal.green_phases:
            state = phases[green_idx].state
            vehicle_directions: Set[str] = set()

            # Find in this green phase, which vehicle link is green, and find the vehicle direction.
            for i, char in enumerate(state):
                # Skip pedestrian link
                if i in crossing_link_indicies:
                    continue
                if char == "G":
                    # This link is green, find the vehicle direction
                    if i < len(links) and links[i]:
                        from_lane = links[i][0][0]
                        edge = from_lane.rsplit("_", 1)[0]
                        direction = self.signal.edge_to_direction.get(edge)
                        if direction:
                            vehicle_directions.add(direction)
            
            # Confilcting pedestrian direction = prependicular direction of the vehicle direction.
            # NS green conflict with EW pedestrian, EW green conflict with NS pedestrian
            conflict_ped_directions: Set[str] = set()
            if vehicle_directions & {"N", "S"}:
                conflict_ped_directions |= {"E", "W"}
            if vehicle_directions & {"E", "W"}:
                conflict_ped_directions |= {"N", "S"}
            
            self.conflict_map[green_idx] = conflict_ped_directions

    def _check_traffic_light_constraint(self) -> List[int]:
        """
        Traffic light switching constraint:
            - When in Yellow phase, the traffic light cannot switch.
            - Didn't reached the minimum green time, the traffic light cannot switch.
            - Exceed the maximum green time, the traffic light must switch.
        """
        n = len(self.signal.green_phases)
        valid_actions = [1] * n
        current_idx = self.signal.current_green_phase_idx

        # 1. When in Yellow phase, the traffic light cannot switch.
        if self.signal.is_transitioning:
            valid_actions = [0] * n
            valid_actions[current_idx] = 1
            return valid_actions
        # 2. Didn't reached the minimum green time, the traffic light cannot switch.
        if self.signal.phase_timer < self.signal.min_green_time:
            valid_actions = [0] * n
            valid_actions[current_idx] = 1
            return valid_actions
        # 3. Exceed the maximum green time, the traffic light must switch.
        if self.signal.phase_timer >= self.signal.max_green_time:
            valid_actions = [1] * n
            valid_actions[current_idx] = 0
            return valid_actions
        
        return valid_actions

    def _check_pedestrian_safety(self) -> List[int]:
        """
        If the pedestrian is crossing, then block the confilicting green phase.

        How to check if the pedestrian is crossing:
            Use traci.person.getRoadID() to check the pedestrian's current road.
            SUMO's internal edge use ":" starting.
            The format usually like: ':C2_w1_0`, we can use this feature to identify 
            if there is any pedestrian on the walkingarea.
        """
        n = len(self.signal.green_phases)
        valid = [1] * n

        if not self.signal.has_pedestrian_crossing():
            return valid

        # If someone is already in the crossing area, do not allow a phase
        # change. This is deliberately conservative for the project safety
        # layer; the current green phase stays valid.
        current = self.signal.current_green_phase_idx
        valid = [0] * n
        valid[current] = 1
        return valid
    
    def get_valid_actions(self) -> List[int]:
        """
        Get the valid action mask for the current state, 1 means valid, 0 means invalid.
        Use both traffic light switching constraint and pedestrian crossing constraint to filter the valid actions.
        """
        valid_actions = self._check_traffic_light_constraint()
        # Use pedestrian safety constraint to further filter the valid actions, only when it's enabled.
        if self.enable_pedestrian_safety:
            ped_safe_actions = self._check_pedestrian_safety()
            valid_actions = [v & p for v, p in zip(valid_actions, ped_safe_actions)]

        if not any(valid_actions):
            valid_actions[self.signal.current_green_phase_idx] = 1
        
        return valid_actions
    # -- Debug function
    def get_constraint_info(self):
        """
        Return the current constraint status for debugging.
        """
        s = self.signal
        info = {
            "intersection_id": s.intersection_id,
            "pedestrian_safety_enabled": self.enable_pedestrian_safety,
            "is_transitioning": s.is_transitioning,
            "phase_timer": s.phase_timer,
            "current_green_phase_idx": s.current_green_phase_idx,
            "current_phase": s.current_phase,
            "min_green_reached": s.phase_timer >= s.min_green_time,
            "max_green_exceeded": s.phase_timer >= s.max_green_time,
            "valid_actions": self.get_valid_actions(),
            "conflict_map": {k: list(v) for k, v in self.conflict_map.items()}
        }
        return info
