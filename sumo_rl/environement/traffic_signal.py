"""
This file contains the implementation of the traffic signal environment for SUMO-RL.
"""
import os
import sys

# Import traci in a script
if 'SUMO_HOME' in os.environ:
    # Add the SUMO tools directory to the python path
    tools_path = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools_path)
else:
    raise EnvironmentError("Please declare the environment variable 'SUMO_HOME' in Dockerfile or in your system environment variables")

import numpy as np
from gymnasium import spaces
from .constraint_checker import ConstraintChecker
# Tip: The reason we don't import traci directly is to use that as the input parameter for the environment, which allows us to control multiple intersections in the same simulation.

class TrafficSignalEnv:
    """
    This class defines a Traffic signal controlling an intersection.
    It is responsible for retrieving the informations of the intersection and changin traffic light phase using traci API.

    # State space:
    The default state space for each traffic signal is a vector:
    state = [phase_one_hot, min_green, queue_N, queue_E, queue_S, queue_W]
    where:
    - phase_one_hot: it is a one-hot encoded vector of the current active green phase, with length equal to the number of possible phases for the intersection.
    - min_green: Is a binary variable indicating whether minimum green time has been reached in the current phase (1 if minimum green time has been reached, 0 otherwise).
    - queue_N, queue_E, queue_S, queue_W: are the number of vehicles in the queue for each direction (North, East, South, West) respectively.
    - ped_N, ped_E, ped_S, ped_W: are the number of waiting pedestrians for each direction (North, East, South, West) respectively.

    # Action space:
    Action space is discrete, corresponding to which green phase is going to be activated.
    """
    def __init__(
        self, 
        sumo_traci,
        env,
        intersection_id,
        yellow_time,
        min_green_time,
        max_green_time,
        begin_time,
        end_time,
        enable_pedestrian_safety_constraint=True,
        current_phase = 0
    ):
        """Initialize the traffic signal environment.
        Initialize the traffic signal object with the given parameters.
        Args:
            sumo_traci: The traci instance for SUMO simulation to ensure use the same instance for multiple intersections.
            env: The main environment this traffic signal belongs to. 
            intersection_id: One intersection control four traffic light phases, and the id of the intersection is the same as the id of the traffic light in SUMO.
            yellow_time: The duration of the yellow phase in seconds.
            min_green_time: The minimum duration of the green phase in seconds.
            max_green_time: The maximum duration of the green phase in seconds.
            begin_time: The time in seconds whem the traffic signal starts to be controlled ny the agent.
            end_time: The time in seconds when the traffic signal stops to be controlled by the agent.
        """
        self.sumo_traci = sumo_traci
        self.env = env
        self.intersection_id = intersection_id
        self.yellow_time = yellow_time
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.begin_time = begin_time
        self.end_time = end_time
        self.enable_pedestrian_safety_constraint = enable_pedestrian_safety_constraint

        # Step 1: Get controlled lanes (Except internal edge)
        self.lanes = list(
            dict.fromkeys(
                lane for lane in self.sumo_traci.trafficlight.getControlledLanes(self.intersection_id)
                if not self._lane_to_edge(lane).startswith(":")
            )
        )

        # Step 2: Dynamically read green phases and get transition phases from the net file, and set the phase to action mapping.
        self._init_phases()

        # Step 3: edge -> direction mapping
        self._init_edge_to_direction()

        # Step 4: Execution state
        initial_phase = current_phase if current_phase in self.green_phases else self.green_phases[0]
        self.current_phase = initial_phase
        self.current_green_phase_idx = self.green_phases.index(initial_phase)
        self.transition_queue = []
        self.is_transitioning = False
        self.phase_timer = 0
        self.sumo_traci.trafficlight.setPhase(self.intersection_id, self.current_phase)

        # Step 5: Build ConstrainChecker
        self.constraint_checker = ConstraintChecker(self, enable_pedestrian_safety=self.enable_pedestrian_safety_constraint)

    def _lane_to_edge(self, lane_id):
        """
        Convert a lane id to edge id by removing the lane index at the end of the lane id.
        For example, C2B2_0 -> C2B2
        """
        return lane_id.rsplit("_", 1)[0]
    
    def _init_phases(self):
        """
        From traci to read TLS logic, dynamically find green phases and the transition phases between them, and set the mapping from phase to action.

        How to know the main Green light:
            - phase.state includes 'G'.
            - phase.state don't have 'y' .
        
        How to build the transition phases:
            netgenerate promise that phase is circularly ordered like [G, y, G, y]
            So from green phase A to green phase B, as long as we find the index of A and B in the phase list, we can get the transition phases by getting the phases in between A and B in the circular order.
        """
        logics = self.sumo_traci.trafficlight.getAllProgramLogics(self.intersection_id)
        phases = logics[0].phases
        self.phase_states = [p.state for p in phases]
        self.controlled_links = self.sumo_traci.trafficlight.getControlledLinks(self.intersection_id)

        self.green_phases = [
            i for i, p in enumerate(phases)
            if "G" in p.state and "y" not in p.state
        ]
        if len(self.green_phases) < 2:
            raise RuntimeError(
                f"Intersection {self.intersection_id} has less than 2 green phases."
            )
        n = len(phases)
        self.num_phases = n
        self.transition = {}
        for src in self.green_phases:
            for dst in self.green_phases:
                if src == dst:
                    continue
                # Start from src's next phase, and keep adding phases until reach dst
                path, idx = [], (src + 1) % n
                while idx != dst:
                    path.append(idx)
                    idx = (idx + 1) % n
                path.append(dst)
                self.transition[(src, dst)] = path
    
    def _init_edge_to_direction(self):
        """
        Dynamically build the mapping from edge to direction (N, E, S, W) for pedestrian safety constraint.

        We can't directly use name to find the direction, we use coordinate to find the direction, since in some cases the edge name may not contain the direction information, but the coordinate is always correct.
            - Get junction's (x, y)
            - Get each incoming edge's (x, y)
            - Calculate start -> junction vector, use this to determine the direction of the edge.
        """
        jx, jy = self.sumo_traci.junction.getPosition(self.intersection_id)

        # Get all incoming edges (Get from controlled links, except interal edge)
        links = getattr(
            self,
            "controlled_links",
            self.sumo_traci.trafficlight.getControlledLinks(self.intersection_id),
        )
        incoming_edges = set()
        for link_group in links:
            for from_lane, to_lane, via_link in link_group:
                edge = self._lane_to_edge(from_lane)
                if not edge.startswith(":"):
                    incoming_edges.add(edge)
        
        self.edge_to_direction = {}
        self.incoming_edges = incoming_edges

        for edge_id in incoming_edges:
            lane_id = edge_id + "_0" # Get the lane id by adding _0 to the edge id, since we assume each edge has at least one lane, and we only need one lane to determine the direction of the edge.
            try:
                shape = self.sumo_traci.lane.getShape(lane_id)
            except Exception:
                shape = self.sumo_traci.edge.getShape(edge_id)

            fx, fy = shape[0] # edge's starting point
            dx, dy = jx - fx, jy - fy
            if abs(dx) >= abs(dy):
                self.edge_to_direction[edge_id] = "W" if dx > 0 else "E"
            else:
                self.edge_to_direction[edge_id] = "S" if dy > 0 else "N"

    @property
    def current_green_phase(self) -> int:
        """The SUMO phase index for the green phase currently targeted."""
        return self.green_phases[self.current_green_phase_idx]

    @property
    def state_size(self) -> int:
        """
        This intersection's observation vector length.
        Use in sumo_env.py to dynamically adjust the observation_space.

        - len(green_phases) -> phase one-hot
        - 4 -> current phase, phase timer, min-green flag, transition flag
        - 2 * len(lanes) -> vehicle queue and density for each lane
        - 1 -> total vehicle waiting time
        - 4 -> pedestrian queue length for each direction (N, E, S, W)
        - 1 -> total pedestrian waiting time
        - 1 -> pedestrian crossing occupancy
        """
        return len(self.green_phases) + 11 + (2 * len(self.lanes))
    
    def set_phase(self, action):
        """
        Set the traffic light phase for the intersection.
        action: 0,...,len(green_phases)-1, choose to switch to which green phase.

        Originally use self.current_green_phase to store the SUMO phase number (like 0,3)

        Now use self.current_green_phase_idx to store the index of the green phase in self.green_phases.
        """
        action = int(action)
        if action < 0 or action >= len(self.green_phases):
            raise ValueError(
                f"Action {action} is outside the green phase action space "
                f"for intersection {self.intersection_id}."
            )

        target_phase = self.green_phases[action]
        source_phase = self.current_green_phase

        if self.is_transitioning:
            return

        # If target phase is current green phase and not in transitoin, just keep the current phase
        if target_phase == source_phase:
            return
        
        path = self.transition[(source_phase, target_phase)]
        self.is_transitioning = True
        self.current_green_phase_idx = action
        self.transition_queue = list(path)
        self.phase_timer = 0
        
    def update(self):
        """
        Update the transition state of the traffic signal, which is called in each step of the main environment.
        If the traffic signal is in transition, it will update the phase according to the transition queue and timer.
        """
        if self.is_transitioning:
            # The yellow_time is fixed, so if phase_timer is equal or more than yellow_time, it means we will get into next yellow phase or the 
            # target green phase in the next step, so we reset the timer and pop the next phase in the transition queue.
            if self.phase_timer >= self.yellow_time:
                self.phase_timer = 0
            # Get into new phase
            if self.phase_timer == 0:
                next_phase = self.transition_queue.pop(0)
                self.current_phase = next_phase
                self.sumo_traci.trafficlight.setPhase(self.intersection_id, self.current_phase)
                if len(self.transition_queue) == 0:
                    self.is_transitioning = False
        self.phase_timer += 1


    def get_vehicle_queue(self) -> list:
        """
        Get the queue length for each direction (North, East, South, West) and return it as a list.
        Return:
            A list of queue lengths for each direction in the order of [N_to_S, E_to_W, S_to_N, W_to_E].
        """
        queue_lengths = []
        for lane in self.lanes:
            queue_lengths.append(self.sumo_traci.lane.getLastStepHaltingNumber(lane))
        return queue_lengths

    def _capacity_estimates(self):
        """Approximate lane and pedestrian storage capacities for normalization."""
        if not self.lanes:
            return 1.0, 1.0
        length = max(float(self.sumo_traci.lane.getLength(self.lanes[0])), 1.0)
        max_vehicle = max(length / 5.0, 1.0)
        max_pedestrian = max(length / 0.215, 1.0)
        return max_vehicle, max_pedestrian

    def get_vehicle_density(self) -> list:
        """
        Get normalized vehicle occupancy for each controlled incoming lane.
        """
        max_vehicle, _ = self._capacity_estimates()
        densities = []
        for lane in self.lanes:
            try:
                densities.append(
                    min(self.sumo_traci.lane.getLastStepVehicleNumber(lane) / max_vehicle, 1.0)
                )
            except Exception:
                densities.append(0.0)
        return densities

    def get_vehicle_waiting_time(self):
        """
        Get the total waiting time of vehicles in the lanes controlled by this traffic signal and return it as a single value.
        """
        waiting_time = 0
        for lane in self.lanes:
            waiting_time += self.sumo_traci.lane.getWaitingTime(lane)
        return waiting_time

    def get_served_lanes_for_action(self, action: int, include_permissive: bool = True) -> list:
        """
        Return incoming lanes that would receive green for an action.

        Action indices are the RL action indices, not raw SUMO phase indices.
        The mapping is dynamic through self.green_phases, so this works for
        both the 1x1 and 3x3 networks as long as SUMO exposes the phase logic.
        """
        action = int(action)
        if action < 0 or action >= len(self.green_phases):
            return []

        phase_idx = self.green_phases[action]
        phase_state = self.phase_states[phase_idx]
        green_chars = {"G", "g"} if include_permissive else {"G"}
        served = []
        seen = set()

        for link_idx, signal_state in enumerate(phase_state):
            if signal_state not in green_chars:
                continue
            if link_idx >= len(self.controlled_links):
                continue
            for from_lane, _to_lane, _via_lane in self.controlled_links[link_idx]:
                edge = self._lane_to_edge(from_lane)
                if edge.startswith(":") or from_lane in seen:
                    continue
                if from_lane in self.lanes:
                    served.append(from_lane)
                    seen.add(from_lane)

        return served

    def get_phase_pressure(self, action: int) -> float:
        """
        Estimate max-pressure score for a candidate green action.

        For each green movement, pressure is incoming halted vehicles minus
        outgoing halted vehicles. If the outgoing lane cannot be queried, the
        method falls back to incoming queue only.
        """
        action = int(action)
        if action < 0 or action >= len(self.green_phases):
            return float("-inf")

        phase_idx = self.green_phases[action]
        phase_state = self.phase_states[phase_idx]
        pressure = 0.0
        seen_movements = set()

        for link_idx, signal_state in enumerate(phase_state):
            if signal_state not in {"G", "g"}:
                continue
            if link_idx >= len(self.controlled_links):
                continue
            for from_lane, to_lane, _via_lane in self.controlled_links[link_idx]:
                movement = (from_lane, to_lane)
                if movement in seen_movements:
                    continue
                seen_movements.add(movement)

                from_edge = self._lane_to_edge(from_lane)
                if from_edge.startswith(":"):
                    continue

                try:
                    incoming = float(self.sumo_traci.lane.getLastStepHaltingNumber(from_lane))
                except Exception:
                    incoming = 0.0

                try:
                    outgoing = float(self.sumo_traci.lane.getLastStepHaltingNumber(to_lane))
                except Exception:
                    outgoing = 0.0

                pressure += incoming - outgoing

        return pressure

    def _get_person_ids(self):
        """Return known pedestrian/person ids across SUMO versions."""
        try:
            return list(self.sumo_traci.person.getIDList())
        except Exception:
            try:
                return list(self.sumo_traci.simulation.getPersonIDList())
            except Exception:
                return []

    def has_pedestrian_crossing(self) -> bool:
        """
        Whether a pedestrian is currently moving on an internal crossing area
        associated with this intersection.
        """
        for pid in self._get_person_ids():
            try:
                road_id = self.sumo_traci.person.getRoadID(pid)
                speed = self.sumo_traci.person.getSpeed(pid)
            except Exception:
                continue
            if road_id.startswith(":") and self.intersection_id in road_id and speed > 0.1:
                return True
        return False

    def get_pedestrian_crossing_occupancy(self) -> float:
        """Binary crossing occupancy feature used by the PPO state."""
        return 1.0 if self.has_pedestrian_crossing() else 0.0

    def get_pedestrian_queue(self):
        """
        Get the number of waiting pedestrians in the lanes controlled by this traffic signal and return it as a single value.
        """
        ped_queue = {
            "N": 0,
            "E": 0,
            "S": 0,
            "W": 0
        }
        for edge_id in self.incoming_edges:
            direction = self.edge_to_direction[edge_id]
            if direction is None:
                continue
            try:
                ped_ids = self.sumo_traci.edge.getLastStepPersonIDs(edge_id)
            except Exception:
                continue

            for pid in ped_ids:
                # If the pedestrian is waiting, then we count it in the queue, otherwise we don't count it, since it's not really in the queue.
                if self.sumo_traci.person.getWaitingTime(pid) > 0:
                    ped_queue[direction] += 1
        
        return [ped_queue["N"], ped_queue["E"], ped_queue["S"], ped_queue["W"]]


    
    def get_pedestrian_waiting_time(self):
        """
        Get the total waiting time of pedestrians in the lanes controlled by this traffic signal and return it as a single value.
        """
        total_waiting_time = 0

        for edge_id in self.incoming_edges:
            try:
                ped_ids = self.sumo_traci.edge.getLastStepPersonIDs(edge_id)
            except Exception:
                continue
            for pid in ped_ids:                
                # If the edge is in the incoming edges an the pedestrian is waiting, then we count its waiting time.
                if self.sumo_traci.person.getWaitingTime(pid) > 0:
                    total_waiting_time += self.sumo_traci.person.getWaitingTime(pid)
        
        return total_waiting_time

    def get_state_feature(self):
        """
        Return the raw state feature of the traffic signal, which is a dictionary containing one-hot encoding of the current phase,
        whether minimum green time has been reached, queue length for each direction, and pedestrian queue for each direction.
        """
        phase_one_hot = np.zeros(len(self.green_phases), dtype=np.float32)

        # For simplicity, we only consider the green phases in the state representation, and we ignore the yellow phases, since they are transition phases and usually have fixed duration.
        if self.current_phase in self.green_phases:
            phase_one_hot[self.green_phases.index(self.current_phase)] = 1.0
        else:
            phase_one_hot[self.current_green_phase_idx] = 1.0

        phase_index = self.current_phase / max(self.num_phases - 1, 1)
        phase_timer = min(self.phase_timer / max(self.max_green_time, 1), 1.0)
        min_green = 1.0 if self.phase_timer >= self.min_green_time else 0.0
        is_transitioning = 1.0 if self.is_transitioning else 0.0

        raw_vehicle_queue = self.get_vehicle_queue()
        vehicle_density = self.get_vehicle_density()
        ped_queue = self.get_pedestrian_queue()
        vehicle_wait = self.get_vehicle_waiting_time()
        ped_wait = self.get_pedestrian_waiting_time()
        pedestrian_crossing = self.get_pedestrian_crossing_occupancy()

        # Normalize the queue length to [0, 1] ny dividing the MAX_VEHICLE and MAX_PED.
        max_vehicle, max_ped = self._capacity_estimates()
        vehicle_queue = [min(q / max_vehicle, 1.0) for q in raw_vehicle_queue]
        ped_queue = [min(q / max_ped, 1.0) for q in ped_queue]
        vehicle_wait_norm = min(vehicle_wait / max(len(self.lanes) * self.max_green_time, 1), 1.0)
        ped_wait_norm = min(ped_wait / max(4 * self.max_green_time, 1), 1.0)

        state_vector = np.concatenate([
            phase_one_hot,
            np.array([
                phase_index,
                phase_timer,
                min_green,
                is_transitioning,
            ], dtype=np.float32),
            np.array(vehicle_queue, dtype=np.float32),
            np.array(vehicle_density, dtype=np.float32),
            np.array([
                vehicle_wait_norm,
            ], dtype=np.float32),
            np.array(ped_queue, dtype=np.float32),
            np.array([
                ped_wait_norm,
                pedestrian_crossing,
            ], dtype=np.float32),
        ]).astype(np.float32)

        state_vector = np.clip(state_vector, 0.0, 1.0)

        raw_state_feature = {
            "phase_one_hot": phase_one_hot, # List
            "phase_index": phase_index,
            "phase_timer": phase_timer,
            "min_green": min_green, # Number
            "is_transitioning": is_transitioning,
            "vehicle_queue": vehicle_queue, # List
            "vehicle_density": vehicle_density,
            "vehicle_waiting_time": vehicle_wait_norm,
            "ped_queue": ped_queue, # List
            "pedestrian_waiting_time": ped_wait_norm,
            "pedestrian_crossing": pedestrian_crossing,
            "state_vector": state_vector,
        }
        return raw_state_feature

    def get_valid_actions(self):
        """
        Already implemented in the ConstraintChecker class, 
        which is responsible for checking the constraints and return the valid actions
        for the traffic signal and conflicting pedestrian directions for each green phase.
        """
        return self.constraint_checker.get_valid_actions()

        
        









        
