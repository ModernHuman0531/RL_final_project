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
        self.current_phase = current_phase
        self.current_green_phase = 0
        self.transition_queue = []
        self.is_transitioning = False
        self.phase_timer = 0

        # Get the lanes controlled by this traffic signal and makes it mutable for later use
        self.lanes = list(
            dict.fromkeys(sumo_traci.trafficlight.getControlledLanes(self.intersection_id)) # dict.fromkeys() takes tuple and turn it into a dictionary with the tuple elements as keys and None as values, 
            # then we take the keys and make it a list.
        )
        # Define the corresponding phase to action mapping.
        self.green_phases = [0, 3] # 0 is NS green, 3 is EW green, and the rest are yellow phases.
        self.transition = {
            (0, 3):[1, 2, 3], 
            (3, 0):[4, 5, 0]
        }
        # Set edge to direction mapping for later use in pedestrian counting.
        self.edge_to_direction = {
            "n_in": "N",
            "e_in": "E",
            "s_in": "S",
            "w_in": "W"
        }
    

    def set_phase(self, action):
        """
        Set the traffic light phase for the intersection.
        This function only get the target phase from action and transition phases,
        the update part is handled in the main environment file(sumo_env.py).
        """
        # action 0: set NS green, action 1: set EW green
        target_phase = self.green_phases[action]

        if target_phase == self.current_green_phase:
            return
        
        # Get the transition phases between the current phase and target phase.
        path = self.transition[(self.current_green_phase, target_phase)]
        self.is_transitioning = True
        self.current_green_phase = target_phase
        self.transition_queue = list(path)  # copy so pop() doesn't mutate self.transition
        self.phase_timer = 0 # Get into the first transition phase in the next update call, so we set the timer to 0 here.
    
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
                if len(self.transition_queue) == 0:
                    self.is_transitioning = False
                self.current_phase = next_phase
                self.sumo_traci.trafficlight.setPhase(self.intersection_id, self.current_phase)

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

    def get_vehicle_waiting_time(self):
        """
        Get the total waiting time of vehicles in the lanes controlled by this traffic signal and return it as a single value.
        """
        waiting_time = 0
        for lane in self.lanes:
            waiting_time += self.sumo_traci.lane.getWaitingTime(lane)
        return waiting_time

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
        # Get the specified edges pedestrians are currently on.
        for edge_id in self.edge_to_direction.keys():
            ped_ids = self.sumo_traci.edge.getLastStepPersonIDs(edge_id)
            for pid in ped_ids:
                # Get the current edge of the pedestrian
                edge = self.sumo_traci.person.getRoadID(pid)

                if self.sumo_traci.person.getWaitingTime(pid) > 0:
                    direction = self.edge_to_direction[edge]
                    ped_queue[direction] += 1
        
        return list(ped_queue.values())


    
    def get_pedestrian_waiting_time(self):
        """
        Get the total waiting time of pedestrians in the lanes controlled by this traffic signal and return it as a single value.
        """
        total_waiting_time = 0

        for edge_id in self.edge_to_direction.keys():
            ped_ids = self.sumo_traci.edge.getLastStepPersonIDs(edge_id)
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
        phase_one_hot = np.zeros(len(self.green_phases))
        # For simplicity, we only consider the green phases in the state representation, and we ignore the yellow phases, since they are transition phases and usually have fixed duration.
        if self.current_green_phase in self.green_phases:
            phase_one_hot[self.green_phases.index(self.current_green_phase)] = 1
        min_green = 1 if self.phase_timer >= self.min_green_time else 0
        vehicle_queue = self.get_vehicle_queue()
        ped_queue = self.get_pedestrian_queue()

        # Normalize the queue length to [0, 1] ny dividing the MAX_VEHICLE and MAX_PED.
        length = self.sumo_traci.lane.getLength(self.lanes[0]) # Get the length of the lane, which is the same for all lanes controlled by this traffic signal, and use it as the max queue length, since if the queue length exceeds the lane length, it means the lane is fully blocked.
        car_length = 5.0 # Average car length
        pedestrain_lenth = 0.215 # Official website pedestrian class's length
        MAX_VEHICLE = length / car_length
        MAX_PED = length / 0.215
        vehicle_queue = [q / MAX_VEHICLE for q in vehicle_queue]
        ped_queue = [q / MAX_PED for q in ped_queue]

        raw_state_feature = {
            "phase_one_hot": phase_one_hot, # List
            "min_green": min_green, # Number
            "vehicle_queue": vehicle_queue, # List
            "ped_queue": ped_queue # List
        }
        return raw_state_feature

    def get_valid_actions(self):
        """
        Get the valid actions for the current state of the traffic signal.
        For baseline scenario, the constraint only comes from min_green_time and max_green_time, and yellow_time.
        If it's extension scenario, we can also add other constraints, such as pedestrian safety constraint, which means if there are pedestrians waiting, we cannot change to a green phase that will cut them off.
        """
        # Current action space is only 2 actions, which is to set NS green or set EW green, so we only need to check the constraints for these two actions.
        valid_actions = [1, 1]# Default is both actions are valid.

        # Condition 1: Yellow time is fixed, so if we are in the yellow phase, we cannot change to any green phase.
        if self.is_transitioning:
            valid_actions = [0, 0]
            # The action is in the transition process to current_green_phase, and in the logic of set_phase function,
            # if the target phase is the same as current_green_phase, we'll just ignore the action and keep the current phase, don't need to worry we'll reset the timer in that case.
            valid_actions[self.green_phases.index(self.current_green_phase)] = 1 # Only the action that keeps the current green phase is valid.
            return valid_actions
        
        # Condition 2: Didn't reach min_green_time, so we cannot change to another green phase.
        if self.phase_timer < self.min_green_time:
            valid_actions = [0 ,0]
            valid_actions[self.green_phases.index(self.current_green_phase)] = 1 # Only the action that keeps the current green phase is valid.
        
        # Condition 3: Reached max_green_time. so we have to change to another green phase.
        elif self.phase_timer >= self.max_green_time:
            valid_actions = [1 ,1]
            valid_actions[self.green_phases.index(self.current_green_phase)] = 0 # Only the action that changes the current green phase is valid.

        return valid_actions
        
        









        
