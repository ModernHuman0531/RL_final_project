"""
This file turn the SUMO simulation into MDP problem(Gym environment) and provide the interface for the agent to interact with the ebvironment.
"""
import os
import sys
import numpy as np

import gymnasium as gym
from gymnasium import spaces
import traci

from .traffic_signal import TrafficSignalEnv

# Import traci in a script
if 'SUMO_HOME' in os.environ:
    tools_path = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools_path)


class SUMOEnvironment(gym.Env):
    """
    This class defines the SUMO environment(MDP) for reinforcement learning.
    It is responsible for initializing the SUMO simulation, creating traffic signal objects for each intersection,
    and providing the interface for the agent to interact with the environment. 
    Args:
        sumo_cfg_file: The path to the SUMO configuration file.
        delta_time: The time step for the simulation.
        yellow_time: The duration of the yellow light.
        min_green_time: The minimum duration of the green light.
        max_green_time: The maximum duration of the green light.
        begin_time: The start time of the simulation.
        end_time: The end time of the simulation.
        controlled_tls: A list of traffic light ids that are controlled by the agent. If None, all traffic lights in the simulation will be controlled.
        use_gui: Whether to use the SUMO GUI.
        enable_pedestrian_safety: Whether use the pedestrian safety constraint, which means that the traffic signal cannot switch to a phase that would endanger pedestrians who are currently crossing the intersection.
    """
    def __init__(
        self,
        sumo_cfg_file: str,
        delta_time: int=1,
        yellow_time: int=5,
        min_green_time: int=10,
        max_green_time: int=60,
        begin_time: int=0,
        end_time: int=3600,
        controlled_tls: list=None,
        use_gui: bool=False,
        enable_pedestrian_safety: bool=True,
        reward_mode: str="waiting_time",
        queue_reward_weight: float=1.0,
        vehicle_wait_weight: float=1.0,
        pedestrian_wait_weight: float=1.0,
        violation_penalty: float=50.0
    ):
        self.sumo_cfg_file = sumo_cfg_file
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.begin_time = begin_time
        self.end_time = end_time
        self.use_gui = use_gui
        self.enable_pedestrian_safety = enable_pedestrian_safety
        self.controlled_tls = controlled_tls
        self.reward_mode = reward_mode
        self.queue_reward_weight = queue_reward_weight
        self.vehicle_wait_weight = vehicle_wait_weight
        self.pedestrian_wait_weight = pedestrian_wait_weight
        self.violation_penalty = violation_penalty

        self.sumo_traci = traci
        self.sumo_running = False
        # Get the list of traffic light id from .sumocfg file to create traffic signal objects.
        self.traffic_signal_ids = []
        self.traffic_signals = {}
        self.intersection_num = 0
        self.last_invalid_actions = 0
        self.last_reward_components = {}
        

        # Placeholder spaces — redefined in reset() once SUMO reveals the real
        # intersection count. Using 1 here avoids a shape=(0,) invalid space.
        self.action_space = spaces.MultiDiscrete([2])
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32
        )
    
    def _build_traffic_signals(self):
        """
        Build the traffic signal objects for each intersection in the simulation
        using TrafficSignalEnv class defined in traffic_signal.py.
        """
        # Usd a dictionary to store the traffic signal objects, with the key being the intersection id and the value being the traffic signal object.
        traffic_signals = {}
        for intersection_id in self.traffic_signal_ids:
            traffic_signals[intersection_id] = TrafficSignalEnv(
                sumo_traci = self.sumo_traci,
                env = self,
                intersection_id = intersection_id,
                yellow_time = self.yellow_time,
                min_green_time = self.min_green_time,
                max_green_time = self.max_green_time,
                begin_time = self.begin_time,
                end_time = self.end_time,
                current_phase = self.sumo_traci.trafficlight.getPhase(intersection_id),
                enable_pedestrian_safety_constraint = self.enable_pedestrian_safety
            )
        return traffic_signals
    
    def _update_spaces(self):
        """
        According to the number of intersections and phases to 
        dynamically set the gym space.

        In this environment(3x3.net.xml + controlled_tls=["B1"]):
        action_space = MultiDiscrete([4]) # 4 phases for the single intersection
        observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float32) 
        4(Green phases) + 1(min_green) + 4(vehicle_queue) + 8(ped_queue) = 17 features for each intersection.
        """
        n_actions = [len(signal.green_phases) for signal in self.traffic_signals.values()]
        self.action_space = spaces.MultiDiscrete(n_actions)
        total_obs = sum(signal.state_size for signal in self.traffic_signals.values())
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(total_obs,), dtype=np.float32
        )

    # ----- Implement calculational functions for reward and state features ----- #
    def get_total_vehicle_queue(self):
        return sum(sum(signal.get_vehicle_queue()) for signal in self.traffic_signals.values())

    def get_total_vehicle_waiting_time(self):
        return sum(signal.get_vehicle_waiting_time() for signal in self.traffic_signals.values())

    def get_total_pedestrian_waiting_time(self):
        return sum(signal.get_pedestrian_waiting_time() for signal in self.traffic_signals.values())

    def get_total_pedestrian_crossing_occupancy(self):
        return sum(signal.get_pedestrian_crossing_occupancy() for signal in self.traffic_signals.values())

    def calculate_reward(self, previous_queue=None, current_queue=None, invalid_actions=0):
        """
        Calculate the reward for the current state take action in the environment.
        Supported reward modes:
        - waiting_time: negative weighted vehicle and pedestrian waiting time.
        - queue_delta: reward queue reduction from before to after the action.
        - hybrid: queue reduction minus weighted wait and safety repair penalties.
        """
        vehicle_waiting_time = self.get_total_vehicle_waiting_time()
        ped_waiting_time = self.get_total_pedestrian_waiting_time()
        if current_queue is None:
            current_queue = self.get_total_vehicle_queue()
        if previous_queue is None:
            previous_queue = current_queue

        queue_delta = previous_queue - current_queue

        if self.reward_mode == "queue_delta":
            reward = self.queue_reward_weight * queue_delta
        elif self.reward_mode == "hybrid":
            reward = (
                self.queue_reward_weight * queue_delta
                - self.vehicle_wait_weight * vehicle_waiting_time
                - self.pedestrian_wait_weight * ped_waiting_time
                - self.violation_penalty * invalid_actions
            )
        else:
            reward = -(
                self.vehicle_wait_weight * vehicle_waiting_time
                + self.pedestrian_wait_weight * ped_waiting_time
                + self.violation_penalty * invalid_actions
            )

        self.last_reward_components = {
            "reward_mode": self.reward_mode,
            "queue_delta": float(queue_delta),
            "previous_queue": float(previous_queue),
            "current_queue": float(current_queue),
            "vehicle_waiting_time": float(vehicle_waiting_time),
            "pedestrian_waiting_time": float(ped_waiting_time),
            "invalid_actions": int(invalid_actions),
        }
        return reward
    
    def get_state(self):
        """
        Get the current features of the environment provided by the traffic signal objects,
        and turn them into a state representation(Numpy array) that can be fed into the agent.
        """
        all_features = [traffic_signal.get_state_feature() for traffic_signal in self.traffic_signals.values()]
        states = []
        for feature in all_features:
            """
            feature includes:
            - phase_one_hot: one-hot encoding of the current phase, which is a list of length equal to the number of green phases, with 1 indicating the current green phase and 0 indicating the other phases.
            - min_green: whether the minimum green time has been reached, which is a binary value (1 or 0).
            - vehicle_queue: the queue length for each direction, which is a list.
            - ped_queue: the pedestrian queue for each direction, which is a list.

            If above isn't enough, we can also add more features, such as current phase.
            Different from the green phases, current phase includes all the phases, including the yellow phases.
            """
            if "state_vector" in feature:
                state = feature["state_vector"]
            else:
                state = np.concatenate([
                    feature["phase_one_hot"],
                    [feature["min_green"]],
                    feature["vehicle_queue"],
                    feature["ped_queue"]
                ])
            states.append(state)
        states = np.concatenate(states).astype(np.float32)
        return states
    
    def take_action(self, action):
        """
        The reason why we seperate the take_action function from the step function is that
        we need to implement the logic for checking the minimum green time and 
        constraint function for action masking in the main environment file to avoid the illegal action.
        """
        # A list of actions for each traffic signal, where each action is either 0 or 1
        
        self.last_invalid_actions = 0
        action = np.array(action, dtype=np.int32, copy=True)

        for i, signal in enumerate(self.traffic_signals.values()):
            valid_actions = signal.get_valid_actions()
            if valid_actions[action[i]] == 1:
                pass
            else:
                self.last_invalid_actions += 1
                action[i] = self.handle_invalid_action(valid_actions)
        return action

    
    def handle_invalid_action(self, valid_actions):
        """
        The action we want to take is invalid, which means it violates the constraints,
        so we need to pick a valid action from the valid_actions list randomally.

        Should we add penalty for taking invalid action in reward function?
        """
        # Build valid action indicies list, which is the index of the valid actions in the action space.
        valid_indicies = [i for i,valid in enumerate(valid_actions) if valid == 1]
        return np.random.choice(valid_indicies)



    def start_sumo(self):
        """
        Start the SUMO simulation using the traci API.
        For more detail please check the SUMO documentation: https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html
        """
        sumoBinary = "sumo-gui" if self.use_gui else "sumo"
        sumoCmd = [sumoBinary, "-c", self.sumo_cfg_file, 
                "--step-length", str(self.delta_time),
                "--no-step-log", "true"]
        traci.start(sumoCmd)
        self.sumo_running = True
    
    def reset(self):
        """
        Reset the environment to the initial state and return the initial state.
        This function is called at the beginning of each episode.
        Return:
            state: The initial state of the environment after reset.
        """
        if self.sumo_running:
            self.sumo_traci.close()
            self.sumo_running = False
        
        self.start_sumo()
        all_tls = list(self.sumo_traci.trafficlight.getIDList())
        # Apply filter using controlled_tls.
        self.traffic_signal_ids = ([t for t in all_tls if t in self.controlled_tls]
                                   if self.controlled_tls is not None else all_tls)
        self.traffic_signals = self._build_traffic_signals()


        self._update_spaces()
        self.last_invalid_actions = 0
        self.last_reward_components = {}

        state = self.get_state()
        return state, {}
        
        
    def step(self, action):
        """
        Take a step in the environment by applying the given action,
        and return the next_state, reward, terminated, truncated, and info.
        """
        previous_queue = self.get_total_vehicle_queue()

        # Constraint check and action making
        action = self.take_action(action)

        # Execute the action in the SUMO environment using traci API.
        for i, signal in enumerate(self.traffic_signals.values()):
            signal.set_phase(action[i])
        # Run the simulation for one step to let the action take effect.
        for _ in range(self.delta_time):
            for i,signal in enumerate(self.traffic_signals.values()):
                signal.update()
            self.sumo_traci.simulationStep()
        # Get the next state, reward, and check if the episode is terminated.
        next_state = self.get_state()
        current_queue = self.get_total_vehicle_queue()
        reward = self.calculate_reward(
            previous_queue=previous_queue,
            current_queue=current_queue,
            invalid_actions=self.last_invalid_actions,
        )
        terminated = self.is_done()
        truncated = False # We don't have a truncation condition for now.
        info = {
            "applied_action": action.copy(),
            "invalid_actions": self.last_invalid_actions,
            "reward_components": self.last_reward_components.copy(),
        }

        return next_state, reward, terminated, truncated, info
    
    def is_done(self):
        """
        Check the termination condition for the episode. This episode will 
        terminate when the simulation time reaches the end_time defined in the environment.
        """
        current_time = self.sumo_traci.simulation.getTime()
        return current_time >= self.end_time

    def close(self):
        """
        Close the SUMO simulation and clean up resources.
        Call this at the end of training to properly shut down the traci connection.
        """
        if self.sumo_running:
            self.sumo_traci.close()
            self.sumo_running = False

    def get_feasible_actions(self):
        """
        Get the feasible (valid) actions for all intersections in the current state.
        Returns a list of lists, where each inner list contains valid actions for that intersection.
        
        Returns:
            List[List[int]]: For each intersection, a list of valid action indices [0, 1, ...].
        """
        feasible_actions = []
        for signal in self.traffic_signals.values():
            valid_mask = signal.get_valid_actions()  # Returns [0 or 1, 0 or 1]
            valid_indices = [i for i, valid in enumerate(valid_mask) if valid == 1]
            feasible_actions.append(valid_indices)
        return feasible_actions
    
    



