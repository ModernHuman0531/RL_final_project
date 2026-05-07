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
        use_gui: Whether to use the SUMO GUI.
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
        use_gui: bool=False,
    ):
        self.sumo_cfg_file = sumo_cfg_file
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.begin_time = begin_time
        self.end_time = end_time
        self.use_gui = use_gui

        self.sumo_traci = traci
        self.sumo_running = False
        # Get the list of traffic light id from .sumocfg file to create traffic signal objects.
        self.traffic_signal_ids = []
        self.traffic_signals = {}
        self.intersection_num = 0
        

        # Placeholder spaces — redefined in reset() once SUMO reveals the real
        # intersection count. Using 1 here avoids a shape=(0,) invalid space.
        self.action_space = spaces.MultiDiscrete([2])
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(11,),
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
                current_phase = self.sumo_traci.trafficlight.getPhase(intersection_id)
            )
        return traffic_signals

    # ----- Implement calculational functions for reward and state features ----- #
    def calculate_reward(self, alpha=1.0, beta=1.0):
        """
        Calculate the reward for the current state take action in the environment.
        The reward is calculated as a weighted sum of the negative total waiting time 
        of vehicles and pedestrians in the lanes controlled by this traffic signal.
        """
        vehicle_waiting_time, ped_waiting_time = 0, 0
        for signal in self.traffic_signals.values():
            vehicle_waiting_time += signal.get_vehicle_waiting_time()
            ped_waiting_time += signal.get_pedestrian_waiting_time()
        reward = -(alpha * vehicle_waiting_time + beta * ped_waiting_time)
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
        
        for i, signal in enumerate(self.traffic_signals.values()):
            valid_actions = signal.get_valid_actions()
            if valid_actions[action[i]] == 1:
                pass
            else:
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
        self.traffic_signal_ids = list(self.sumo_traci.trafficlight.getIDList())
        self.traffic_signals = self._build_traffic_signals()
        self.intersection_num = len(self.traffic_signals)

        # Redefine spaces now that the real intersection count is known.
        self.action_space = spaces.MultiDiscrete([2] * self.intersection_num)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.intersection_num * 11,),
            dtype=np.float32
        )

        state = self.get_state()
        return state, {}
        
        
    def step(self, action):
        """
        Take a step in the environment by applying the given action,
        and return the next_state, reward, terminated, truncated, and info.
        """
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
        reward = self.calculate_reward()
        terminated = self.is_done()
        truncated = False # We don't have a truncation condition for now.
        info = {}

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
    
    



