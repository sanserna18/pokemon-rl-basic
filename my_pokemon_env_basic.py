import os
import sys
import random


import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import csv
from helpers.memory_map import PokemonMemory


class MyPokemonEnv(gym.Env):
    """Ultra-minimal Pokemon Red env with CSV logging of model I/O."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.debug = self.config.get('debug', False)
        self.headless = self.config.get('headless', False)  # Default to headless for basic env

        # Basic config
        self.rom_path = self.config.get('gb_path')
        self.init_state = self.config.get('init_state')
        if not self.rom_path or not os.path.exists(self.rom_path):
            raise FileNotFoundError("gb_path not set or ROM not found")
        if not self.init_state or not os.path.exists(self.init_state):
            raise FileNotFoundError("init_state not set or file not found")

        # Emulator
        self.pyboy = PyBoy(self.rom_path, window="null" if not self.debug else "SDL2", sound_emulated=False)
        self.pyboy.set_emulation_speed(0)
        self.memory = PokemonMemory(self.pyboy)

        # Action space: 6 buttons
        self.valid_events = [
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]
        self.action_space = spaces.Discrete(len(self.valid_events))

        # Observation: minimal numeric vector
        # [start_menu_open, x, y, map_id]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

        # Setup CSV logging
        basic_folder = os.path.dirname(os.path.abspath(__file__))
        log_dir = self.config.get('session_path') or basic_folder
        os.makedirs(log_dir, exist_ok=True)
        self.io_log_path = os.path.join(log_dir, f'reports/{self.config["reward_version"]}_{random.randint(1, 1000000)}.csv')
        self.io_log_file = open(self.io_log_path, 'w', newline='')
        self.io_csv_writer = csv.writer(self.io_log_file)
        self.io_csv_writer.writerow([
            'step',
            'action_name',
            'reward',
            'obs_distance_from_entrance', 'obs_distance_from_last_position_from_high_reward', 'obs_reward_taken_flag', 'obs_relevant_thing_in_current_map',
            'map_id', 'x', 'y',
        ])
        self.reward_config = self.config['reward_version']
        print(f"Reward configuration: {self.reward_config}")

        # Several sets for observations
        self.last_pos = (40, 5, 3)
        self.second_last_pos = (40, 5, 3)
        self.last_reward = 0.0
        self.new_map_found = False
        self.first_position_in_a_new_map = set([(40,5,3)])
        self.overall_high_reward_tiles_per_map = set()

        self.session_high_reward_tiles_per_map = set()
        self.session_first_position_in_a_new_map = set([(38,4,6)])
        self.session_new_map_found = False
        self.session_reward = 0.0
        self.session_party_number = 0

        

        self.step_count = 0

        # Reset to initial state
        self.reset()

    def _read_map(self):
        # Use our memory helper (x, y, map_id)
        try:
            x, y, map_id = self.memory.read_position()
            return int(map_id), int(x), int(y)
        except Exception:
            return 0, 0, 0

    def _encode_map_id(self, map_id):
        """One-hot encode map ID for better neural network understanding"""
        common_maps = [0, 37, 38, 40, 41, 42, 43, 44, 45]
        map_vector = np.zeros(len(common_maps))
        if map_id in common_maps:
            map_vector[common_maps.index(map_id)] = 1.0
        return map_vector

    def _get_obs(self):

        ## 1. Steps before calculations for observations
        map_id, x, y = self._read_map()
        
        # Code for first_position_in_a_new_map
        visited_maps = {pos[0] for pos in self.first_position_in_a_new_map}
        if map_id not in visited_maps:
            self.new_map_found = True
            self.first_position_in_a_new_map.add((map_id, x, y))
            print(f"OVERALL: New map found: {map_id} on Step {self.step_count}")
        else:
            self.new_map_found = False
        
        visited_maps = {pos[0] for pos in self.session_first_position_in_a_new_map}
        if map_id not in visited_maps:
            self.session_new_map_found = True
            self.session_first_position_in_a_new_map.add((map_id, x, y))
            print(f"SESSION: New map found: {map_id} on Step {self.step_count}")
        else:
            self.session_new_map_found = False

        # Add the new high reward tile to the overall high reward tile per map
        if self.last_reward >= 1.0:
            visited_maps = {pos[0] for pos in self.overall_high_reward_tiles_per_map}
            if map_id not in visited_maps:
                self.overall_high_reward_tiles_per_map.add(self.second_last_pos)

            visited_maps = {pos[0] for pos in self.session_high_reward_tiles_per_map}
            if map_id not in visited_maps:
                self.session_high_reward_tiles_per_map.add(self.second_last_pos)
            

        ## 2. Calculations for observations
        # Calculate distance to first_position_in_a_new_map
        matching_tuple = next((pos for pos in self.first_position_in_a_new_map if pos[0] == map_id), None)
        distance_from_entrance = abs(matching_tuple[1] - x) + abs(matching_tuple[2] - y)

        # Calculate distance to last_position_from_high_reward
        matching_tuple = next((pos for pos in self.overall_high_reward_tiles_per_map if pos[0] == map_id), None)
        reward_taken_flag = any(pos[0] == map_id for pos in self.session_high_reward_tiles_per_map)
        
        if matching_tuple is not None:
            distance_from_last_position_from_high_reward = abs(matching_tuple[1] - x) + abs(matching_tuple[2] - y)
        else:
            distance_from_last_position_from_high_reward = -1.0

        map_encoding = self._encode_map_id(map_id)

        obs = np.concatenate([
            map_encoding,
            np.array([distance_from_entrance, distance_from_last_position_from_high_reward, reward_taken_flag, self.memory.relevant_thing_in_current_map()])
        ])

        return obs

    def _calculate_reward(self, current_pos):
        reward = 0.0
        """Calculate reward based on position change"""
        last_pos = self.last_pos
        if last_pos is not None and last_pos != current_pos:
            reward += self.reward_config['movement']  # Moved to new tile
        else:
            reward += self.reward_config['penalty_same_position']

        """Calculate reward based on new room"""
        if self.session_new_map_found == True:
            reward += self.reward_config['map_new']

        if self.session_new_map_found == False and current_pos[0] != last_pos[0] and self.step_count > 0:
            reward += self.reward_config['penalty_revisit_map']

        if self.memory.read_party() > self.session_party_number:
            reward += self.reward_config['new_pokemon_in_party']
            print(f"SESSION: New pokemon in party: {self.memory.read_party()}")
            self.session_party_number = self.memory.read_party()
        
        self.session_reward += reward

        return reward

    def _log_step_to_csv(self, action, reward, obs, current_pos):
        """Log the step data to CSV file"""
        action_names = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"]
        action_name = action_names[int(action)] if 0 <= int(action) < len(action_names) else f"UNKNOWN({action})"
        input_obs = getattr(self, 'last_observation', obs)
        
        self.io_csv_writer.writerow([
            self.step_count + 1,
            action_name,
            float(reward),
            float(input_obs[0]), float(input_obs[1]), float(input_obs[2]), float(input_obs[3]),
            int(current_pos[0]), int(current_pos[1]), int(current_pos[2]),
        ])
       
        self.io_log_file.flush()


    def reset(self, seed=None):
        super().reset(seed=seed)
        with open(self.init_state, 'rb') as f:
            self.pyboy.load_state(f)
        # Warm up a few frames
        for _ in range(30):
            self.pyboy.tick()
        self.step_count = 0
        obs = self._get_obs()
        self.last_observation = obs

        self.session_new_map_found = False
        self.session_first_position_in_a_new_map = set([(40,5,3)])
        self.session_party_number = 0
        self.session_high_reward_tiles_per_map = set()
        self.session_reward = 0.0
        

        return obs, {}

    def step(self, action):
        # Send button press
        event = self.valid_events[int(action)]
        self.pyboy.send_input(event)
        for _ in range(12):
            self.pyboy.tick()
        # Try to release if it's a press-type event
        release_map = {
            WindowEvent.PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START: WindowEvent.RELEASE_BUTTON_START,
        }
        self.pyboy.send_input(release_map.get(event, event))

        # Let game progress
        for _ in range(108):
            self.pyboy.tick()

        # Calculate reward based on position change

        map_id, x, y = self._read_map()
        
        current_pos = (map_id, x, y)
        obs = self._get_obs()

        reward = self._calculate_reward(current_pos)

        self.last_reward = reward
        self.second_last_pos = self.last_pos
        self.last_pos = current_pos

        self._log_step_to_csv(action, reward, obs, current_pos)

        self.step_count += 1
        self.last_observation = obs

        action_names = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"]
        action_name = action_names[int(action)] if 0 <= int(action) < len(action_names) else f"UNKNOWN({action})"
        return obs, reward, False, False, {}

    def close(self):
        self.pyboy.stop()
        try:
            self.io_log_file.close()
            print(f"CSV logging complete. Data saved to {self.io_log_path}")
        except Exception:
            print(f"Error closing CSV file: {e}")
            pass


