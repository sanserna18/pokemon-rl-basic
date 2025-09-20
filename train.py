#!/usr/bin/env python3
"""
Ultra Simple Pokemon Red Training
Just the bare minimum to train a PPO agent.
"""

import os
import sys
import argparse
import json
import time

# Ensure parent directory (ash_folder) is on sys.path so imports like
# `utils` and `reward_config` work when running from the `basic` subfolder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from stable_baselines3 import PPO
from my_pokemon_env_basic import MyPokemonEnv



def parse_args():
    parser = argparse.ArgumentParser(description='Train Pokemon Red agent')
    parser.add_argument('--reward-version', '-rv', type=str, default='v1_basic', help='Reward configuration version to use')
    
    return parser.parse_args()

def load_reward_config(reward_version):
    with open('/Users/jorgesanchez/Documents/AprendizajePKM/AprendizajePKM/ash_folder/basic/reward_configs.json', 'r') as f:
        
        reward_configs = json.load(f)

    
    return reward_configs['versions'][reward_version]['rewards']


def main():
    print("Starting simple Pokemon Red training...")
    args = parse_args()
    
    
    # Basic environment config
    config = {
        'headless': False,  # Run without GUI for speed
        'debug': True,    # This ensures window="null" is used (not self.debug = False, so window="null")
        'init_state': "/Users/jorgesanchez/Documents/AprendizajePKM/AprendizajePKM/PokemonRedExperiments/init.state",
        'gb_path': "/Users/jorgesanchez/Documents/AprendizajePKM/AprendizajePKM/PokemonRedExperiments/PokemonRed.gb",
        'reward_version': load_reward_config(args.reward_version)
    }
    
    # Create environment
    env = MyPokemonEnv(config)
    print("Environment created!")
    
    # Create PPO model with simple settings
    model = PPO("MlpPolicy", env, verbose=1)
    print("PPO model created!")
    
    # Training with periodic resets
    total_training_steps = 2048 * 20
    steps_per_episode = 2048
    episodes = total_training_steps // steps_per_episode
    
    print(f"Training for {episodes} episodes of {steps_per_episode} steps each...")
    
    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        
        # Train for one episode
        model.learn(total_timesteps=steps_per_episode, reset_num_timesteps=False)
        print(f"EPISODE REWARD: {env.session_reward:,.2f}")
        print("Resetting environment...")
        env.reset()

    # Save the model
    model_name = f"pokemon_model_{args.reward_version}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    model.save(f"models/{model_name}")
    print(f"Training complete! Model saved as 'models/{model_name}'")

if __name__ == "__main__":
    main()
