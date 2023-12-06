import re
import time
import argparse

### Gym / TextWorld stuff
import gym
import textworld.gym

### TRL
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler

### Local Imports
from agent import AgentFactory
from argparse import ArgumentParser

### We don't actually care about the answer - we just want to make sure the model contains a properly formatted <CMD> tag
def reward_function(model_response : str, answer : str) -> float:
    pattern = r"</CMD>"
    cmd_exists = re.search(pattern, model_response) is None
    
    if cmd_exists:
        return 0.0
    else:
        return -1.0


def main(args : argparse.Namespace):
    ### Instantiate agent
    agent = AgentFactory.create(args.model)

    ### Register a text-based game as a new Gym's environment.
    game_env_id = textworld.gym.register_game(args.game, max_episode_steps=50)
    game_env = gym.make(game_env_id)

    ### Get a TRL text environment
    #trl_text_env = t

### Train
def finetune(agent, env):
    pass



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--game", required=True)
    args = parser.parse_args()
    main(args)