import warnings

import gym
import textworld.gym
import re
import time
from typing import Dict, List, Tuple, Any
from argparse import ArgumentParser

### Local Imports
import argparse
from agent.agent import Agent
from agent import AgentFactory
from utils import sanitize_observation, sanitize_response

###
### Run episode
###
def run_episode(agent : Agent, environment, generation_kwargs) -> Tuple:
    ### Reset agent 
    agent.reset_chat()

    ### Reset environment
    ### IGNORE initial observation which contains TextWorld ASCII art. Instead just do goal and look.
    observation, info = environment.reset()
    #observation = sanitize_observation(observation)

    ### Hardcoded
    pattern = r"<CMD>(.*?)<\/CMD>"

    ### Data to track
    score, moves, done = 0, 0, False

    print("observation {}: {}".format("initial", observation))

    ### Take the first action
    if not args.manual_mode:
        _, command, _ = agent.act(observation, generation_kwargs, None)
    else:
        command = input("> ")

    while True:        
        ### Increment moves!
        moves += 1
    
        ### Get command from agent outputs
        if not args.manual_mode:
            command = command.replace("</s>", "")
            command = re.search(pattern, command).group(1)
            #command = sanitize_response(command)

            print("command {}: {}".format(moves, command))

        observation, score, done, info = environment.step(command)
        #observation = sanitize_observation(observation)

        print("observation {}: {}".format(moves, observation))

        ### Act
        if not args.manual_mode:
            _, command, _ = agent.act(observation, generation_kwargs, None)
        else:
            command = input("> ")
        
        ### Check confusion after agent acts, exit early potentially
        if agent.is_confused or done:
            break

        ### Render
        if args.manual_mode:
            environment.render()
    
    ### Compute 'win' from game output
    win = "You lost" not in observation and not agent.is_confused and moves < args.max_episodes
    
    return win, score, moves, agent.is_confused

###
### Single model + Episode evaluation for one game
###
def main(args : argparse.Namespace):
    ### Instantiate agent
    agent = AgentFactory.create(args.model, args.llama_version)

    ### Register a text-based game as a new Gym's environment.
    env_id = textworld.gym.register_game(args.game, max_episode_steps=args.max_episodes)
    env = gym.make(env_id, new_step_api=False)

    ### For token generation
    generation_kwargs = {
        "do_sample": False,
        "max_new_tokens": 32,
    }

    win, score, moves, confused = run_episode(agent, env, generation_kwargs)

    print("win: {}\nmoves: {}\nscore: {}\nconfused: {}".format(win, moves, score, confused))

###
### Evaluate one episode foreach game using a single model
###
def test_all_main(args : argparse.Namespace):
    pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test-all", action="store_true")
    parser.add_argument("--manual-mode", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=12)
    
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--game", type=str, required=True)
    parser.add_argument("--llama-version", type=str, default="7B")
    args = parser.parse_args()

    ### Decide evaluation mode
    if not args.test_all:
        main(args)
    else:
        test_all_main(args)