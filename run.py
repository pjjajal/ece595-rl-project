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

###
### Run episode
###
def run_episode(agent : Agent, environment) -> Tuple:
    ### Reset agent 
    agent.reset_chat()

    ### Reset environment
    observation, info = environment.reset()

    print("observation {}: {}".format("initial", observation.replace("\n", "")))

    ### Hardcoded
    pattern = r"<CMD>(.*?)<\/CMD>"

    ### Data to track
    score, moves, done = 0, 0, False

    ### Take the first action
    if not args.manual_mode:
        command = agent.act(observation)
    else:
        command = input("> ")

    while True:        
        ### Increment moves!
        moves += 1
    
        ### Get command from agent outputs
        if not args.manual_mode:
            command = command.replace("</s>", "")
            command = re.search(pattern, command).group(1)

            print("command {}: {}".format(moves, command))

        observation, score, done, info = environment.step(command)

        print("observation {}: {}".format(moves, observation.replace("\n", "")))

        ### Act
        if not args.manual_mode:
            command = agent.act(observation.replace("\n", ""))
        else:
            command = input("> ")
        
        ### Check confusion after agent acts, exit early potentially
        if agent.is_confused or done:
            break

        ### Render
        environment.render()
    
    ### Compute 'win' from game output
    win = "You lost" not in observation and not agent.is_confused
    
    return win, score, moves, agent.is_confused

###
### Single model + Episode evaluation for one game
###
def main(args : argparse.Namespace):
    ### Instantiate agent
    agent = AgentFactory.create(args.model, args.llama_version)

    ### Register a text-based game as a new Gym's environment.
    env_id = textworld.gym.register_game(args.game, max_episode_steps=50)
    env = gym.make(env_id, new_step_api=False)

    win, score, moves, confused = run_episode(agent, env)

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
    
    parser.add_argument("--model", type=str, default="mistral")
    parser.add_argument("--game", type=str, required=True)
    parser.add_argument("--llama-version", type=str, default="13B")
    args = parser.parse_args()

    ### Decide evaluation mode
    if not args.test_all:
        main(args)
    else:
        test_all_main(args)