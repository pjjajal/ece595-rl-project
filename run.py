import warnings
warnings.simplefilter("ignore")

import gym
import textworld.gym
import re
import time

### Local Imports
from agent import AgentFactory
from argparse import ArgumentParser

def main(args):
    ### Instantiate agent
    agent = AgentFactory.create(args.model)

    ### Register a text-based game as a new Gym's environment.
    env_id = textworld.gym.register_game(args.game, max_episode_steps=50)
    env = gym.make(env_id)  # Start the environment.

    ### Create Initial State, start new episode
    pattern = r"<CMD>(.*?)<\/CMD>"
    obs, infos = env.reset() 
    score, moves, done = 0, 0, False

    ### Take the first action
    command = agent.act(obs)

    while not done:
        command = command.replace("</s>", "")
        ### TODO: Seems to be source of regex errors
        command = re.search(pattern, command).group(1)
        obs, score, done, infos = env.step(command)
        if done:
            break

        ### Act
        command = agent.act(obs.replace("\n", ""))
        time.sleep(1)
        input(">")
        moves += 1

    env.close()
    print("moves: {}; score: {}".format(moves, score))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--game", required=True)
    args = parser.parse_args()
    main(args)