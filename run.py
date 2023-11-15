import gym
import textworld.gym
import re
import time
from agent import AgentFactory
from argparse import ArgumentParser

import warnings

warnings.simplefilter("ignore")


def main(args):
    agent = AgentFactory.create(args.model)

    pattern = r"<CMD>(.*?)<\/CMD>"

    # Register a text-based game as a new Gym's environment.
    env_id = textworld.gym.register_game(args.game, max_episode_steps=50)
    env = gym.make(env_id)  # Start the environment.

    obs, infos = env.reset()  # Start new episode.
    print(obs)
    # env.render()

    score, moves, done = 0, 0, False
    # command = agent.initial_action()
    command = agent.act(obs)
    print(command)
    while not done:
        command = command.replace("</s>", "")
        # print(command)
        command = re.search(pattern, command).group(1)
        obs, score, done, infos = env.step(command)
        if done:
            print(obs)
            break
        command = agent.act(obs.replace("\n", ""))
        print(command)
        time.sleep(1)
        input(">")
        print(obs)
        # env.render()
        moves += 1

    env.close()
    print("moves: {}; score: {}".format(moves, score))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--game", required=True)
    args = parser.parse_args()
    main(args)