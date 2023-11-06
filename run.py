import gym
import textworld.gym
import re
import time
from mistral import Agent

import warnings

warnings.simplefilter("ignore")

agent = Agent()

pattern = r"<CMD>(.*?)<\/CMD>"

# Register a text-based game as a new Gym's environment.
env_id = textworld.gym.register_game("games/thunt_1.z8", max_episode_steps=50)
env = gym.make(env_id)  # Start the environment.

obs, infos = env.reset()  # Start new episode.
print(obs)
# env.render()

score, moves, done = 0, 0, False
command = agent.initial_action()
while not done:
    command = command.replace("</s>", "")
    print(command)
    command = re.search(pattern, command).group(1)
    print("COMMAND:", command)
    obs, score, done, infos = env.step(command)
    print(obs, score, done, infos)
    command = agent.act(obs.replace("\n", ""))
    print(command)
    time.sleep(1)
    input(">")
    # print(obs)
    # env.render()
    moves += 1

env.close()
# print("moves: {}; score: {}".format(moves, score))
