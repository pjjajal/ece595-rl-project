import gym
import textworld.gym

import warnings

warnings.simplefilter("always")

# Register a text-based game as a new Gym's environment.
env_id = textworld.gym.register_game("games/thunt_1.z8", max_episode_steps=50)
print(env_id)
env = gym.make(env_id)  # Start the environment.

obs, infos = env.reset()  # Start new episode.
print(obs)

score, moves, done = 0, 0, False
while not done:
    command = input("> ")
    obs, score, done, infos = env.step(command)
    print(obs, score, done, infos)
    moves += 1

env.close()