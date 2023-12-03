import warnings
#warnings.simplefilter("ignore")

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
    env = gym.make(env_id)

    ### Create Initial State, start new episode
    pattern = r"<CMD>(.*?)<\/CMD>"
    obs, info = env.reset() 
    score, moves, done = 0, 0, False

    ### Take the first action
    command = agent.act(obs)

    while not done:
        command = command.replace("</s>", "")
        command = re.search(pattern, command).group(1)
        obs, score, done, info = env.step(command)
        
        ### Normal exit conditions
        if done:
            print("run.py: Exiting game")
            break

        ### Act
        command = agent.act(obs.replace("\n", ""))
        #time.sleep(1)
        input(">")
        moves += 1
        
        ### Check confusion after agent acts, exit early potentially
        if agent.is_confused:
            print("run.py: Agent confused, exiting game early")
            break

        ### Render
        env.render()

    env.close()
    #print("run.py: moves {}score {}".format(moves, score))
    print("run.py: score={}".format(score))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--game", required=True)
    args = parser.parse_args()
    main(args)