import warnings

import gym
import textworld.gym
import re

### Local Imports
from agent import AgentFactory
from argparse import ArgumentParser

def main(args):
    ### Instantiate agent
    agent = AgentFactory.create(args.model, args.llama_version)

    ### Register a text-based game as a new Gym's environment.
    env_id = textworld.gym.register_game(args.game, max_episode_steps=50)
    env = gym.make(env_id, new_step_api=False)

    ### Create Initial State, start new episode
    pattern = r"<CMD>(.*?)<\/CMD>"
    obs, info = env.reset()

    ### Data to track
    score, moves, done = 0, 0, False

    ### Take the first action
    command = agent.act(obs)

    while True:
        ### Normal exit conditions - check doneness
        if done:
            print("run.py: Exiting game")
            break
        
        ### Get command from agent outputs
        command = command.replace("</s>", "")
        command = re.search(pattern, command).group(1)

        ### This is bugged...
        ### All of these fields are default values
        ### Only 'obs' seems to be updated properly
        #obs, score, done, info = env.step(command)
        obs, _, _, _ = env.step(command)

        ### Update doneness flag
        if "*** The End ***" in obs:
            score = 100
            done = True

        ### Act
        command = agent.act(obs.replace("\n", ""))

        ### Wait for user input
        input(">")

        ### Increment moves!
        moves += 1
        
        ### Check confusion after agent acts, exit early potentially
        if agent.is_confused:
            print("run.py: Agent confused, exiting game early")
            break

        ### Render
        env.render()

    env.close()
    print("run.py: took {} moves and got score={}".format(moves, score))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="mistral")
    parser.add_argument("--game", type=str, required=True)
    parser.add_argument("--llama-version", type=str, default="13B")
    args = parser.parse_args()
    main(args)