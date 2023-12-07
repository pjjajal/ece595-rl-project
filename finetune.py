import re
import time
import argparse
from typing import Dict, List, Tuple, Any
from argparse import ArgumentParser

### Gym / TextWorld stuff
import gym
import textworld.gym

### TRL
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed, TextEnvironment

### Local Imports
from agent import AgentFactory
from agent.agent import Agent

###
### Run episode w/ 
###
def run_episode(agent : Agent, environment) -> Tuple:
    ### Reset agent 
    agent.reset_chat()

    ### Reset environment
    observation, info = environment.reset()

    ### Hardcoded
    pattern = r"<CMD>(.*?)<\/CMD>"

    ### Data to track
    score, moves, done = 0, 0, False

    print("observation {}: {}".format("initial", observation))

    ### Take the first action
    if not args.manual_mode:
        command = agent.act(observation)
    else:
        command = input("> ")

    while True:        
        ### Increment moves!
        moves += 1
    
        ### Get command from agent outputs
        command = command.replace("</s>", "")
        command = re.search(pattern, command).group(1)

        observation, score, done, info = environment.step(command)

        ### Act
        command = agent.act(observation.replace("\n", ""))
        
        ### Check confusion after agent acts, exit early potentially
        if agent.is_confused or done:
            break

        ### Render
        environment.render()
    
    ### Compute 'win' from game output
    win = "You lost" not in observation and not agent.is_confused
    
    return win, score, moves, agent.is_confused

###
### We don't actually care about the answer - we just want to make sure the model contains a properly formatted <CMD> tag
###
def reward_function(model_response : str, answer : str) -> float:
    pattern = r"</CMD>"
    cmd_exists = re.search(pattern, model_response) is not None

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
    initial_game_observation, game_env_info = game_env.reset()

    ### Get a TRL text environment
    # trl_text_env = TextEnvironment(
    #     model=agent.model,
    #     tokenizer=agent.tokenizer,
    #     reward_fn=reward_function,
    #     max_turns=32,
    #     generation_kwargs = {
    #         "do_sample" : "false",
    #         "max_new_tokens" : "32",
    #     }
    # )

    ### Create PPO Config
    ppo_config_args = {
        "batch_size" : 1,
        "learning_rate" : 1e-5,
    }

    ### Config
    ppo_config = PPOConfig(**ppo_config_args)

    ### Create PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config, 
        model=agent.model,
        ref_model=None,
        tokenizer=agent.tokenizer,
        reward_model=reward_function,
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--game", required=True)
    args = parser.parse_args()
    main(args)