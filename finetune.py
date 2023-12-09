import re
import time
import argparse
from typing import Dict, List, Tuple, Any
from argparse import ArgumentParser

### Gym / TextWorld stuff
import gym
import textworld.gym

### Torch
import torch

### Progress Bar!
from tqdm import tqdm

### TRL
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed, TextEnvironment

### Local Imports
from agent import AgentFactory
from agent.agent import Agent
from utils import sanitize_observation, sanitize_response

### Helper function for tokenizing observations
### May need to change / modify
def tokenize_observation(agent : Agent, observation : str) -> Any:
    return agent.tokenizer(observation, return_tensors="pt").input_ids.cuda()

###
### Run episode w/ 
###
def run_episode(agent : Agent, environment, ppo_trainer : PPOTrainer) -> Tuple:
    ### Reset agent 
    agent.reset_chat()

    ### Track queries and responses (need to encode both of these eh?)
    ### Used for our visualization
    episode_query_string = ""
    episode_response_string = ""

    episode_query_tensor_list = []
    episode_response_tensor_list = []

    ### Reset environment
    observation, info = environment.reset()
    observation = sanitize_observation(observation, False)

    ### Record as string and tensor
    episode_query_string = observation
    episode_query_tensor_list.append( tokenize_observation(agent, observation) )

    ### Hardcoded
    pattern = r"<CMD>(.*?)<\/CMD>"

    ### Data to track
    score, moves, done = 0, 0, False

    ### Take the first action
    response = agent.act(observation)

    while True:
        ### Increment moves!
        moves += 1
    
        ### Get command from agent outputs
        response = response.replace("</s>", "")
        response = re.search(pattern, response).group(1)
        response = sanitize_response(response)

        ### Append to response list
        episode_response_string += response
        episode_response_tensor_list.append( tokenize_observation(agent, observation) )

        ### Let the environment act and sanitize the output
        observation, score, done, info = environment.step(response)

        ### Check this so we don't append the final observation
        if not done:
            observation = sanitize_observation(observation)
            episode_query_string += observation
            episode_query_tensor_list.append( tokenize_observation(agent, observation) )

        ### Act
        response = agent.act(observation)
        
        ### Check confusion after agent acts, exit early potentially
        if agent.is_confused or done:
            break

        ### Render
        ### environment.render()
    
    ### Compute 'win' from game output
    win = "You lost" not in observation and not agent.is_confused and moves < 32

    ### Compute reward based on win and score
    if win:
        reward = score
    else:
        reward = score - 10.0

    ### Last thing - strip last space (if it exists) for printing
    episode_query_string = episode_query_string.strip()
    episode_response_string = episode_response_string.strip()

    print("Reward: {}\nEpisode Query: {}\nEpisode Response: {}".format(reward, episode_query_string, episode_response_string))

    ### Return info
    return episode_query_tensor_list, episode_response_tensor_list, [torch.tensor(data=[reward])]

def main(args : argparse.Namespace):
    ### Instantiate agent
    agent = AgentFactory.create(args.model, args.llama_version)

    ### Register a text-based game as a new Gym's environment.
    game_env_id = textworld.gym.register_game(args.game, max_episode_steps=32)
    game_env = gym.make(game_env_id)

    ### Create PPO Config
    ppo_config_args = {
        "batch_size" : 1,
        "learning_rate" : 1.5e-5,
    }

    ### Generation kwargs
    # generation_kwargs = {"do_sample" : True, "top_p" : 0.95, "top_k" : 32, "temperature" : 0.50, "max_new_tokens" : 32}

    ### Config
    ppo_config = PPOConfig(**ppo_config_args)

    ### Create PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=agent.model,
        ref_model=None,
        tokenizer=agent.tokenizer,
    )

    ### Get the episode query and response strings
    query_tensors, response_tensors, reward_tensor = run_episode(agent, game_env, ppo_trainer)

    print(f"Query Tensor Len: {(query_tensors)}\nResponse Tensor Len: {(response_tensors)}\n")

    ### Tokenize these, then pass these through the ppotrainer to update the model
    stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensor)

    ### Save our model
    ppo_trainer.save_model(f"models/test_{agent.model_name}.pth")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--game", required=True)
    parser.add_argument("--llama-version", type=str, default="13B")
    args = parser.parse_args()
    main(args)