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
### Create class for reward tracking
###
class EpisodicRewardTracker:
    def __init__(self):
        self.reward = 0

    def set_reward(self, new_reward : float):
        self.reward = new_reward

    def add_reward(self, reward : float):
        self.reward += reward

    def reset(self):
        self.reward = 0

    ### Used to get around the way PPOTrainer implements a reward model
    def __call__(self, string : str) -> float:
        return self.reward

### Helper function for tokenizing observations
### May need to change / modify
def tokenize_observation(agent : Agent, observation : str) -> str:
    return agent._tokenize(observation)

### Sanitization Functions
def sanitize_observation(observation : str, enforce_alphanumeric_only : bool = False) -> str:
    ### Remove disgusting room info formatting
    room_info_formatting_pattern = r"-=[a-zA-Z0-9_\s]*=-"
    alphanumeric_only_pattern = r"^[\W_]+|[\W_]+$"
    quest_move_counter_pattern = r"[0-9]*/[0-9]*"
    carat_removal_pattern = r"\>"
    multiple_newline_reduce = r"\n+"
    #correct_sentence_pattern = r"\.[a-zA-Z0-9_]+[^$]"

    ### Replace multiple new lines with a single newline
    sanitized_observation = observation

    ### Pattern matching fun
    sanitized_observation = re.sub(room_info_formatting_pattern, '', sanitized_observation)
    sanitized_observation = re.sub(quest_move_counter_pattern, '', sanitized_observation)
    sanitized_observation = re.sub(carat_removal_pattern, '', sanitized_observation)
    sanitized_observation = re.sub(multiple_newline_reduce, ' ', sanitized_observation)

    ### NOTE: Should go after the other operations
    if enforce_alphanumeric_only:
        sanitized_observation = re.sub(alphanumeric_only_pattern, '', sanitized_observation)

    ### Strip to remove any trailing spaces
    sanitized_observation = sanitized_observation.strip()

    return sanitized_observation

def sanitize_response(response : str) -> str:
    sanitized_response = response.strip() + ". "
    return sanitized_response

###
### Run episode w/ 
###
def run_episode(agent : Agent, environment, reward_tracker : EpisodicRewardTracker) -> Tuple:
    ### Reset agent 
    agent.reset_chat()

    ### Track queries and responses (need to encode both of these eh?)
    ### Used by PPOTrainer
    episode_query_string = ""
    episode_response_string = ""

    ### Reset tracker
    reward_tracker.reset()

    ### Reset environment
    observation, info = environment.reset()
    observation = sanitize_observation(observation, True)
    episode_query_string = observation

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

        ### Let the environment act and sanitize the output
        observation, score, done, info = environment.step(response)

        ### Check this so we don't append the final observation
        if not done:
            observation = sanitize_observation(observation)
            episode_query_string += observation

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

    ### Update reward tracker
    reward_tracker.set_reward(reward)

    ### Last thing - strip last space (if it exists)
    ### Return info
    return episode_query_string.strip(), episode_response_string.strip()

def main(args : argparse.Namespace):
    ### Instantiate agent
    agent = AgentFactory.create(args.model, args.llama_version)

    ### Register a text-based game as a new Gym's environment.
    game_env_id = textworld.gym.register_game(args.game, max_episode_steps=32)
    game_env = gym.make(game_env_id)

    ### Create PPO Config
    ppo_config_args = {
        "batch_size" : 1,
        "learning_rate" : 1e-5,
    }

    ### Generation kwargs
    generation_kwargs = {"do_sample" : True, "top_p" : 0.95, "top_k" : 32, "temperature" : 0.50, "max_new_tokens" : 32}

    ### Config
    ppo_config = PPOConfig(**ppo_config_args)

    ### Instantiate a reward trakcer
    reward_tracker = EpisodicRewardTracker()

    ### Create PPO Trainer
    # ppo_trainer = PPOTrainer(
    #     config=ppo_config,
    #     model=agent.model,
    #     ref_model=None,
    #     tokenizer=agent.tokenizer,
    #     reward_model=reward_tracker,
        
    # )

    episode_query_string, episode_response_string = run_episode(agent, game_env, reward_tracker)

    print("Reward: {}\nEpisode Query: {}\nEpisode Response: {}".format(reward_tracker.reward, episode_query_string, episode_response_string))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--game", required=True)
    parser.add_argument("--llama-version", type=str, default="13B")
    args = parser.parse_args()
    main(args)