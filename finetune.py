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
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed, TextEnvironment

### Local Imports
from agent import AgentFactory
from agent.agent import Agent
from utils import sanitize_observation, sanitize_response

### Helper function for tokenizing observations
### May need to change / modify
def tokenize_observation(agent : Agent, observation : str) -> Any:
    tokens = agent.tokenizer(observation, return_tensors="pt").input_ids.cuda()
    return tokens

###
### Run episode w/ 
###
def run_episode(agent : Agent, environment, ppo_trainer : PPOTrainer, generation_kwargs : Dict) -> Tuple:
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
    observation = sanitize_observation(observation)

    ### Record as string and tensor
    episode_query_string = observation
    episode_query_tensor_list.append( tokenize_observation(agent, observation) )

    ### Hardcoded
    pattern = r"<CMD>(.*?)<\/CMD>"

    ### Data to track
    score, moves, done = 0, 0, False

    ### Take the first action
    generated_response_output, response = agent.act(observation, generation_kwargs, ppo_trainer)

    while True:
        ### Increment moves!
        moves += 1
    
        ### Get command from agent outputs
        response = response.replace("</s>", "")
        response = re.search(pattern, response).group(1)
        response = sanitize_response(response)

        ### Append to response list
        episode_response_string += response
        episode_response_tensor_list.append( generated_response_output )

        ### Let the environment act and sanitize the output
        observation, score, done, info = environment.step(response)

        ### Check this so we don't append the final observation
        if not done:
            observation = sanitize_observation(observation)
            episode_query_string += observation
            episode_query_tensor_list.append( tokenize_observation(agent, observation) )

        ### Act
        generated_response_output, response = agent.act(observation, generation_kwargs, ppo_trainer )
        
        ### Check confusion after agent acts, exit early potentially
        if agent.is_confused or done:
            break

        ### Render
        ### environment.render()
    
    ### Compute 'win' from game output
    win = "You lost" not in observation and not agent.is_confused and moves < args.max_episode_steps

    ### Compute reward based on win and score
    if win:
        reward = score
    else:
        reward = score - 10.0

    ### Last thing - strip last space (if it exists) for printing
    episode_query_string = episode_query_string.strip()
    episode_response_string = episode_response_string.strip()

    print("Win: {}\nReward: {}\nEpisode Query: {}\nEpisode Response: {}".format(win, reward, episode_query_string, episode_response_string))

    ### Concat
    episode_query_concat_tensor = torch.cat(episode_query_tensor_list, dim=-1).squeeze()
    episode_response_summary_tensor = episode_response_tensor_list[-1].squeeze()

    ### Pad manually with eos tokens

    # for q, r in zip(episode_query_tensor_list, episode_response_tensor_list):
    #     print(f"q shape: {q.shape}")
    #     print(f"r shape: {r.shape}")
    #     print("+++++++++++++++++++++++")

    # print(f"query concat tensor shape: {episode_query_concat_tensor.shape}")
    # print(f"response summary tensor shape: {episode_response_summary_tensor.shape}")

    ### Return info
    return [ episode_query_concat_tensor ], [ episode_response_summary_tensor ], [torch.tensor(data=[reward], dtype=torch.float32, device=episode_response_summary_tensor.device)]

def main(args : argparse.Namespace):
    ### Instantiate agent
    agent = AgentFactory.create(args.model, args.llama_version)

    ### Register a text-based game as a new Gym's environment.
    game_env_id = textworld.gym.register_game(args.game, max_episode_steps=args.max_episode_steps)
    game_env = gym.make(game_env_id)

    ### Generation kwargs
    ### Use model defaults, Gets stuck?
    #generation_kwargs = {}

    ### Does not work
    generation_kwargs = {"do_sample" : True, "top_k" : 0.0, "top_p" : 1.0, "max_new_tokens" : 32, "pad_token_id" : agent.tokenizer.eos_token_id}

    ### "Works", but get NaN
    #generation_kwargs = {"do_sample" : True, "top_k" : 16, "top_p" : 0.50, "max_new_tokens" : 32, "pad_token_id" : agent.tokenizer.pad_token_id, "temperature" : 1.0, "remove_invalid_values" : True}

    ### "Works", but get NaN
    #generation_kwargs = {"do_sample" : False, "max_new_tokens" : 32, "pad_token_id" : agent.tokenizer.pad_token_id}

    ### Config
    ppo_config = PPOConfig(
        model_name=agent.model_name,
        log_with=None,
        learning_rate=1e-5,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        optimize_device_cache=True,
        seed=0,
        use_score_scaling=False,
        use_score_norm=False,
        score_clip=None,
        max_grad_norm=1.0,
    )

    ### Create PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=agent.model,
        ref_model=None,
        tokenizer=agent.tokenizer,
    )

    ### Get the episode query and response strings
    query_tensor, response_tensor, reward_tensor = run_episode(agent, game_env, ppo_trainer, generation_kwargs)

    ### Tokenize these, then pass these through the ppotrainer to update the model
    stats = ppo_trainer.step(query_tensor, response_tensor, reward_tensor)

    ### Stats gang
    print(f"Training stats: {stats}")
    
    ### Really Annoying. Stole this from _save_pretrained(...) of PPOTrainer
    ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).save_pretrained("models/")
    ppo_trainer.tokenizer.save_pretrained("models/")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--max-episode-steps", type=int, default=8)
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--game", required=True)
    parser.add_argument("--llama-version", type=str, default="13B")
    args = parser.parse_args()
    main(args)