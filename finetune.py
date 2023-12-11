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
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
    set_seed,
    TextEnvironment,
)

### Local Imports
from agent import AgentFactory
from agent.agent import Agent
from utils import sanitize_observation, sanitize_response

import warnings
from pprint import pprint


### Helper function for tokenizing observations
### May need to change / modify
def tokenize_observation(agent: Agent, observation: str) -> Any:
    tokens = (
        agent.tokenizer(observation, return_tensors="pt").input_ids.cuda().squeeze()
    )
    return tokens


###
### Run episode w/
###
def run_episode(
    agent: Agent, environment, ppo_trainer: PPOTrainer, generation_kwargs: Dict
) -> Dict:
    ### Reset agent
    agent.reset_chat()

    ### Batch Data
    ### Emulates dataloader behavior of TRL examples
    batch = {
        "input_ids": [],
        "responses_generated": [],
        "observations": [],
        "responses_decoded": [],
        "reward": [],
    }

    ### Reset environment
    observation, info = environment.reset()

    ### Hardcoded
    pattern = r"<CMD>(.*?)<\/CMD>"

    ### Data to track
    score, moves, done = 0, 0, False

    while True:
        ### Increment moves!
        moves += 1

        ### Check this so we don't append the final observation
        if not done:
            observation = sanitize_observation(observation)
            tokenized_observation = tokenize_observation(agent, observation)
            batch['observations'].append(observation)
            batch["input_ids"].append(tokenized_observation)

        ### Get our next response
        generated_response_output, response, tokenized_observation = agent.act(
            observation, generation_kwargs, ppo_trainer
        )
        # batch["input_ids"].append(tokenized_observation)


        ### Get command from agent outputs
        response = response.replace("</s>", "")
        response = re.search(pattern, response).group(1)
        response = sanitize_response(response)

        ### Append to response list
        batch["responses_generated"].append(generated_response_output)

        ### Let the environment act and sanitize the output
        observation, score, done, info = environment.step(response)

        ### Check confusion after agent acts, exit early potentially
        if done or agent.is_confused:
            break

    ### Compute 'win' from game output
    win = (
        "You lost" not in observation
        and not agent.is_confused
        and moves < args.max_episode_steps
    )

    print(f"Win: {win}\nScore: {score}\nMoves: {moves}\nConfused: {agent.is_confused}")

    ### Get decoded responses
    batch["responses_decoded"] = [
        agent.tokenizer.decode(r.squeeze()) for r in batch["responses_generated"]
    ]


    ### Now, consolidate batch info
    ## In particular, we want to consolidate responses_generated, and input_ids
    batch["responses_generated"] = [
        torch.cat(batch["responses_generated"], dim=-1).squeeze()
    ]
    batch["input_ids"] = [torch.cat(batch["input_ids"], dim=-1)]
    batch["reward"] = (
        [torch.tensor(data=[1.0], dtype=torch.float32, device="cuda")]
        if win
        else [torch.tensor(data=[-1.0], dtype=torch.float32, device="cuda")]
    )

    ### Return batch info
    return batch

def main(args: argparse.Namespace):
    ### Instantiate agent
    agent = AgentFactory.create(args.model, args.llama_version)

    ### Register a text-based game as a new Gym's environment.
    game_env_id = textworld.gym.register_game(
        args.game,
        max_episode_steps=args.max_episode_steps,
        # batch_size=8,
        # asynchronous=False,
    )
    game_env = gym.make(game_env_id)

    generation_kwargs = {
        "min_length": -1, # don't ignore the EOS token
        "top_k": 0.0, # no top-k sampling
        "top_p": 1.0, # no nucleus sampling
        "do_sample": True, # yes, we want to sample
        "pad_token_id": agent.tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
        "max_new_tokens": 32, # specify how many tokens you want to generate at most
    }

    ### Config
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=1,
        mini_batch_size=1,
        kl_penalty="abs",
        init_kl_coef=0.02,
    )

    ### Create PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=agent.model,
        ref_model=None,
        num_shared_layers=None,
        tokenizer=agent.tokenizer,
    )

    ### Epochs
    for e in range(100):
        batch = {
            "input_ids": [],
            "responses_decoded": [],
            "responses_generated": [],
            "reward": [],
        }

        ### Iteratively generate batches
        for b in range(ppo_config.batch_size):
            episode_batch = run_episode(agent, game_env, ppo_trainer, generation_kwargs)
            batch["input_ids"] += episode_batch["input_ids"]
            batch["responses_decoded"] += episode_batch["responses_decoded"]
            batch["responses_generated"] += episode_batch["responses_generated"]
            batch["reward"] += episode_batch["reward"]

        ### Tokenize these, then pass these through the ppotrainer to update the model
        pprint("running backward")
        print(batch["responses_decoded"])
        stats = ppo_trainer.step(
            batch["input_ids"], batch["responses_generated"], batch["reward"]
        )

        ### Stats gang
        print(f"Training stats mean reward: {stats['ppo/returns/mean']}\nKL Loss: {stats['ppo/mean_non_score_reward']}")

    ### Really Annoying. Stole this from _save_pretrained(...) of PPOTrainer
    # ppo_trainer.accelerator.unwrap_model(ppo_trainer.model).save_pretrained("models/")
    # ppo_trainer.tokenizer.save_pretrained("models/")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--max-episode-steps", type=int, default=8)
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--game", required=True)
    parser.add_argument("--llama-version", type=str, default="7B")
    args = parser.parse_args()
    main(args)