import re
import time
import argparse

### Gym / TextWorld stuff
import gym
import textworld.gym

### TRL
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed, TextEnvironment

### Local Imports
from agent import AgentFactory
from argparse import ArgumentParser

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

    ### Get the initial observation, we can use it as a prompt
    game_goal, _, _, _ = game_env.step("goal")

    ### Get a TRL text environment
    trl_text_env = TextEnvironment(
        model=agent.model,
        tokenizer=agent.tokenizer,
        reward_fn=reward_function,
        max_turns=32,
        generation_kwargs = {
            "do_sample" : "false",
            "max_new_tokens" : "32",
        }
    )

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
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--game", required=True)
    args = parser.parse_args()
    main(args)