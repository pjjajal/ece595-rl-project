import re
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer
from typing import Dict, Tuple

### Torch
import torch

### Local Imports
from .agent import Agent

class MistralAgent(Agent):
    def __init__(self, version : str = "7B") -> None:
        model_postfix : str = "Instruct-v0.1-GPTQ"

        ### Validate
        if model_postfix not in ["Instruct-v0.1-AWQ", "Instruct-v0.1-GPTQ", "Instruct-v0.1"]:
            raise ValueError("llama_agent.py: model_postfix is invalid")

        super().__init__()
        
        model_name_or_path = f"TheBloke/Mistral-{version}-{model_postfix}"
        #model_name_or_path = f"mistralai/Mistral-{version}-{model_postfix}"

        ### Save model name
        self.model_name = model_name_or_path

        print(f"mistral_agent.py: Instantiating model: {model_name_or_path}")

        ### Based on model postfix, load in a particular manner
        ### AWQ Model, load quantized
        if model_postfix == "Instruct-v0.1-AWQ":
            self.model = AutoAWQForCausalLM.from_quantized(
                model_name_or_path,
                fuse_layers=True,
                trust_remote_code=False,
                safetensors=True,
            )

        ### Can load with transformers / TRL interface (should be able to!)
        ### Huggingface Transformer model
        elif model_postfix == "Instruct-v0.1-GPTQ":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                trust_remote_code=False,
                revision="main"
            )

        elif model_postfix == "Instruct-v0.1":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                trust_remote_code=False,
                revision="main"
            )     
        
        ###
        ### NOTE: This needs to not be done if we are loading a pre-trained model
        ### Required for training: Add a value head to the model as well
        ###
        # self.model = AutoModelForCausalLMWithValueHead(
        #     pretrained_model=self.model,
        #     v_head_init_strategy="normal",
        #     v_head_initializer_range=0.2,
        #     summary_dropout_prob=None
        # )
        
        ### NOTE: Note sure if this is proper
        # self.model.is_peft_model = False if not hasattr(self.model, "is_peft_model") else self.model.is_peft_model

        ###
        ### Get tokenizer
        ###
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=False
        )

        ### MANUALLY SET PAD TOKEN
        ### NOTE: I am not sure if this is proper
        # self.tokenizer.pad_token = self.tokenizer.eos_token

        ### This stuff is fine. Should be left alone here
        self.tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        self.pattern = r"(<CMD>.*?<\/CMD>)(</s>)*"

        # TODO Update this have objective info from the environment, etc.
        # Major parts of this need to be created from the environment.
        self.chat = None
        self.reset_chat()

    def reset_chat(self):
        ###
        ### TextWorld Prompts
        ###
        v1_prompt = "You are playing TextWorld. I will describe the environment. You must issue commands to play the game based on my guidance. Commands are of the form <CMD> [insert command] </CMD>."
        v2_prompt = v1_prompt + " If you see or notice an object, try picking it up. Otherwise, search rooms and open doors to find an object."
        v3_prompt = v2_prompt + " You can only do the following actions in your command: look, go [north, south, west, east], open, close, take, drop, lock, unlock."
        v4_prompt = v3_prompt + " Follow the order of my commands if I give more than one."
        v5_prompt = v1_prompt + " Follow the order of my commands exactly if I give more than command. You can only use the following verbs in your command when playing TextWorld: look, go [north, west, south, east], search an object, open, close, take, drop, lock, unlock."
        
        v6_prompt = v4_prompt + " Say \"goal\" to remind yourself how to win the game if you are lost. If you do not follow commands of the goal in order, you will not win the game."
        v7_prompt = v1_prompt + " Follow the order of my commands if I give more than one. You can only do the following actions in your command: look, go [north, south, west, east], open, close, take, drop, lock, unlock."

        self.chat = [
            {
                "role": "user",
                "content": v7_prompt,
            },
            {
                "role": "assistant",
                #"content": "I am playing TextWorld. I will issue commands based on the environment that you describe, and I will describe my command with a short phrase. Can you provide the objective?",
                "content": "I am playing TextWorld. I will issue commands based on the environment that you describe, and I will describe my command with a short phrase. I will follow the tasks of the objective in the order you tell me. What is the objective?",
            },
        ]

    def _tokenize(self, obs):
        self.chat.append({"role": "user", "content": obs})
        tokens = self.tokenizer.apply_chat_template(
            self.chat, add_generation_prompt=True, tokenize=False
        )
        tokens = tokens + "<CMD>"
        
        tokens = self.tokenizer(tokens, return_tensors="pt").input_ids.cuda()

        return tokens

    def _detokenize(self, generation_output, input_length):
        decoded_outputs = self.tokenizer.batch_decode(
            generation_output[:, input_length:], skip_special_tokens=True
        )[0]
        decoded_outputs = "<CMD>" + decoded_outputs
        decoded_outputs = re.search(self.pattern, decoded_outputs)

        ### Assertion!
        ### Handling empty output
        if(decoded_outputs is None):
            ### Update confused flag
            self.is_confused = True

            ### Create an output that indicates this and try and continue - or just decide to stop early in run.py
            empty_decoded_output_none_assertion = "<CMD>I am confused</CMD>"
            decoded_outputs = re.search(self.pattern, empty_decoded_output_none_assertion)
            #print("Decoded outputs (immediate): {}".format(decoded_outputs))
        else:
            ### Clear confusion
            self.is_confused = False

        decoded_outputs = decoded_outputs.group(1)
        self.chat.append({"role": "assistant", "content": decoded_outputs})

        return decoded_outputs

    def act(self, obs : str, generate_kwargs : Dict, ppo_trainer : PPOTrainer = None) -> Tuple:
        tokens = self._tokenize(obs)
        input_length = tokens.shape[1]

        ### Generate output
        if ppo_trainer is None:
            generation_output = self.model.generate(input_ids=tokens, **generate_kwargs)

        ### Use PPOTrainer
        else:
            ### VERY IMPORTANT
            ### Truncate generated output to input_length to get the 'new' generated content only
            ### This makes it so the generation_output corresponds exactly to the decoded_outputs (see dtokeinze above for the slicing behavior of generation_output)
            generation_output = ppo_trainer.generate(tokens.squeeze(), **generate_kwargs)

        decoded_outputs = self._detokenize(generation_output, input_length)
        return generation_output[:, input_length:], decoded_outputs, tokens