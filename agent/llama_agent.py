import re
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, List, Tuple, Any

### Torch
import torch

### TRL
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed, TextEnvironment

### Local Imports
from .agent import Agent

class LlamaAgent(Agent):
    def __init__(self, version : str = "7B", model_postfix : str = "Chat-GPTQ") -> None:
        ### Validate
        if model_postfix not in ['Chat-AWQ', 'Chat-GPTQ']:
            raise ValueError("llama_agent.py: model_postfix is invalid")
    
        super().__init__()
        model_name_or_path = f"TheBloke/Llama-2-{version}-{model_postfix}"

        ### Save model name
        self.model_name = model_name_or_path

        print(f"llama_agent.py: Instantiating model: {model_name_or_path}")

        ### NOTE: Cannot use on quant models already
        # lora_config = LoraConfig(
        #     r=16,
        #     lora_alpha=32,
        #     lora_dropout=0.05,
        #     bias="none",
        #     task_type="CAUSAL_LM",
        # )
        
        ### NOTE: Cannot quant GPTQ model
        # nf4_config = BitsAndBytesConfig(
        #     load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
        # )

        if model_postfix == "Chat-AWQ":
            self.model = AutoAWQForCausalLM.from_quantized(
                model_name_or_path,
                fuse_layers=True,
                trust_remote_code=False,
            )
        elif model_postfix == "Chat-GPTQ":
            self.model = AutoModelForCausalLM.from_pretrained(
                ### fromPretrained Args
                model_name_or_path,
                device_map="auto",
                trust_remote_code=False,
                revision="main",
            )

        ###
        ### NOTE: This needs to not be done if we are loading a pre-trained model
        ### Required for training: Add a value head to the model as well
        ###
        self.model = AutoModelForCausalLMWithValueHead(
            pretrained_model=self.model,
            v_head_init_strategy="normal",
            v_head_initializer_range=0.2,
            summary_dropout_prob=None
        )

        print(f"Model v_head: {self.model.v_head}")

        ### NOTE: Note sure if this is proper
        self.model.is_peft_model = False if not hasattr(self.model, "is_peft_model") else self.model.is_peft_model

        ###
        ### Get tokenizer
        ###
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=False
        )

        ### MANUALLY SET PAD TOKEN
        ### NOTE: I am not sure if this is proper
        self.tokenizer.pad_token = self.tokenizer.eos_token

        ### This stuff is fine. Should be left alone here
        self.tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        self.pattern = r"(<CMD>.*?<\/CMD>)(</s>)*"

        # TODO Update this have objective info from the environment, etc.
        # Major parts of this need to be created from the environment.
        self.chat = None
        self.reset_chat()

    def reset_chat(self):
        self.chat = [
            {
                "role": "user",
                "content": "You are playing TextWorld. I will describe the environment. You must issue commands to play the game based on my guidance. Commands are of the form <CMD> [insert command] </CMD>. If you see or notice an object, try picking it up. Otherwise, search rooms and open doors to find an object.",
            },
            {
                "role": "assistant",
                "content": "I am playing TextWorld. I will issue commands based upon the environment that you describe, and I will describe my action in one sentence. Can you provide the objective?",
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
            generation_output = self.model.generate(tokens, **generate_kwargs)
        ### Use PPOTrainer
        else:
            generation_output = ppo_trainer.generate(tokens.squeeze(), generate_ref_response=False, return_prompt=False, **generate_kwargs)

        decoded_outputs = self._detokenize(generation_output, input_length)
        return generation_output, decoded_outputs