import re
from abc import ABC, abstractmethod
from pprint import pprint

import torch
from awq import AutoAWQForCausalLM
from optimum.bettertransformer import BetterTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead


class Agent(ABC):
    def __init__(
        self,
    ) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        self._chat = []
        self.pattern = r"(<CMD>.*?<\/CMD>)(</s>)*"

    @abstractmethod
    def _tokenize(self, obs):
        pass

    @abstractmethod
    def _detokenize(self, generation_output, input_length):
        pass

    @abstractmethod
    def initial_action(self):
        pass

    @abstractmethod
    def act(self, obs):
        pass


class LlamaAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def _tokenize(self, obs):
        return super()._tokenize(obs)

    def _detokenize(self, generation_output, input_length):
        return super()._detokenize(generation_output, input_length)

    def initial_action(self):
        return super().initial_action()

    def act(self, obs):
        return super().act(obs)


class MistralAgent:
    def __init__(self) -> None:
        # AWQ models are 4-bit quantized and really bloody fast.
        model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"

        self.model = AutoAWQForCausalLM.from_quantized(
            model_name_or_path,
            fuse_layers=True,
            trust_remote_code=False,
            safetensors=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=False
        )
        self.tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

        self.pattern = r"(<CMD>.*?<\/CMD>)(</s>)*"

        # TODO Update this have objective info from the environment, etc.
        # Major parts of this need to be created from the environment.
        self.chat = [
            {
                "role": "user",
                # "content": "You are playing TextWorld. To play TextWorld I will describe the evironment to you and you will issue commands. The TextWorld's text parser is limited, keep your answers to one command and 1-3 words. If you don't know what to do type 'help'.",
                "content": "You are playing TextWorld. I will describe the environment. You must issue commands. Commands are of the form <CMD> [insert command] </CMD>",
            },
            {
                "role": "assistant",
                "content": "I am playing TextWorld. I will issue commands based upon the environment that you describe that are 1-3 words long. Can you provide the objective?",
            },
            # # TODO Programmatically generate this.
            # {
            #     "role": "user",
            #     "content": """
            #     Your objective is to lift the stick of butter from the floor of the restroom. You have the following commands available to you:
            #         look:                describe the current room
            #         goal:                print the goal of this game
            #         inventory:           print player's inventory
            #         go <dir>:            move the player north, east, south or west
            #         examine ...:         examine something more closely
            #         eat ...:             eat edible food
            #         open ...:            open a door or a container
            #         close ...:           close a door or a container
            #         drop ...:            drop an object on the floor
            #         take ...:            take an object that is on the floor
            #         put ... on ...:      place an object on a supporter
            #         take ... from ...:   take an object from a container or a supporter
            #         insert ... into ...: place an object into a container
            #         lock ... with ...:   lock a door or a container with a key

                    
            #     You will only return one of these commands.
            #     """,
            # },
            # {
            #     "role": "assistant",
            #     "content": "My objective is to lift the stick of butter from the restroom. Describe my environment.",
            # },
            # # TODO Programmatic initial state
            # {
            #     "role": "user",
            #     "content": """-= Restroom =-
            #     You are in a restroom. It seems to be pretty typical here.
            #     There is an unblocked exit to the north. You don't like doors? Why not try going west, that entranceway is unguarded.
            #     There is a stick of butter on the floor.

            #     What command will you take? Tell me in: <CMD>[YOUR COMMAND]</CMD>.""",
            # },
        ]

    def _tokenize(self, obs):
        self.chat.append({"role": "user", "content": obs})
        tokens = self.tokenizer.apply_chat_template(
            self.chat, add_generation_prompt=True, tokenize=False
        )
        tokens = tokens + "<CMD>"
        print("\n\n{}\n\n".format(tokens))
        tokens = self.tokenizer(tokens, return_tensors="pt").input_ids.cuda()

        return tokens

    def _detokenize(self, generation_output, input_length):
        decoded_outputs = self.tokenizer.batch_decode(
            generation_output[:, input_length:], skip_special_tokens=True
        )[0]
        decoded_outputs = "<CMD>" + decoded_outputs
        decoded_outputs = re.search(self.pattern, decoded_outputs)
        decoded_outputs = decoded_outputs.group(1)
        self.chat.append({"role": "assistant", "content": decoded_outputs})
        return decoded_outputs

    def initial_action(self, objective, initial_state):
        tokens = self.tokenizer.apply_chat_template(
            self.chat, add_generation_prompt=True, tokenize=False
        )
        print(tokens)
        tokens = tokens + "<CMD>"
        tokens = self.tokenizer(tokens, return_tensors="pt").input_ids.cuda()
        input_length = tokens.shape[1]
        # Generate output
        generation_output = self.model.generate(
            tokens,
            do_sample=True,
            temperature=0.5,
            top_p=0.95,
            top_k=40,
            max_new_tokens=20,
        )

        decoded_outputs = self._detokenize(generation_output, input_length)
        return decoded_outputs

    def act(self, obs):
        tokens = self._tokenize(obs)
        input_length = tokens.shape[1]
        # Generate output
        generation_output = self.model.generate(
            tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=40,
            max_new_tokens=20,
        )
        decoded_outputs = self._detokenize(generation_output, input_length)
        return decoded_outputs

class AgentFactory:
    def create(model_name) -> Agent:
        if "mistral" in model_name:
            return MistralAgent()
        elif "llama" in model_name:
            return LlamaAgent()


if __name__ == "__main__":
    agent = AgentFactory.create('mistral')
    while True:
        obs = input("")
        action = agent.act(obs)
        print(action)