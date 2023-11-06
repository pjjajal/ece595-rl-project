import torch
from awq import AutoAWQForCausalLM
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
import re



class Agent:
    def __init__(self) -> None:
        # AWQ models are 4-bit quantized and really bloody fast.
        model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"
        # model_name_or_path = "TheBloke/Mistral-7B-v0.1-AWQ"
        # model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.1"

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

        # model = AutoModelForCausalLM.from_pretrained(
        #     "mistralai/Mistral-7B-Instruct-v0.1",
        #     use_flash_attention_2=True,
        #     device_map="auto",
        #     load_in_8bit=True
        # )
        # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

        # TODO Update this have objective info from the environment, etc.
        # Major parts of this need to be created from the environment.
        self.chat = [
            {
                "role": "user",
                "content": "You are playing TextWorld. To play TextWorld I will describe the evironment to you and you will issue commands. The TextWorld's text parser is limited, keep your answers to one command and 1-3 words. If you don't know what to do type 'help'.",
            },
            {
                "role": "assistant",
                "content": "I am playing TextWorld. I will issue commands based upon the environment that you describe that are 1-3 words long. Can you provide the objective?",
            },
            # TODO Programmatically generate this.
            {
                "role": "user",
                "content": """
                Your objective is to lift the stick of butter from the floor of the restroom.

                You have the following commands available to you:
                    look:                describe the current room
                    goal:                print the goal of this game
                    inventory:           print player's inventory
                    go <dir>:            move the player north, east, south or west
                    examine ...:         examine something more closely
                    eat ...:             eat edible food
                    open ...:            open a door or a container
                    close ...:           close a door or a container
                    drop ...:            drop an object on the floor
                    take ...:            take an object that is on the floor
                    put ... on ...:      place an object on a supporter
                    take ... from ...:   take an object from a container or a supporter
                    insert ... into ...: place an object into a container
                    lock ... with ...:   lock a door or a container with a key

                    
                You will only return one of these commands.
                """,
            },
            {
                "role": "assistant",
                "content": "My objective is to lift the stick of butter from the restroom. Describe my environment.",
            },
            # TODO Programmatic initial state
            {
                "role": "user",
                "content": """-= Restroom =-
                You are in a restroom. It seems to be pretty typical here.
                There is an unblocked exit to the north. You don't like doors? Why not try going west, that entranceway is unguarded.
                There is a stick of butter on the floor.

                What command will you take? Tell me in: <CMD>[YOUR COMMAND]</CMD>.""",
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
        decoded_outputs = re.search(self.pattern, decoded_outputs).group(1)
        self.chat.append({"role": "assistant", "content": decoded_outputs})
        return decoded_outputs

    def initial_action(self):
        tokens = self.tokenizer.apply_chat_template(
            self.chat, add_generation_prompt=True, tokenize=False
        )
        tokens = tokens + "<CMD>"
        tokens = self.tokenizer(tokens, return_tensors="pt").input_ids.cuda()
        input_length = tokens.shape[1]
        # Generate output
        generation_output = self.model.generate(
            tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            max_new_tokens=20,
        )

        decoded_outputs =  self._detokenize(generation_output, input_length)
        # decoded_outputs = self.tokenizer.batch_decode(
        #     generation_output[:, input_length:], skip_special_tokens=True
        # )[0]
        # decoded_outputs = "<CMD>" + decoded_outputs
        # decoded_outputs = re.search(self.pattern, decoded_outputs).group(1)
        # self.chat.append({"role": "assistant", "content": decoded_outputs})
        return decoded_outputs

    def act(self, obs):
        tokens = self._tokenize(obs)
        input_length = tokens.shape[1]
        # Generate output
        generation_output = self.model.generate(
            tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            max_new_tokens=20,
        )
        # decoded_outputs = self.tokenizer.batch_decode(
        #     generation_output[:, input_length:], skip_special_tokens=True
        # )[0]
        # decoded_outputs = "<CMD>" + decoded_outputs
        
        # self.chat.append({"role": "assistant", "content": decoded_outputs})
        # # print(decoded_outputs)
        # print(self.chat)
        decoded_outputs = self._detokenize(generation_output, input_length)
        return decoded_outputs


if __name__ == "__main__":
    agent = Agent()

    # _ = agent.act(x)
    # print(_)
    while True:
        obs = input("")
        action = agent.act(obs)
