import re
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

### Local Imports
from .agent import Agent

class MistralAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
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
        #print("\n\n{}\n\n".format(tokens))
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
            do_sample=False,
            #temperature=0.5,
            #temperature=0.0,
            #top_p=0.95,
            #top_k=40,
            max_new_tokens=32,
        )

        decoded_outputs = self._detokenize(generation_output, input_length)
        return decoded_outputs

    def act(self, obs):
        tokens = self._tokenize(obs)
        input_length = tokens.shape[1]
        # Generate output
        generation_output = self.model.generate(
            tokens,
            do_sample=False,
            #temperature=0.6,
            #temperature=0.0,
            #top_p=0.95,
            #top_k=40,
            max_new_tokens=32,
        )
        decoded_outputs = self._detokenize(generation_output, input_length)
        return decoded_outputs