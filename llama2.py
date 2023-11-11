import re
from pprint import pprint

import torch
from optimum.bettertransformer import BetterTransformer

# from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead


class Agent:
    def __init__(self) -> None:
        model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto", torch_dtype=torch.float16
        )
        self.device = self.model.device
        print(self.tokenizer.chat_template)
        self.chat = []
        # self.tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        tokens = self._tokenize("Hello World")
        generation_output = self.model.generate(
            tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
            top_k=40,
            max_new_tokens=20,
        )
        decoded_outputs = self.tokenizer.batch_decode(generation_output,)
        print(decoded_outputs)

    def _tokenize(self, obs):
        self.chat.append({"role": "user", "content": obs})
        tokens = self.tokenizer.apply_chat_template(
            self.chat, add_generation_prompt=True, tokenize=False
        )
        print(tokens)
        # tokens = tokens + "<CMD>"
        tokens = self.tokenizer(tokens, return_tensors="pt").input_ids.to(self.device)

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

    def initial_action(self):
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
            temperature=0.3,
            top_p=0.95,
            top_k=40,
            max_new_tokens=20,
        )

        decoded_outputs = self._detokenize(generation_output, input_length)
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
            temperature=0.3,
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
        # pprint(self.chat)
        return decoded_outputs


if __name__ == "__main__":
    agent = Agent()

    # _ = agent.act(x)
    # print(_)
    while True:
        obs = input("")
        action = agent.act(obs)
