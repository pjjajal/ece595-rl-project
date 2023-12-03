import re
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

### Local Imports
from .agent import Agent

class LlamaAgent(Agent):
    def __init__(self, version) -> None:
        super().__init__()
        model_name_or_path = f"TheBloke/Llama-2-{version}-Chat-AWQ"

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
                "content": "You are playing TextWorld. I will describe the environment. You must issue commands. Commands are of the form <CMD> [insert command] </CMD>",
            },
            {
                "role": "assistant",
                "content": "I am playing TextWorld. I will issue commands based upon the environment that you describe that are 1-3 words long. Can you provide the objective?",
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