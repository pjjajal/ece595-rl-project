### Local Imports
from .agent import Agent

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
