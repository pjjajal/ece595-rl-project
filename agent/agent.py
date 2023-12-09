from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(
        self,
    ) -> None:
        self.model_name = None
        self.model = None
        self.tokenizer = None
        self.device = None
        self.chat = []
        self.pattern = r"(<CMD>.*?<\/CMD>)(</s>)*"
        self.is_confused = False

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

    @abstractmethod
    def reset_chat(self):
        pass