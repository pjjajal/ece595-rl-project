from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(
        self,
    ) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        self._chat = []
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