### Local Imports
from .agent import Agent
from .llama_agent import LlamaAgent
from .mistral_agent import MistralAgent

class AgentFactory:
    def create(model_name : str, llama_version : str) -> Agent:
        if "mistral" in model_name:
            return MistralAgent()
        elif "llama" in model_name:
            return LlamaAgent(llama_version)