from abc import ABC, abstractmethod
from typing import Tuple, Dict
from jarvis.llm.modules.module_registry import ModuleRegistry


class LLMModule(ABC):
    def __init__(self, name: str, description: str, parameters: Dict[str, Tuple[str, str]]) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        ModuleRegistry.get_instance().register(self)

    @abstractmethod
    def activate(self, message: str) -> str:
        pass

    @abstractmethod
    def get_preprompts(self) -> list[dict]:
        """
        Returns an array of example prompts for the model
        """
        return []
