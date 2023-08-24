from abc import ABC, abstractmethod
from typing import TypedDict, Dict
from jarvis.llm.modules.module_registry import ModuleRegistry

class FunctionParameter(TypedDict):
    type: str
    description: str

class LLMModule(ABC):
    def __init__(self, name: str, description: str, parameters: Dict[str, FunctionParameter]) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        ModuleRegistry.get_instance().register(self)

    @abstractmethod
    def activate(self, arguments: dict) -> str:
        pass

    @abstractmethod
    def get_preprompts(self) -> list[dict]:
        """
        Returns an array of example prompts for the model
        """
        return []
