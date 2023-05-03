from abc import ABC, abstractmethod
from jarvis.llm.modules.module_registry import ModuleRegistry


class LLMModule(ABC):
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        ModuleRegistry.get_instance().register(self)

    def should_activate(self, message: str) -> bool:
        return message.startswith(f"[{self.prefix}]")

    @abstractmethod
    def activate(self, message: str) -> str:
        pass
