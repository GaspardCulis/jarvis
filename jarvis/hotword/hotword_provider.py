from abc import ABC, abstractmethod

class HotwordProvider(ABC):
    @abstractmethod
    def wait(self) -> None:
        pass
