from abc import ABC, abstractmethod

class TTSProvider(ABC):
    @abstractmethod
    def speak(self, message: str):
        pass
