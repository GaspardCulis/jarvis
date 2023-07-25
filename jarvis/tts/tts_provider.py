from abc import ABC, abstractmethod
from typing import Dict, Tuple

class TTSProvider(ABC):
    @abstractmethod
    def speak(self, message: str):
        pass
