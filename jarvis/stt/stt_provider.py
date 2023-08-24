from abc import ABC, abstractmethod

class STTProvider(ABC):
    @abstractmethod
    def listen(self) -> str:
        """
        Listen to the microphone and return the audio as a string.
        """
        pass