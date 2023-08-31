from jarvis.hotword.hotword_provider import HotwordProvider
from jarvis.tts.tts_provider import TTSProvider
import os
import pvporcupine
from random import choice
from pvrecorder import PvRecorder


HOTWORD_RESPONSES = [
    "Oui ?", 
    "Qu'y a-t-il ?", 
    "Que puis-je faire pour vous ?", 
    "Comment puis-je vous aider ?"
]

class Porcupine(HotwordProvider):
    def __init__(self, tts: TTSProvider | None = None, hotword_responses: list[str] = HOTWORD_RESPONSES) -> None:
        super().__init__()
        self.tts = tts
        self.hotword_responses = hotword_responses
        self.porcupine = pvporcupine.create(
            access_key = os.getenv("PORCUPINE_API_KEY"),
            model_path = os.getenv("PORCUPINE_MODEL_PATH"),
            keyword_paths = [os.getenv("PORCUPINE_PPN_PATH") or ""]
        )
        self.recorder = PvRecorder(
            device_index=-1,
            frame_length=self.porcupine.frame_length
        )

    def wait(self) -> None:
        print("[Porcupine] Waiting for hotword...")
        self.recorder.start()
        while True:
            pcm = self.recorder.read()
            result = self.porcupine.process(pcm)
            if result >= 0:
                print("Hotword detected")
                break
        self.recorder.stop()
        if self.tts:
            self.tts.speak(choice(self.hotword_responses))