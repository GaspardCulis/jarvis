from elevenlabs import generate, stream, set_api_key, play
from jarvis.tts.tts_provider import TTSProvider
import os

class ElevenLabs(TTSProvider):
    def __init__(self, voice = "Josh", model = "eleven_multilingual_v1"):
        set_api_key(os.getenv("ELEVENLABS_API_KEY"))
        self.voice = voice
        self.model = model
    
    def speak(self, message: str):
        audio_stream = generate(
            text=message,
            voice=self.voice,
            model=self.model,
            stream=True
        )

        stream(audio_stream)