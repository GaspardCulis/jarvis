import elevenlabs
from elevenlabs import generate, stream, set_api_key, play, api
from elevenlabs.api.error import RateLimitError, APIError
from elevenlabs_unleashed.manager import ELUAccountManager 
from jarvis.tts.tts_provider import TTSProvider

class ElevenLabs(TTSProvider):
    def __init__(self, voice = "Josh", model = "eleven_multilingual_v1"):
        self.voice = voice
        self.model = model
        self.eluac = ELUAccountManager(set_api_key)
        print("[ElevenLabs] Initializing the ElevenLabs TTS provider...")
        self.eluac.next()
        print("[ElevenLabs] Initialization done.")

    
    def speak(self, message: str):
        try:
            audio_stream = generate(
                text=message,
                voice=self.voice,
                model=self.model,
                stream=True
            )
        except (RateLimitError) as e:
            print("[ElevenLabs] Maximum number of requests reached. Getting a new API key...")
            self.eluac.next()
            self.speak(message)
            return

        print("[ElevenLabs] Starting the stream...")
        try:
            stream(audio_stream) # type: ignore
        except (APIError) as e:
            print("[ElevenLabs] Text is too long.")