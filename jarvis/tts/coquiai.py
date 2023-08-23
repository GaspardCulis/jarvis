# Partially and shamelessly copied from https://gist.github.com/reuben/5f483772864c1ac9687ba1c7aa3cb11f

from typing import Iterator
from jarvis.tts.tts_provider import TTSProvider
import subprocess
import requests
import shutil
import os

class CoquiAI(TTSProvider):
    API_URL = "https://app.coqui.ai/api/v2/samples/multilingual/render/?format=wav"

    def __init__(self, voice_id: str, lang: str = "fr") -> None:
        self.token = os.getenv("COQUI_API_TOKEN")
        self.voice_id = voice_id
        self.lang = lang

    def stream(self, audio_stream: Iterator[bytes]) -> bytes:
        if not shutil.which("mpv"):
            raise ValueError("mpv not found")

        mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
        mpv_process = subprocess.Popen(
            mpv_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        audio = b""

        for chunk in audio_stream:
            if chunk is not None:
                mpv_process.stdin.write(chunk)  # type: ignore
                mpv_process.stdin.flush()  # type: ignore
                audio += chunk

        if mpv_process.stdin:
            mpv_process.stdin.close()
        mpv_process.wait()

        return audio

    def tts(self, message: str) -> Iterator[bytes]:
        res = requests.post(
            self.API_URL,
            json={"text": message, "voice_id": self.voice_id, "language": self.lang},
            headers={"Authorization": f"Bearer {self.token}"},
        )

        if res.status_code != 200:
            raise ValueError(f"CoquiAI API returned {res.status_code} status code, error: {res.text}")

        for chunk in res.iter_content(chunk_size=2048):
            if chunk:
                yield chunk

    def speak(self, message: str) -> None:
        audio_stream = self.tts(message)
        self.stream(audio_stream)