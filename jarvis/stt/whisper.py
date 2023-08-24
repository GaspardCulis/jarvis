from jarvis.stt.stt_provider import STTProvider
import wave
import struct
import whisper
import numpy as np
from pvrecorder import PvRecorder

class WhisperConfig:
    def __init__(
            self, 
            prompt_audio_path = "/tmp/jarvis_prompt.wav", 
            audio_device_index = -1,
            device: str | None = None,
            max_silent_frames = 30,
            min_volume = 300,
            max_silent_frames_ratio = 0.9
        ) -> None:
        self.prompt_audio_path = prompt_audio_path
        self.audio_device_index = audio_device_index
        self.device = device
        self.max_silent_frames = max_silent_frames
        self.min_volume = min_volume
        self.max_silent_frames_ratio = max_silent_frames_ratio


class Whisper(STTProvider):
    def __init__(self, config = WhisperConfig()) -> None:
        super().__init__()
        self.C = config
        self.model = whisper.load_model("medium", device=self.C.device)
        self.recorder = PvRecorder(
            device_index=self.C.audio_device_index,
            frame_length=512
        )


    def listen(self) -> str | None:
        self.recorder.start()
        audio = []
        frames_count = 0
        silent_frames_count = 0
        temp_silent_frames_count = 0
        while True:
            frames_count += 1
            frame = self.recorder.read()
            audio.extend(frame)
            max_frame_vol = np.max(np.abs(frame))
            if max_frame_vol < self.C.min_volume:
                silent_frames_count += 1
                temp_silent_frames_count += 1
                if temp_silent_frames_count > self.C.max_silent_frames:
                    break
            else:
                temp_silent_frames_count = 0

        self.recorder.stop()
        frames_ratio = silent_frames_count / frames_count
        if frames_ratio > self.C.max_silent_frames_ratio: # Nothing was said
            return None
        # Save audio 
        with wave.open(self.C.prompt_audio_path, "w") as f:
            f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
            f.writeframes(struct.pack("h" * len(audio), *audio))
        # Decode using whisper
        return self.model.transcribe(self.prompt_audio_path)["text"] # type: ignore