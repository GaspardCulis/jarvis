from dotenv import load_dotenv  # noqa
load_dotenv()  # noqa

from jarvis.llm.modules.module_registry import ModuleRegistry
from jarvis.llm.modules.terminal import TerminalModule
from jarvis.llm.gpt_turbo import LLM
from jarvis.tts.elevenlabs import ElevenLabs

import pvporcupine
from pvrecorder import PvRecorder
import wave
import numpy as np
import struct
import whisper
import os
from random import choice

term = TerminalModule()

llm = LLM()
tts = ElevenLabs()
stt = whisper.load_model("medium")
porcupine = pvporcupine.create(
    access_key = os.getenv("PORCUPINE_API_KEY"),
    model_path = os.getenv("PORCUPINE_MODEL_PATH"),
    keyword_paths = [os.getenv("PORCUPINE_PPN_PATH")]
)
hotword_responses = ["Oui ?", "Qu'y a-t-il ?", "Que puis-je faire pour vous ?", "Comment puis-je vous aider ?"]

recorder = PvRecorder(
    device_index=-1,
    frame_length=porcupine.frame_length)

prompt_audio_path = "/tmp/jarvis_prompt.wav"

llm.message_history += ModuleRegistry.get_instance().get_preprompts()

response = {}
while True:
    if response.get("function_call"):
        output = ModuleRegistry.get_instance().gpt_function_call(response["function_call"])
        print("Module output: ", output)
        response = llm.prompt({
            "role": "function",
            "name": response["function_call"]["name"],
            "content": output
        })
        continue
    
    if response.get("content"):
        print(response["content"])
        tts.speak(response["content"])

    # Hotword detection
    recorder.start()
    while True:
        pcm = recorder.read()
        result = porcupine.process(pcm)
        if result >= 0:
            print("Hotword detected")
            break
    recorder.stop()
    tts.speak(choice(hotword_responses))
    recorder.start()
    # Listen audio prompt
    audio = []
    silent_frames_count = 0
    while True:
        frame = recorder.read()
        audio.extend(frame)
        max_frame_vol = np.max(np.abs(frame))
        print(max_frame_vol)
        if max_frame_vol < 300:
            silent_frames_count += 1
            if silent_frames_count > 30:
                break
        else:
            silent_frames_count = 0
    # Save audio 
    recorder.stop()
    with wave.open(prompt_audio_path, "w") as f:
        f.setparams((1, 2, 16000, 512, "NONE", "NONE"))
        f.writeframes(struct.pack("h" * len(audio), *audio))
    # Decode using whisper
    result = stt.transcribe(prompt_audio_path)

    message = result["text"]
    print("Transcribed audio = ", message)
    response = llm.prompt(message)
            
