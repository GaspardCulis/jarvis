import json
from dotenv import load_dotenv  # noqa
load_dotenv()  # noqa

from jarvis.llm.modules.module_registry import ModuleRegistry
from jarvis.llm.modules.terminal import TerminalModule
from jarvis.llm.modules.music import MusicSearch, MusicPlay
from jarvis.llm.gpt_turbo import LLM
from jarvis.tts.elevenlabs import ElevenLabs
from jarvis.stt.whisper import Whisper

import pvporcupine
from pvrecorder import PvRecorder
import numpy as np
import struct
import os
from random import choice

term = TerminalModule()
music_search = MusicSearch()
music_play = MusicPlay()

llm = LLM()
llm.message_history += ModuleRegistry.get_instance().get_preprompts()
# print(json.dumps(llm.message_history, indent=2))

tts = ElevenLabs()

stt = Whisper()

porcupine = pvporcupine.create(
    access_key = os.getenv("PORCUPINE_API_KEY"),
    model_path = os.getenv("PORCUPINE_MODEL_PATH"),
    keyword_paths = [os.getenv("PORCUPINE_PPN_PATH") or ""]
)

hotword_responses = ["Oui ?", "Qu'y a-t-il ?", "Que puis-je faire pour vous ?", "Comment puis-je vous aider ?"]

recorder = PvRecorder(
    device_index=-1,
    frame_length=porcupine.frame_length
)

response = {}
while True:
    if response.get("function_call"):
        output = ModuleRegistry.get_instance().gpt_function_call(response["function_call"])
        print("Module output: ", output)
        if output:
            response = llm.prompt({
                "role": "function",
                "name": response["function_call"]["name"],
                "content": output
            })
        else:
            response = {}
        continue
    
    if response.get("content"):
        print(response["content"])
        try:
            tts.speak(response["content"])
        except KeyboardInterrupt:
            print("Skipping tts")
    else:
        # Hotword detection
        print("Waiting for hotword")
        recorder.start()
        while True:
            pcm = recorder.read()
            result = porcupine.process(pcm)
            if result >= 0:
                print("Hotword detected")
                break
        recorder.stop()
        tts.speak(choice(hotword_responses))
    # Listen audio prompt
    message = stt.listen()
    if not message:
        print("Nothing was said")
        response = {}
        continue

    print("Transcribed audio = ", message)
    response = llm.prompt(message)
            
