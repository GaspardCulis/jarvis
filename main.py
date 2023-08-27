from dotenv import load_dotenv  # noqa
load_dotenv()  # noqa

import os
from jarvis.llm.modules.module_registry import ModuleRegistry
from jarvis.llm.modules.terminal import TerminalModule
from jarvis.llm.modules.music import MusicSearch, MusicPlay
from jarvis.llm.contexts import JARVIS_CONTEXTS
from jarvis.llm.gpt_turbo import LLM
from jarvis.tts.elevenlabs import ElevenLabs
from jarvis.stt.whisper import Whisper
from jarvis.hotword.porcupine import Porcupine

term = TerminalModule()
music_search = MusicSearch()
music_play = MusicPlay()

llm = LLM(JARVIS_CONTEXTS[os.getenv("ASSISTANT_LANG") or "en"])
llm.message_history += ModuleRegistry.get_instance().get_preprompts()
# print(json.dumps(llm.message_history, indent=2))

tts = ElevenLabs()

stt = Whisper()

hotword = Porcupine(tts=tts)

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
        hotword.wait()

    # Listen audio prompt
    message = stt.listen()
    if not message:
        print("Nothing was said")
        response = {}
        continue

    print("Transcribed audio = ", message)
    response = llm.prompt(message)
            
