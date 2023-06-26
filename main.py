from dotenv import load_dotenv  # noqa
load_dotenv()  # noqa

from jarvis.llm.modules.module_registry import ModuleRegistry
from jarvis.llm.modules.terminal import TerminalModule
from jarvis.llm.gpt_turbo import LLM
from jarvis.tts.elevenlabs import ElevenLabs

term = TerminalModule()

llm = LLM()
tts = ElevenLabs()

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
    else:
        if response.get("content"):
            print(response["content"])
            tts.speak(response["content"])
        message = input("User: ")
        response = llm.prompt(message)
            
