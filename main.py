from dotenv import load_dotenv  # noqa
load_dotenv()  # noqa

from jarvis.llm.modules.module_registry import ModuleRegistry
from jarvis.llm.modules.terminal import TerminalModule
from jarvis.llm.gpt_turbo import LLM


term = TerminalModule()

llm = LLM()

response = ""
while True:
    message = ModuleRegistry.get_instance().evaluate(response)
    if message:
        print(f"MODULE: {message}")
    else:
        message = input("User: ")

    response = llm.prompt(message)

    print(f"Jarvis: {response}")
