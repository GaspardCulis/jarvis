from dotenv import load_dotenv  # noqa
load_dotenv()  # noqa

from jarvis.llm.modules.module_registry import ModuleRegistry
from jarvis.llm.modules.terminal import TerminalModule
from jarvis.llm.gpt_turbo import LLM


term = TerminalModule()

llm = LLM()

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
        print(response["content"])
    else:
        message = input("User: ")
        response = llm.prompt(message)
        if response.get("content"):
            print(response["content"])
