from dotenv import load_dotenv  # noqa
load_dotenv()  # noqa

from jarvis.llm.modules.module_registry import ModuleRegistry
from jarvis.llm.modules.terminal import TerminalModule
from jarvis.llm.gpt_turbo import LLM


term = TerminalModule()

llm = LLM(
    """You are Jarvis. Tony Stark's personnal assistant. 
You have some modules installed that allow you to interact with external things by beginning your answer with the module prefix. You have to only provide the module command in your answer for it to execute.

The modules are: \"""
[terminal]<bash command> #A terminal module that allows me to interact with a persistent bash shell.
[silent] #Use this command when you have nothing to say
\"""

Each module will give you their output by responding as the user, starting the message with the module prefix.

Example: \"""
user:What are the files in your current directory?
assistant:[terminal]ls
user:[terminal]file1 file2 file3
assistant:The files in my current directory are file1, file2 and file3
user:Create a symbolic link from file1 to my local bin folder
assistant:[terminal]ln -s file1 ~/.local/bin/file1
user:[terminal]NO-OUTPUT
assistant:Symbolic link created
user:Great!
assistant:[silent]
\"""
"""
)
response = ""
while True:
    message = ModuleRegistry.get_instance().evaluate(response)
    if message:
        print(message)
    else:
        message = input("User: ")

    response = llm.prompt(message)

    print(f"Jarvis: {response}")
