from module import LLMModule

import os

class TerminalModule(LLMModule):
    def __init__(self) -> None:
        super().__init__("terminal")
        self.terminal = os.popen


    def activate(self, message: str) -> str:
        command = message.split("<|terminal|>")[1]
        self.terminal.write(command + "\n")
        self.terminal.flush()
        return self.terminal.read()