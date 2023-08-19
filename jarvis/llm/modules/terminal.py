import os
import subprocess
from time import sleep
import psutil
import json
from jarvis.llm.modules.module import LLMModule


def get_pid_childs_count(pid: int):
    ps = psutil.Process(pid=pid)
    return len(ps.children())


class TerminalModule(LLMModule):
    def __init__(self) -> None:
        super().__init__("terminal", "Executes a bash command in a persitent shell session.", {"command":("string", "The command to execute")}) 
        self.terminal = subprocess.Popen(
            ['/bin/bash'], shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.set_blocking(self.terminal.stdout.fileno(), False)
        os.set_blocking(self.terminal.stderr.fileno(), False)

    def get_preprompts(self) -> list[dict]:
        return [
            {
                "role": "user", 
                "content": 
                "What OS am I running ?"
            },
            {
                "role": 
                "assistant", 
                "content": None,
                "function_call": {
                    "name": self.name,
                    "arguments": "{\n  \"command\": \"cat /etc/os-release\"\n}"
                }
            },
            {
                'role': 'function', 
                'name': self.name, 
                'content': self.activate({
                    "command": "cat /etc/os-release"
                })
            },
            {
                "role": "assistant",
                "content": "Your operating system is " + json.loads(self.activate({
                    "command": """lsb_release -ds | sed 's/"//g'"""
                }))["stdout"]
            }
        ]

    def activate(self, arguments: dict) -> str:
        command = arguments.get("command")
        if not command:
            return "Error: No 'command' argument provided"

        child_count = len(psutil.Process(pid=self.terminal.pid).children())
        self.terminal.stdin.write(f"{command}\n".encode())
        self.terminal.stdin.flush()
        sleep(0.1)
        while (len(psutil.Process(pid=self.terminal.pid).children()) != child_count):
            sleep(0.1)

        out = ''
        for line in self.terminal.stdout:
            try:
                out += line.decode()
            except UnicodeDecodeError as e:
                pass

        err = ''
        for line in self.terminal.stderr:
            err += line.decode()

        exit_code = self.terminal.poll() or 0
        return json.dumps({
            "stdout": out if out else ("success" if not (err or exit_code) else ""), # Hack, if all stdou, stderr and exit_code are empty, GPT will think that the command failed and will retry
            "stderr": err,
            "exit_code": exit_code
        })
