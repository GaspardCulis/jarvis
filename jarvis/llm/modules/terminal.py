import os
import subprocess
from time import sleep
import psutil
from jarvis.llm.modules.module import LLMModule


def get_pid_childs_count(pid: int):
    ps = psutil.Process(pid=pid)
    return len(ps.children())


class TerminalModule(LLMModule):
    def __init__(self) -> None:
        super().__init__("TERMINAL")
        self.terminal = subprocess.Popen(
            ['/bin/bash'], shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.set_blocking(self.terminal.stdout.fileno(), False)
        os.set_blocking(self.terminal.stderr.fileno(), False)

    def activate(self, message: str) -> str:
        command = message.split(f"[{self.prefix}]")[1]
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

        return f"[{self.prefix}]{out}{err}".strip()
