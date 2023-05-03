import subprocess
from time import sleep
import psutil
import os

process = subprocess.Popen(
    ['/bin/bash'], shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

os.set_blocking(process.stdout.fileno(), False)
os.set_blocking(process.stderr.fileno(), False)


def run_cmd(cmd: str) -> str:
    child_count = len(psutil.Process(pid=process.pid).children())
    process.stdin.write(f"{cmd}\n".encode())
    process.stdin.flush()
    sleep(0.1)
    while (len(psutil.Process(pid=process.pid).children()) != child_count):
        sleep(0.1)

    out = ''
    for line in process.stdout:
        out += line.decode()

    err = ''
    for line in process.stderr:
        err += line.decode()

    return f"{out}{err}".strip()


print(run_cmd("cd jarvis && pwd && cd llm && sleep 2 && pwd"))
print(run_cmd("pwd"))
print(run_cmd("cd .."))
print(run_cmd("cd momo"))
