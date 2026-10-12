"""Run a shell command and stream its output into the notebook cell (subprocess output is otherwise invisible)."""
import subprocess
import sys


def run(cmd: str) -> None:
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"command failed with exit code {p.returncode}: {cmd}")


if __name__ == "__main__":
    run(" ".join(sys.argv[1:]))
