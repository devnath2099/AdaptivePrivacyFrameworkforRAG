"""Stream subprocess diagnostics into notebooks and preserve them on disk."""
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time


def run_logged(command, log_dir="review2_final/logs", name="command"):
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = directory / f"{name}_{stamp}_{time.time_ns()}.log"
    tail = deque(maxlen=60)
    print(f"Log: {path.resolve()}", flush=True)
    with path.open("w", encoding="utf-8") as log:
        log.write(f"Command: {command!r}\n")
        log.flush()
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    tail.append(line)
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        log.write(f"\nExit code: {code}\n")
    if code:
        raise RuntimeError(f"Subprocess exited with code {code}. Full log: {path.resolve()}\n"
                           f"Last output lines:\n{''.join(tail)}")
    return path


def run(*args):
    return run_logged([sys.executable, "-u", "-m", "review2_final", *args], name=args[0] if args else "pipeline")
