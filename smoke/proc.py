"""Subprocess launcher that streams live AND persists the full console log.

Task E: on Kaggle a lost notebook session used to take the final traceback with
it, so a mid-training crash (e.g. one rank OOM-ing or its DataLoader stalling)
left no evidence. `stream_and_capture` tees the child's merged stdout+stderr to
the notebook in real time and to `<run_dir>/full_train_console.log`, prints the
log path plus the tail on failure, and re-raises with the real return code.

It never writes the environment (WandB / HF / GCS secrets live there); only the
command list is logged, and that carries no secrets.
"""
import subprocess
import sys
from pathlib import Path


def stream_and_capture(cmd, log_path, env=None, tail_lines=400):
    """Run ``cmd``, streaming output live and appending it to ``log_path``.

    Raises subprocess.CalledProcessError with the child's return code on a
    non-zero exit, after printing the log path and the last ``tail_lines`` lines.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Command is safe to log (no secrets); env is deliberately NOT written.
    header = "$ " + " ".join(str(c) for c in cmd) + "\n"
    with open(log_path, "a", buffering=1, encoding="utf-8") as log:
        log.write(header)
        log.flush()
        print(header, end="", flush=True)
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True,
        )
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
        finally:
            proc.stdout.close()
            returncode = proc.wait()

    if returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"\n===== training FAILED (exit {returncode}) =====")
        print(f"full console log: {log_path}")
        print(f"----- last {tail_lines} lines -----")
        print("\n".join(tail[-tail_lines:]))
        raise subprocess.CalledProcessError(returncode, cmd)
    print(f"\nfull console log saved to: {log_path}")
    return returncode
