#!/usr/bin/env python3
"""Monitor GPU4/5 λ=0.0 Python processes. When they exit, kill bash loops to prevent overwriting parallel results."""
import os, time, signal

# (bash_pid, python_pid, label)
jobs = [
    (149685, 149686, "GPU4 sp3 λ=0.0"),
    (149875, 149876, "GPU5 sp4 λ=0.0"),
]

def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

print("Monitor started: watching GPU4/5 λ=0.0 Python processes", flush=True)
while jobs:
    remaining = []
    for bash_pid, py_pid, label in jobs:
        if pid_alive(py_pid):
            remaining.append((bash_pid, py_pid, label))
        else:
            print(f"[{label}] Python exited. Killing bash loop PID={bash_pid}...", flush=True)
            try:
                os.kill(bash_pid, signal.SIGTERM)
                time.sleep(1)
                if pid_alive(bash_pid):
                    os.kill(bash_pid, signal.SIGKILL)
                print(f"[{label}] Bash loop killed.", flush=True)
            except OSError as e:
                print(f"[{label}] Failed to kill bash: {e}", flush=True)
    jobs = remaining
    if jobs:
        time.sleep(60)
print("Monitor: all λ=0.0 done, bash loops terminated.", flush=True)
