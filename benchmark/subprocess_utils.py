"""Shared subprocess invocation: timing, timeout handling, logging.

Used by both run_sweep.py and run_profile.py so this logic never
duplicates/drifts between the two drivers.
"""
import os
import subprocess
import time


def run_one(cmd, cwd, log_path, timeout_s):
    """Runs cmd with cwd set, merging stdout+stderr into log_path.

    Times the call with perf_counter (monotonic, high resolution).
    On timeout, subprocess.run has already killed the child before
    raising -- this just logs a clear marker and reports it rather than
    letting the exception propagate and stall the whole sweep. Likewise,
    OSError (e.g. a bad --*-repo/--*-venv-python path, permission denied)
    is caught here rather than left to crash the whole unattended run --
    confirmed necessary empirically: a bad --optimized-repo path raised
    an uncaught FileNotFoundError out of subprocess.run() and killed the
    entire sweep before this was added.

    Returns (returncode, elapsed_seconds, timed_out). returncode is None
    if the process timed out or could not even be started.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as logf:
        logf.write("$ " + " ".join(cmd) + "\n")
        logf.write(f"(cwd={cwd})\n\n")
        logf.flush()
        t0 = time.perf_counter()
        try:
            result = subprocess.run(cmd, cwd=cwd, stdout=logf,
                                     stderr=subprocess.STDOUT, timeout=timeout_s)
            elapsed = time.perf_counter() - t0
            return result.returncode, elapsed, False
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            logf.write(f"\n\nTIMED OUT after {elapsed:.1f}s (limit {timeout_s}s)\n")
            return None, elapsed, True
        except OSError as e:
            elapsed = time.perf_counter() - t0
            logf.write(f"\n\nFAILED TO START PROCESS: {e}\n")
            return None, elapsed, False
