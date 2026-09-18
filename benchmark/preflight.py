#!/usr/bin/env python3
"""Sanity checks to run before the full benchmark sweep.

Verifies both repos have the required topology/layout files, both venv
interpreters can import scalesim, reports each version's numpy version
(warning, not blocking, on a mismatch), and checks --results-root is
writable with enough free disk space. Exits non-zero with a full list of
problems rather than letting the first real failure surface partway
through an unattended multi-hour sweep.
"""
import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import MODELS, LAYOUT_FILE

VERSIONS = ("vanilla", "optimized")


def check_files(repo_root, version):
    problems = []
    for model_name, info in MODELS.items():
        path = os.path.join(repo_root, info["topology"])
        if not os.path.isfile(path):
            problems.append(f"[{version}] missing topology for '{model_name}': {path}")
    layout_path = os.path.join(repo_root, LAYOUT_FILE)
    if not os.path.isfile(layout_path):
        problems.append(f"[{version}] missing layout file: {layout_path}")
    scale_py = os.path.join(repo_root, "scalesim", "scale.py")
    if not os.path.isfile(scale_py):
        problems.append(f"[{version}] missing entry point: {scale_py}")
    return problems


_PROBE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_import_probe.py")
_PROBE_CWD = os.path.dirname(_PROBE_SCRIPT)  # benchmark/ itself -- has no nested scalesim/


def check_python(venv_python, version, repo_root):
    """Runs _import_probe.py as a plain script (not "-c") so resolution
    goes through the same path a real scale.py subprocess would use, then
    confirms the scalesim it actually imported lives under repo_root --
    catching a venv that resolves to some other (possibly stale) install
    instead of the intended repo, which "-c"'s cwd-priority import
    behavior can mask. Confirmed necessary on this machine: a stale
    non-editable scalesim copy in ~/.local/lib/python3.10/site-packages
    was silently picked up ahead of an editable install by a plain
    script-style subprocess, even though a "-c" check looked fine."""
    problems = []
    numpy_version = None
    try:
        out = subprocess.run(
            [venv_python, _PROBE_SCRIPT], cwd=_PROBE_CWD,
            capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            problems.append(
                f"[{version}] '{venv_python}' failed to import scalesim/numpy:\n{out.stderr}")
        else:
            lines = out.stdout.strip().splitlines()
            if len(lines) < 2:
                problems.append(f"[{version}] '{venv_python}' probe produced unexpected output: {out.stdout!r}")
            else:
                scalesim_file, numpy_version = lines[0], lines[1]
                resolved_dir = os.path.realpath(os.path.dirname(os.path.dirname(scalesim_file)))
                expected_dir = os.path.realpath(repo_root)
                if resolved_dir != expected_dir:
                    problems.append(
                        f"[{version}] '{venv_python}' imports scalesim from {scalesim_file}, "
                        f"which is NOT under --{version}-repo ({repo_root}) -- this venv is "
                        f"not correctly isolated/installed for this repo (stale system-wide or "
                        f"user-site install shadowing it?). Fix before running the sweep, or "
                        f"every run under this version will silently test the wrong code.")
    except FileNotFoundError:
        problems.append(f"[{version}] venv python not found: {venv_python}")
    except subprocess.TimeoutExpired:
        problems.append(f"[{version}] import probe timed out")
    return problems, numpy_version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla-repo", required=True)
    ap.add_argument("--optimized-repo", required=True)
    ap.add_argument("--vanilla-venv-python", required=True)
    ap.add_argument("--optimized-venv-python", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    args = ap.parse_args()

    repos = {"vanilla": args.vanilla_repo, "optimized": args.optimized_repo}
    venvs = {"vanilla": args.vanilla_venv_python, "optimized": args.optimized_venv_python}

    all_problems = []
    numpy_versions = {}

    for version in VERSIONS:
        all_problems += check_files(repos[version], version)
        problems, numpy_version = check_python(venvs[version], version, repos[version])
        all_problems += problems
        numpy_versions[version] = numpy_version

    if numpy_versions["vanilla"] and numpy_versions["optimized"] \
            and numpy_versions["vanilla"] != numpy_versions["optimized"]:
        print(f"WARNING: numpy version mismatch -- vanilla={numpy_versions['vanilla']} "
              f"optimized={numpy_versions['optimized']}. Not blocking, but this can "
              f"confound the speed comparison since the optimization is numpy-vectorization-based.")

    os.makedirs(args.results_root, exist_ok=True)
    if not os.access(args.results_root, os.W_OK):
        all_problems.append(f"--results-root not writable: {args.results_root}")
    else:
        free_gb = shutil.disk_usage(args.results_root).free / (1024 ** 3)
        if free_gb < args.min_free_gb:
            all_problems.append(
                f"only {free_gb:.1f}GB free at {args.results_root}, wanted at least "
                f"{args.min_free_gb}GB (individual runs can produce 1-2GB+ of trace "
                f"data before the harness's per-run cleanup deletes it)")

    print()
    if all_problems:
        print(f"PREFLIGHT: {len(all_problems)} problem(s) found:")
        for p in all_problems:
            print(f"  - {p}")
        sys.exit(1)

    print("PREFLIGHT: all checks passed.")
    print(f"  numpy: vanilla={numpy_versions['vanilla']} optimized={numpy_versions['optimized']}")
    sys.exit(0)


if __name__ == "__main__":
    main()
