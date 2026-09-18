#!/usr/bin/env python3
"""Main timing driver: runs the vanilla-vs-optimized benchmark sweep.

For each (model, combo, repeat, version) task: generates a .cfg, shells
out to that version's own venv python running scalesim/scale.py (separate
checkouts can't share one Python environment), times it, parses
COMPUTE_REPORT.csv, deletes the run's per-layer trace subdirectories
(scale.py's -s flag is dead code -- save_disk_space is hardcoded False in
its __main__ regardless of what -s is passed, so every run writes full
per-cycle traces no matter what; this must be cleaned up per-run), and
appends one row to results_raw.csv.

Resumable: safe to Ctrl-C and rerun the same command -- rows already
recorded with status "ok" are skipped, everything else is retried.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combos
import models
import config_gen
import subprocess_utils

COMBO_FIELDS = ["array_size", "sram_kb", "interface_bandwidth", "dataflow",
                 "ifmap_offset", "filter_offset", "ofmap_offset", "rng_seed"]

FIELDNAMES = [
    "run_id", "phase", "model", "combo_id", "version", "repeat_idx",
    "array_size", "sram_kb", "interface_bandwidth", "dataflow",
    "ifmap_offset", "filter_offset", "ofmap_offset", "rng_seed",
    "wall_seconds", "returncode", "status", "timed_out",
    "total_cycles", "total_cycles_incl_prefetch", "num_layers_reported",
    "topology_file", "config_path", "output_dir", "log_path",
    "numpy_version", "hostname", "python_version", "cpu_model", "timestamp",
]


# ---------------------------------------------------------------- tasks --

def build_tasks(repeats):
    """model -> combo -> repeat -> version, so vanilla/optimized for the
    same (model, combo, repeat) run back-to-back (spreads any thermal
    drift evenly across both sides instead of biasing one)."""
    tasks = []
    main_combos = combos.all_main_combos()
    for model in models.MAIN_GRID_MODELS:
        for combo_id, combo in main_combos.items():
            for repeat_idx in range(1, repeats + 1):
                for version in ("vanilla", "optimized"):
                    tasks.append(dict(
                        phase="main_grid", model=model, combo_id=combo_id, combo=combo,
                        repeat_idx=repeat_idx, version=version,
                        run_id=f"{model}__{combo_id}__{version}__rep{repeat_idx}",
                    ))

    addendum_combos = combos.dataflow_addendum_combos()
    for model in combos.DATAFLOW_ADDENDUM_MODELS:
        for combo_id, combo in addendum_combos.items():
            for repeat_idx in range(1, repeats + 1):
                for version in ("vanilla", "optimized"):
                    tasks.append(dict(
                        phase="dataflow_addendum", model=model, combo_id=combo_id, combo=combo,
                        repeat_idx=repeat_idx, version=version,
                        run_id=f"{model}__{combo_id}__{version}__rep{repeat_idx}",
                    ))
    return tasks


def build_smoke_tasks():
    combo = combos.all_main_combos()["anchor"]
    return [
        dict(phase="main_grid", model="alexnet", combo_id="anchor", combo=combo,
             repeat_idx=1, version=version, run_id=f"alexnet__anchor__{version}__rep1")
        for version in ("vanilla", "optimized")
    ]


def parse_only(only_str):
    filters = {}
    for pair in only_str.split(","):
        key, _, value = pair.partition("=")
        filters[key.strip()] = value.strip()
    return filters


def task_matches(task, filters):
    for key, value in filters.items():
        if key == "repeat":
            if str(task["repeat_idx"]) != value:
                return False
        elif key == "combo":
            if task["combo_id"] != value:
                return False
        elif str(task.get(key, "")) != value:
            return False
    return True


# ----------------------------------------------------------- execution --

def capture_static_meta(venv_python):
    out = subprocess.run(
        [venv_python, "-c",
         "import numpy, platform; "
         "print(numpy.__version__); print(platform.python_version()); "
         "print(platform.processor() or platform.machine())"],
        capture_output=True, text=True, timeout=60)
    lines = out.stdout.strip().splitlines()
    lines += [""] * (3 - len(lines))
    numpy_version, python_version, cpu_model = lines[:3]
    return dict(numpy_version=numpy_version, python_version=python_version, cpu_model=cpu_model,
                hostname=os.uname().nodename if hasattr(os, "uname") else "")


def parse_compute_report(output_dir):
    """Sums COMPUTE_REPORT.csv's 'Total Cycles' (col 2) and 'Total Cycles
    (incl. prefetch)' (col 1) columns, confirmed against
    results/scale_example_run_32x32_ws/COMPUTE_REPORT.csv's header:
    LayerID, Total Cycles (incl. prefetch), Total Cycles, Stall Cycles, ..."""
    path = os.path.join(output_dir, "COMPUTE_REPORT.csv")
    total_cycles = 0
    total_cycles_incl_prefetch = 0
    num_layers = 0
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row or not row[0].strip():
                continue
            total_cycles_incl_prefetch += int(float(row[1]))
            total_cycles += int(float(row[2]))
            num_layers += 1
    if num_layers == 0:
        raise ValueError("COMPUTE_REPORT.csv had no data rows")
    return total_cycles, total_cycles_incl_prefetch, num_layers


def cleanup_trace_dirs(output_dir):
    """Deletes per-layer trace subdirectories, keeping the 4 top-level
    summary report CSVs. Needed because scale.py's -s flag is dead code
    (save_disk_space is hardcoded False), so every run writes full
    per-cycle traces regardless -- observed up to ~2.2GB for one run."""
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if os.path.isdir(full):
            shutil.rmtree(full, ignore_errors=True)


def execute_task(task, repos, venvs, results_root, timeout_s, static_meta):
    model_info = models.MODELS[task["model"]]
    combo = task["combo"]
    run_id = task["run_id"]
    version = task["version"]
    repo_root = repos[version]
    venv_python = venvs[version]

    cfg_dir = os.path.join(results_root, "cfgs")
    cfg_path = config_gen.write_config(cfg_dir, run_id, **{k: combo[k] for k in COMBO_FIELDS})

    runs_parent_dir = os.path.join(results_root, "runs", version)
    os.makedirs(runs_parent_dir, exist_ok=True)
    output_dir = os.path.join(runs_parent_dir, run_id)

    topology_path = os.path.join(repo_root, model_info["topology"])
    layout_path = os.path.join(repo_root, models.LAYOUT_FILE)
    scale_py = os.path.join(repo_root, "scalesim", "scale.py")

    cmd = [venv_python, scale_py,
           "-c", cfg_path, "-t", topology_path, "-l", layout_path,
           "-p", runs_parent_dir, "-i", model_info["input_type"], "-s", "N"]

    log_path = os.path.join(results_root, "logs", version, run_id + ".log")
    returncode, elapsed, timed_out = subprocess_utils.run_one(cmd, repo_root, log_path, timeout_s)

    row = dict(
        run_id=run_id, phase=task["phase"], model=task["model"], combo_id=task["combo_id"],
        version=version, repeat_idx=task["repeat_idx"],
        array_size=combo["array_size"], sram_kb=combo["sram_kb"],
        interface_bandwidth=combo["interface_bandwidth"], dataflow=combo["dataflow"],
        ifmap_offset=combo["ifmap_offset"], filter_offset=combo["filter_offset"],
        ofmap_offset=combo["ofmap_offset"], rng_seed=combo["rng_seed"],
        wall_seconds=round(elapsed, 3), returncode=returncode, timed_out=timed_out,
        total_cycles="", total_cycles_incl_prefetch="", num_layers_reported="",
        topology_file=topology_path, config_path=cfg_path, output_dir=output_dir, log_path=log_path,
        numpy_version=static_meta[version]["numpy_version"],
        hostname=static_meta[version]["hostname"],
        python_version=static_meta[version]["python_version"],
        cpu_model=static_meta[version]["cpu_model"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    if timed_out:
        row["status"] = "timeout"
    elif returncode != 0:
        row["status"] = "failed"
    else:
        try:
            total_cycles, total_cycles_incl_prefetch, num_layers = parse_compute_report(output_dir)
            row["total_cycles"] = total_cycles
            row["total_cycles_incl_prefetch"] = total_cycles_incl_prefetch
            row["num_layers_reported"] = num_layers
            row["status"] = "ok"
        except Exception as e:
            row["status"] = "parse_error"
            with open(log_path, "a") as logf:
                logf.write(f"\n\nPARSE ERROR: {e}\n")

    cleanup_trace_dirs(output_dir)
    return row


# ----------------------------------------------------------------- csv --

def load_done_run_ids(csv_path):
    """Last recorded status per run_id (append-only file, retries can add
    more than one row for the same id) -- only 'ok' counts as done."""
    if not os.path.isfile(csv_path):
        return set()
    latest_status = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            latest_status[row["run_id"]] = row["status"]
    return {rid for rid, status in latest_status.items() if status == "ok"}


# ---------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla-repo", required=True)
    ap.add_argument("--optimized-repo", required=True)
    ap.add_argument("--vanilla-venv-python", required=True)
    ap.add_argument("--optimized-venv-python", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--timeout-s", type=float, default=3600.0,
                     help="per-run wall-clock cap; kills and logs, sweep continues (default 1h)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--smoke-test", action="store_true",
                     help="run one cheap task through the whole pipeline and exit")
    ap.add_argument("--only", default=None,
                     help="e.g. model=resnet50,combo=grid_a64_s16_calc,version=vanilla,repeat=1")
    args = ap.parse_args()

    repos = {"vanilla": args.vanilla_repo, "optimized": args.optimized_repo}
    venvs = {"vanilla": args.vanilla_venv_python, "optimized": args.optimized_venv_python}

    os.makedirs(args.results_root, exist_ok=True)
    results_csv = os.path.join(args.results_root, "results_raw.csv")

    print("Capturing environment metadata...")
    static_meta = {v: capture_static_meta(venvs[v]) for v in ("vanilla", "optimized")}
    if static_meta["vanilla"]["numpy_version"] != static_meta["optimized"]["numpy_version"]:
        print(f"WARNING: numpy version mismatch -- vanilla={static_meta['vanilla']['numpy_version']} "
              f"optimized={static_meta['optimized']['numpy_version']}")

    tasks = build_smoke_tasks() if args.smoke_test else build_tasks(args.repeats)

    if args.only:
        filters = parse_only(args.only)
        tasks = [t for t in tasks if task_matches(t, filters)]
        if not tasks:
            print(f"--only matched 0 tasks (filters={filters})")
            sys.exit(1)

    done = load_done_run_ids(results_csv)
    file_exists = os.path.isfile(results_csv) and os.path.getsize(results_csv) > 0

    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
            f.flush()

        total = len(tasks)
        for i, task in enumerate(tasks, start=1):
            if task["run_id"] in done:
                print(f"[{i}/{total}] {task['run_id']} -- already done, skipping")
                continue
            print(f"[{i}/{total}] {task['run_id']} -- running...")
            row = execute_task(task, repos, venvs, args.results_root, args.timeout_s, static_meta)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
            print(f"[{i}/{total}] {task['run_id']} -> {row['wall_seconds']:.1f}s status={row['status']}")

    print(f"\nDone. Results in {results_csv}")


if __name__ == "__main__":
    main()
