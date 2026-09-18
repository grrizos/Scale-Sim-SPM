#!/usr/bin/env python3
"""cProfile subset: 2 models x combos.PROFILE_COMBO_IDS x 2 versions, 1 rep
each (not the full sweep -- profiling is about relative %-breakdown
stability, not noise-sensitive timing, so one rep is enough).

Writes profile_breakdown.csv with the top-25 functions by cumulative time
and by tottime for every profiled run.
"""
import argparse
import csv
import os
import pstats
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combos
import models
import config_gen
import subprocess_utils

TOP_N = 25
COMBO_FIELDS = ["array_size", "sram_kb", "interface_bandwidth", "dataflow",
                 "ifmap_offset", "filter_offset", "ofmap_offset", "rng_seed"]
FIELDNAMES = ["run_id", "model", "combo_id", "version", "sort_metric", "rank",
              "function", "file", "line", "ncalls_primitive", "ncalls_total",
              "tottime", "cumtime", "pct_of_total_cumtime"]


def build_tasks():
    all_combos = combos.all_main_combos()
    tasks = []
    for model in models.PROFILE_MODELS:
        for combo_id in combos.PROFILE_COMBO_IDS:
            for version in ("vanilla", "optimized"):
                tasks.append(dict(
                    model=model, combo_id=combo_id, combo=all_combos[combo_id],
                    version=version, run_id=f"prof__{model}__{combo_id}__{version}",
                ))
    return tasks


def extract_rows(run_id, model, combo_id, version, pstats_path):
    """Two independent top-25 extractions per run (cumulative, tottime) --
    sort_stats mutates its Stats object in place, so use a fresh one per
    metric rather than re-sorting the same object."""
    rows = []
    for metric in ("cumulative", "tottime"):
        stats = pstats.Stats(pstats_path)
        stats.sort_stats(metric)
        total_tt = stats.total_tt
        for rank, key in enumerate(stats.fcn_list[:TOP_N], start=1):
            filename, line, funcname = key
            cc, nc, tt, ct, _callers = stats.stats[key]
            rows.append(dict(
                run_id=run_id, model=model, combo_id=combo_id, version=version,
                sort_metric=metric, rank=rank, function=funcname, file=filename, line=line,
                ncalls_primitive=cc, ncalls_total=nc,
                tottime=round(tt, 6), cumtime=round(ct, 6),
                pct_of_total_cumtime=round((ct / total_tt * 100.0) if total_tt else 0.0, 3),
            ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla-repo", required=True)
    ap.add_argument("--optimized-repo", required=True)
    ap.add_argument("--vanilla-venv-python", required=True)
    ap.add_argument("--optimized-venv-python", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    args = ap.parse_args()

    repos = {"vanilla": args.vanilla_repo, "optimized": args.optimized_repo}
    venvs = {"vanilla": args.vanilla_venv_python, "optimized": args.optimized_venv_python}

    profiles_root = os.path.join(args.results_root, "profiles")
    cfg_dir = os.path.join(args.results_root, "cfgs")
    out_csv = os.path.join(args.results_root, "profile_breakdown.csv")

    tasks = build_tasks()
    all_rows = []
    for i, task in enumerate(tasks, start=1):
        version = task["version"]
        repo_root = repos[version]
        venv_python = venvs[version]
        model_info = models.MODELS[task["model"]]
        combo = task["combo"]

        cfg_path = config_gen.write_config(
            cfg_dir, task["run_id"], **{k: combo[k] for k in COMBO_FIELDS})

        runs_parent_dir = os.path.join(profiles_root, version, "runs")
        os.makedirs(runs_parent_dir, exist_ok=True)
        pstats_dir = os.path.join(profiles_root, version)
        os.makedirs(pstats_dir, exist_ok=True)
        pstats_path = os.path.join(pstats_dir, task["run_id"] + ".pstats")

        topology_path = os.path.join(repo_root, model_info["topology"])
        layout_path = os.path.join(repo_root, models.LAYOUT_FILE)
        scale_py = os.path.join(repo_root, "scalesim", "scale.py")

        # Script-path form (not "-m scalesim.scale") to sidestep any
        # cProfile/-m combination quirk across Python versions.
        cmd = [venv_python, "-m", "cProfile", "-o", pstats_path, scale_py,
               "-c", cfg_path, "-t", topology_path, "-l", layout_path,
               "-p", runs_parent_dir, "-i", model_info["input_type"], "-s", "N"]

        # Verified empirically: "-m cProfile" puts cwd ('') at sys.path[0]
        # (unlike a plain "python scale.py" invocation, where sys.path[0]
        # is scale.py's own directory), so "import scalesim" inside
        # scale.py resolves against whatever repo cwd is, NOT necessarily
        # venv_python's own site-packages/editable-install. This only
        # imports the right code because cwd is always repo_root here --
        # don't decouple them.
        log_path = os.path.join(args.results_root, "logs", version, task["run_id"] + ".log")
        print(f"[{i}/{len(tasks)}] {task['run_id']} -- profiling...")
        returncode, elapsed, timed_out = subprocess_utils.run_one(
            cmd, repo_root, log_path, args.timeout_s)

        output_dir = os.path.join(runs_parent_dir, task["run_id"])
        if timed_out or returncode != 0 or not os.path.isfile(pstats_path):
            print(f"  FAILED (returncode={returncode}, timed_out={timed_out}) -- see {log_path}")
            continue

        rows = extract_rows(task["run_id"], task["model"], task["combo_id"], version, pstats_path)
        all_rows.extend(rows)
        print(f"  -> {elapsed:.1f}s, {len(rows)} rows extracted")

        # Same disk-usage mitigation as run_sweep.py.
        if os.path.isdir(output_dir):
            for name in os.listdir(output_dir):
                full = os.path.join(output_dir, name)
                if os.path.isdir(full):
                    shutil.rmtree(full, ignore_errors=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {out_csv} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
