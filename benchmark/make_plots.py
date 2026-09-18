#!/usr/bin/env python3
"""Reads results_raw.csv / profile_breakdown.csv and writes static PNG
charts (matplotlib, already a project dependency). Safely re-runnable at
any point, including mid-sweep against a partial results_raw.csv.
"""
import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combos as combos_mod

VANILLA_COLOR = "#888888"
OPTIMIZED_COLOR = "#2b7de9"
MAIN_GRID_COMBO_ORDER = list(combos_mod.all_main_combos().keys())
GRID_ONLY_COMBO_ORDER = [c for c in MAIN_GRID_COMBO_ORDER if c.startswith("grid_") or c == "anchor"]


def load_dedup_rows(csv_path):
    """Last occurrence per run_id wins (resume-safe against retry duplicates)."""
    latest = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            latest[row["run_id"]] = row
    return list(latest.values())


def aggregate_wall_seconds(rows):
    """(model, combo_id, version) -> (median, min, max) over ok repeats."""
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (row["model"], row["combo_id"], row["version"])
        grouped[key].append(float(row["wall_seconds"]))
    return {key: (statistics.median(vals), min(vals), max(vals)) for key, vals in grouped.items()}


def plot_per_model(agg, models_list, out_dir):
    for model in models_list:
        combo_ids = [c for c in MAIN_GRID_COMBO_ORDER
                     if (model, c, "vanilla") in agg or (model, c, "optimized") in agg]
        if not combo_ids:
            continue
        x = list(range(len(combo_ids)))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(combo_ids) * 1.2), 5))
        for offset, version, color in ((-1, "vanilla", VANILLA_COLOR), (1, "optimized", OPTIMIZED_COLOR)):
            medians, err_low, err_high = [], [], []
            for c in combo_ids:
                med, lo, hi = agg.get((model, c, version), (0.0, 0.0, 0.0))
                medians.append(med)
                err_low.append(med - lo)
                err_high.append(hi - med)
            xs = [xi + offset * width / 2 for xi in x]
            ax.bar(xs, medians, width=width, label=version, color=color,
                   yerr=[err_low, err_high], capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(combo_ids, rotation=45, ha="right")
        ax.set_ylabel("wall seconds (median of repeats)")
        ax.set_title(f"{model}: vanilla vs optimized runtime")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"speed_{model}.png"), dpi=150)
        plt.close(fig)


def plot_speedup_heatmap(agg, models_list, out_dir):
    data = []
    for model in models_list:
        row = []
        for c in GRID_ONLY_COMBO_ORDER:
            v = agg.get((model, c, "vanilla"))
            o = agg.get((model, c, "optimized"))
            row.append((v[0] - o[0]) / v[0] * 100.0 if v and o and v[0] > 0 else float("nan"))
        data.append(row)

    fig, ax = plt.subplots(figsize=(max(8, len(GRID_ONLY_COMBO_ORDER) * 1.1),
                                     max(3, len(models_list) * 0.8)))
    im = ax.imshow(data, cmap="RdYlGn", vmin=-20, vmax=60, aspect="auto")
    ax.set_xticks(range(len(GRID_ONLY_COMBO_ORDER)))
    ax.set_xticklabels(GRID_ONLY_COMBO_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(len(models_list)))
    ax.set_yticklabels(models_list)
    for i in range(len(models_list)):
        for j in range(len(GRID_ONLY_COMBO_ORDER)):
            val = data[i][j]
            if val == val:  # not NaN
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="speedup % (optimized vs vanilla)")
    ax.set_title("Speedup % across the speed-parameter grid")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "speedup_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_dataflow_addendum(agg, out_dir):
    models_list = combos_mod.DATAFLOW_ADDENDUM_MODELS
    combo_ids = ["anchor"] + [f"anchor_df{d}" for d in combos_mod.DATAFLOW_ADDENDUM_EXTRA_DATAFLOWS]
    labels = ["ws"] + list(combos_mod.DATAFLOW_ADDENDUM_EXTRA_DATAFLOWS)

    fig, axes = plt.subplots(1, len(models_list), figsize=(6 * len(models_list), 5), squeeze=False)
    for ax, model in zip(axes[0], models_list):
        width = 0.35
        x = list(range(len(combo_ids)))
        for offset, version, color in ((-1, "vanilla", VANILLA_COLOR), (1, "optimized", OPTIMIZED_COLOR)):
            medians = [agg.get((model, c, version), (0.0, 0.0, 0.0))[0] for c in combo_ids]
            xs = [xi + offset * width / 2 for xi in x]
            ax.bar(xs, medians, width=width, label=version, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("wall seconds (median)")
        ax.set_title(model)
        ax.legend()
    fig.suptitle("Dataflow generalization check")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dataflow_addendum.png"), dpi=150)
    plt.close(fig)


def plot_profile_breakdown(profile_rows, out_dir):
    groups = defaultdict(list)
    for row in profile_rows:
        if row["sort_metric"] != "cumulative":
            continue
        key = (row["model"], row["combo_id"], row["version"])
        groups[key].append(row)

    pairs = defaultdict(dict)
    for (model, combo_id, version), rows in groups.items():
        pairs[(model, combo_id)][version] = sorted(rows, key=lambda r: int(r["rank"]))[:10]

    for (model, combo_id), by_version in pairs.items():
        if "vanilla" not in by_version or "optimized" not in by_version:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, version, color in zip(axes, ("vanilla", "optimized"), (VANILLA_COLOR, OPTIMIZED_COLOR)):
            rows = by_version[version]
            funcs = [r["function"] for r in rows][::-1]
            pcts = [float(r["pct_of_total_cumtime"]) for r in rows][::-1]
            ax.barh(funcs, pcts, color=color)
            ax.set_xlabel("% of total cumulative time")
            ax.set_title(version)
        fig.suptitle(f"{model} / {combo_id}: top-10 functions by cumulative time")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"profile_{model}_{combo_id}.png"), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    args = ap.parse_args()

    results_csv = os.path.join(args.results_root, "results_raw.csv")
    profile_csv = os.path.join(args.results_root, "profile_breakdown.csv")
    out_dir = os.path.join(args.results_root, "plots")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(results_csv):
        print(f"No results_raw.csv found at {results_csv} yet -- nothing to plot.")
        return

    rows = load_dedup_rows(results_csv)
    agg = aggregate_wall_seconds(rows)
    models_list = sorted({row["model"] for row in rows})

    plot_per_model(agg, models_list, out_dir)
    plot_speedup_heatmap(agg, models_list, out_dir)
    if any(m in combos_mod.DATAFLOW_ADDENDUM_MODELS for m in models_list):
        plot_dataflow_addendum(agg, out_dir)

    if os.path.isfile(profile_csv):
        with open(profile_csv, newline="") as f:
            profile_rows = list(csv.DictReader(f))
        plot_profile_breakdown(profile_rows, out_dir)
    else:
        print(f"No profile_breakdown.csv found at {profile_csv} -- skipping profiling charts.")

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
