#!/usr/bin/env python3
"""Post-processes results_raw.csv into a vanilla-vs-optimized cycle-count
correctness check for the main-grid combos (first repeat only -- cycle
counts are deterministic given a config, so 1 rep is enough). Pure
post-processing, no re-running: confirms the speedup didn't change
simulated results.
"""
import argparse
import csv
import os

FIELDNAMES = [
    "model", "combo_id", "dataflow", "vanilla_run_id", "optimized_run_id",
    "vanilla_total_cycles", "optimized_total_cycles", "cycles_delta", "cycles_match",
    "vanilla_total_cycles_incl_prefetch", "optimized_total_cycles_incl_prefetch",
    "cycles_incl_prefetch_match",
    "vanilla_num_layers_reported", "optimized_num_layers_reported", "layer_count_match",
    "overall_status",
]


def load_latest_rows(results_csv):
    """Last occurrence per run_id wins -- resume-safe against an
    append-only results_raw.csv that can contain retry duplicates."""
    latest = {}
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            latest[row["run_id"]] = row
    return latest


def build_pairs(rows_by_id):
    pairs = {}
    for row in rows_by_id.values():
        if row["phase"] != "main_grid" or row["repeat_idx"] != "1":
            continue
        key = (row["model"], row["combo_id"])
        pairs.setdefault(key, {})[row["version"]] = row
    return pairs


def compare_pair(model, combo_id, vrow, orow):
    if not vrow or not orow or vrow["status"] != "ok" or orow["status"] != "ok":
        any_row = vrow or orow or {}
        return dict(
            model=model, combo_id=combo_id, dataflow=any_row.get("dataflow", ""),
            vanilla_run_id=vrow["run_id"] if vrow else "",
            optimized_run_id=orow["run_id"] if orow else "",
            vanilla_total_cycles="", optimized_total_cycles="", cycles_delta="", cycles_match="",
            vanilla_total_cycles_incl_prefetch="", optimized_total_cycles_incl_prefetch="",
            cycles_incl_prefetch_match="",
            vanilla_num_layers_reported=vrow["num_layers_reported"] if vrow else "",
            optimized_num_layers_reported=orow["num_layers_reported"] if orow else "",
            layer_count_match="", overall_status="INCOMPLETE",
        )

    v_cycles, o_cycles = int(vrow["total_cycles"]), int(orow["total_cycles"])
    v_cycles_ip = int(vrow["total_cycles_incl_prefetch"])
    o_cycles_ip = int(orow["total_cycles_incl_prefetch"])
    v_layers, o_layers = vrow["num_layers_reported"], orow["num_layers_reported"]

    cycles_match = v_cycles == o_cycles
    cycles_ip_match = v_cycles_ip == o_cycles_ip
    layer_match = v_layers == o_layers
    overall = "MATCH" if (cycles_match and cycles_ip_match and layer_match) else "MISMATCH"

    return dict(
        model=model, combo_id=combo_id, dataflow=vrow["dataflow"],
        vanilla_run_id=vrow["run_id"], optimized_run_id=orow["run_id"],
        vanilla_total_cycles=v_cycles, optimized_total_cycles=o_cycles,
        cycles_delta=o_cycles - v_cycles, cycles_match=cycles_match,
        vanilla_total_cycles_incl_prefetch=v_cycles_ip,
        optimized_total_cycles_incl_prefetch=o_cycles_ip,
        cycles_incl_prefetch_match=cycles_ip_match,
        vanilla_num_layers_reported=v_layers, optimized_num_layers_reported=o_layers,
        layer_count_match=layer_match, overall_status=overall,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    args = ap.parse_args()

    results_csv = os.path.join(args.results_root, "results_raw.csv")
    out_csv = os.path.join(args.results_root, "correctness_check.csv")

    rows_by_id = load_latest_rows(results_csv)
    pairs = build_pairs(rows_by_id)

    out_rows = [
        compare_pair(model, combo_id, by_version.get("vanilla"), by_version.get("optimized"))
        for (model, combo_id), by_version in sorted(pairs.items())
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    n_mismatch = sum(1 for r in out_rows if r["overall_status"] == "MISMATCH")
    n_incomplete = sum(1 for r in out_rows if r["overall_status"] == "INCOMPLETE")
    print(f"Wrote {out_csv}: {len(out_rows)} combo(s), {n_mismatch} MISMATCH, {n_incomplete} INCOMPLETE")
    if n_mismatch:
        print("WARNING: some combos show different simulated cycle counts between vanilla and optimized.")


if __name__ == "__main__":
    main()
