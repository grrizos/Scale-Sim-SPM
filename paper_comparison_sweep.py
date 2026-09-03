#!/usr/bin/env python3
"""
Compare SCALE-Sim fixed-partition baselines against the SMM heterogeneous
scheme, mirroring the methodology in Zouzoula et al. (ICPP'24), Figures 5 & 8:
baseline and SMM get the SAME total on-chip GLB budget, swept across sizes,
and we compare off-chip access volume + total simulated cycles.

Baseline config matches the paper's setup: 16x16 PE array, output-stationary
dataflow, USER bandwidth mode with 16 elements/cycle off-chip (both the
ifmap/filter SRAM bank bandwidth and the DRAM Bandwidth field), a 4kB fixed
ofmap buffer, and the remaining budget split ifmap:filter in 25:75 / 50:50 /
75:25 ratios (paper's sa_25_75 / sa_50_50 / sa_75_25).

The SMM side runs SMMScaleSimRunner directly as a library (not the CLI) so we
can pull REAL simulated per-layer report items (get_compute_report_items,
get_detail_report_items) off each layer's single_layer_sim object -- the
actual cycle-accurate simulated result, not SMM's own analytical self-estimate.

Both baseline and SMM runs share the exact same config-writing function, so
array size / dataflow / bandwidth are guaranteed identical between them.
"""
import os
import sys
import time
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scalesim.scale_sim import scalesim
from smm_scalesim_runner import SMMScaleSimRunner

ARRAY_DIM = 16      # paper: 16x16 PEs
DATAFLOW = "os"     # paper baseline: output stationary
BW = 16             # paper: 16 elements/cycle off-chip bandwidth
OFMAP_KB = 4        # paper: fixed 4kB ofmap buffer for all baseline configs

RATIO_MAP = {"25_75": (0.25, 0.75), "50_50": (0.5, 0.5), "75_25": (0.75, 0.25)}


def write_cfg(cfg_dir, run_name, ifmap_kb, filter_kb, ofmap_kb):
    path = os.path.join(cfg_dir, run_name + ".cfg")
    with open(path, "w") as f:
        f.write(f"""[general]
run_name = {run_name}

[architecture_presets]
ArrayHeight:    {ARRAY_DIM}
ArrayWidth:     {ARRAY_DIM}
IfmapSramSzkB:   {ifmap_kb}
FilterSramSzkB:  {filter_kb}
OfmapSramSzkB:   {ofmap_kb}
IfmapOffset:    0
FilterOffset:   10000000
OfmapOffset:    20000000
Bandwidth : {BW}
Dataflow : {DATAFLOW}
MemoryBanks:   1
ReadRequestBuffer: 32
WriteRequestBuffer: 32

[layout]
IfmapCustomLayout: False
IfmapSRAMBankBandwidth: {BW}
IfmapSRAMBankNum: 1
IfmapSRAMBankPort: 2
FilterCustomLayout: False
FilterSRAMBankBandwidth: {BW}
FilterSRAMBankNum: 1
FilterSRAMBankPort: 2

[sparsity]
SparsitySupport : false
SparseRep : ellpack_block
OptimizedMapping : false
BlockSize : 8
RandomNumberGeneratorSeed : 40

[run_presets]
InterfaceBandwidth: CALC
UseRamulatorTrace: False
""")
    return path


def sum_reports(out_dir):
    total_cycles = 0
    with open(os.path.join(out_dir, "COMPUTE_REPORT.csv")) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row or not row[0].strip():
                continue
            total_cycles += int(float(row[2]))   # "Total Cycles" column

    total_accesses = 0
    with open(os.path.join(out_dir, "DETAILED_ACCESS_REPORT.csv")) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row or not row[0].strip():
                continue
            total_accesses += int(float(row[11])) + int(float(row[14])) + int(float(row[17]))
    return total_cycles, total_accesses


def run_baseline(topology, layout, cfg_dir, results_dir, glb_kb, ratio_name):
    ifrac, ffrac = RATIO_MAP[ratio_name]
    remaining = glb_kb - OFMAP_KB
    ifmap_kb = max(1, round(remaining * ifrac))
    filter_kb = max(1, round(remaining * ffrac))
    run_name = f"base_{ratio_name}_{glb_kb}kb"
    cfg_path = write_cfg(cfg_dir, run_name, ifmap_kb, filter_kb, OFMAP_KB)

    t0 = time.time()
    s = scalesim(save_disk_space=True, verbose=False, config=cfg_path,
                 topology=topology, layout=layout, input_type_gemm=False)
    s.run_scale(top_path=results_dir)
    elapsed = time.time() - t0

    cycles, accesses = sum_reports(os.path.join(results_dir, run_name))
    return dict(kind=f"baseline_{ratio_name}", glb_kb=glb_kb, cycles=cycles,
                accesses=accesses, seconds=round(elapsed, 1))


def run_smm(topology, layout, cfg_dir, results_dir, glb_kb, objective):
    run_name = f"smm_{objective}_{glb_kb}kb"
    # Buffer sizes in this cfg are irrelevant -- SMM overrides them per layer.
    # Only array/dataflow/bandwidth matter, and they match write_cfg exactly.
    cfg_path = write_cfg(cfg_dir, run_name, 64, 64, OFMAP_KB)

    t0 = time.time()
    runner = SMMScaleSimRunner(
        topology_file=topology, config_file=cfg_path, glb_size_kb=glb_kb,
        objective=objective, homogeneous=False, allow_prefetch=True,
        output_dir=os.path.join(results_dir, run_name), verbose=False,
        save_ifmap_trace=False, save_filter_trace=False, save_ofmap_trace=False,
    )
    runner.run()
    elapsed = time.time() - t0

    total_cycles = 0
    total_accesses = 0
    for sim in runner.layer_sims:
        items = sim.get_compute_report_items()
        total_cycles += items[1]   # total_cycles (excl. prefetch overlap)
        d = sim.get_detail_report_items()
        total_accesses += d[11] + d[14] + d[17]

    return dict(kind=f"smm_{objective}", glb_kb=glb_kb, cycles=total_cycles,
                accesses=total_accesses, seconds=round(elapsed, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", required=True)
    ap.add_argument("--layout", default="./layouts/conv_nets/test.csv")
    ap.add_argument("--glb_sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--ratios", nargs="+", default=list(RATIO_MAP.keys()))
    ap.add_argument("--objectives", nargs="+", default=["accesses", "latency"])
    ap.add_argument("--scratch", required=True, help="scratch dir for temp cfgs/results")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    cfg_dir = os.path.join(args.scratch, "cfgs")
    results_dir = os.path.join(args.scratch, "results")
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Resume support: skip (kind, glb_kb) pairs already present in out_csv
    # (crash-safe -- rows are flushed to disk as soon as each run finishes).
    done = set()
    file_exists = os.path.exists(args.out_csv)
    if file_exists:
        with open(args.out_csv, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["kind"], row["glb_kb"]))

    fieldnames = ["kind", "glb_kb", "cycles", "accesses", "seconds"]
    out_f = open(args.out_csv, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        out_f.flush()

    def emit(res):
        key = (res["kind"], str(res["glb_kb"]))
        if key in done:
            print(f"  (skip, already done) {res['kind']} @ {res['glb_kb']}KB", flush=True)
            return
        writer.writerow(res)
        out_f.flush()
        done.add(key)
        print(f"  -> {res}", flush=True)

    t_start = time.time()
    for glb_kb in args.glb_sizes:
        for rname in args.ratios:
            key = (f"baseline_{rname}", str(glb_kb))
            if key in done:
                print(f"[baseline {rname} @ {glb_kb}KB] already done, skipping", flush=True)
                continue
            print(f"[baseline {rname} @ {glb_kb}KB] running...", flush=True)
            emit(run_baseline(args.topology, args.layout, cfg_dir, results_dir, glb_kb, rname))
        for obj in args.objectives:
            key = (f"smm_{obj}", str(glb_kb))
            if key in done:
                print(f"[smm {obj} @ {glb_kb}KB] already done, skipping", flush=True)
                continue
            print(f"[smm {obj} @ {glb_kb}KB] running...", flush=True)
            emit(run_smm(args.topology, args.layout, cfg_dir, results_dir, glb_kb, obj))

    out_f.close()
    print(f"\nTotal sweep time: {round(time.time() - t_start, 1)}s")
    print(f"Wrote {args.out_csv}")
