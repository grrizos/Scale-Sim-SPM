# onsram/run_paper_reproduction.py
"""
Reproduces the OnSRAM paper's own headline experiment (Fig. 7, OnSRAM-Static
vs No SPM Mgmt) as closely as this port's available models and SCALE-Sim's
own modeling allow. A thin, additive wrapper around run_onsram.py's existing
run_onsram()/run_onsram_scale_sim() -- no new SPM-management logic here,
just paper-matched defaults and a comparison printout. Mirrors
cosma/run_paper_baselines.py's own role (an additive sibling script that
reproduces a paper's own reported experiment through this project's real
SCALE-Sim pipeline) and its isolation convention: small orchestration/
logging helpers below (heartbeat, model-name, log-file writing) are
deliberately duplicated from run_onsram.py's own private (leading-
underscore) versions rather than importing them -- see run_onsram.py's own
module docstring for why private internals aren't a stable contract to
reach into.

Paper: "OnSRAM: Efficient Inter-Node On-Chip Scratchpad Management in Deep
Learning Accelerators" (S. Pal, S. Venkataramani, V. Srinivasan, K.
Gopalakrishnan; ACM TECS 2022, DOI 10.1145/3530909). Confirmed directly
against the primary PDF (~/Downloads/OnSRAM_...pdf), Abstract and Section 6
"Experimental Methodology":
  - Accelerator: "a 3 TFLOP dense 2D-systolic array and a 375 GFLOP FP16
    SIMD special function unit array, supported by a 2 MB on-chip SPM with
    a bandwidth of 384 GBps, and a 32 GBps external memory." Batch size 1.
  - 12 benchmark DNNs: AlexNet, VGG16, GoogLeNet, Inception-v3, Inception-v4,
    ResNet-50, SSD300, ResNeXt, MobileNetV1, SqueezeNet, PTB-LSTM (an LSTM
    language model), Multi-Head Attention (a transformer).
  - Headline (Fig. 7): OnSRAM-Static achieves 1.02-4.8x speedup vs "No SPM
    Mgmt" across those 12 models. The GeoMean bar and most per-model bars
    have no printed numeric label in the figure -- only two Infinite-SPM
    peaks are annotated (3.86/3.81 for the ResNeXt cluster, 5.17/4.76 for
    MobileNetV1/SqueezeNet's cluster). This script therefore prints OUR
    measured per-model numbers next to that textual 1.02-4.8x range, never
    a fabricated per-model paper ground truth we don't actually have.
  - Sec 7.1: "For sequential DNNs like AlexNet, VGG, and PTB, there is very
    little gap to be bridged between No SPM Mgmt and Infinite SPM" -- i.e.
    VGG16 is explicitly expected to show only a small speedup, not a
    modeling shortfall if it comes out close to 1.0x.

What this reproduction can and can't match:
  - EXACT: 2MB SPM budget (--spm-mb default), 32GBps external DRAM
    bandwidth, batch size 1 (this pipeline always simulates one image at a
    time, by construction).
  - A DISCLOSED, already-considered choice, not derived fresh by this
    script: configs/scale_onsram.cfg (pre-existing in this repo, run_name
    'onsram_paper_hw') is used as the default --config. The paper's own
    performance model (Sec 6) is a closed-form roofline bound --
    max(compute_time, data_xfer_time), with compute_time = FLOPs / 3e12
    directly -- it never simulates a concrete systolic array shape or
    clock at all, so there is no array shape to literally "match". That
    config resolves this the only reasonable way available: 39x39 PEs at
    an implicit 1GHz clock, 2 FLOPs/MAC -> 2*39*39*1e9 = 3.042 TFLOP/s,
    matching the paper's 3 TFLOP figure to within 1.4%; the 32/384 GBps
    figures map onto that same 1GHz assumption as 32/384 bytes per cycle
    (Bandwidth/IfmapSRAMBankBandwidth/FilterSRAMBankBandwidth are already
    set to 32 in that file -- the 384 GBps on-chip SPM figure has no
    corresponding knob in this pipeline, since a resident SPM hit is
    modeled as a flat, bandwidth-independent hit_latency cost, not a
    bandwidth-limited transfer -- see onsram_helpers/resident_buffers.py).
    Because SCALE-Sim genuinely simulates array fill/drain/dataflow reuse
    and the paper's own model doesn't, expect the same QUALITATIVE pattern
    (memory-bound mobile networks benefit most; AlexNet/VGG/PTB see little
    gap), not byte-identical numbers -- the same standard this project's
    COSMA paper comparison already holds itself to (see
    onsram/docs/onsram_integration_plan.md Phase F).
  - NOT reproduced: 7 of the paper's 12 models have no model.json export
    in this repo (AlexNet, GoogLeNet, Inception-v4, SSD300, ResNeXt,
    PTB-LSTM, Multi-Head Attention) -- the last three in particular aren't
    even image classifiers (object detection / LSTM / transformer
    attention), a different op mix than this pipeline's conv-focused
    tensor tracking is built for. PAPER_MODELS below lists the 5 that DO
    have a real export: VGG16, ResNet-50, Inception-v3, MobileNetV1,
    SqueezeNet.

Real SCALE-Sim is genuinely slow, same caveat as run_onsram.py's own
module docstring -- and this config's 39x39 array plus Inception-v3's 129
conv-like layers is untested territory as of this writing (this port's own
15-minute-plus-per-budget experience this session was at a 16x16 array on
much smaller models). Calibrate on one small model first
(--model squeezenet, or --no-scale-sim for a decision-only sanity pass)
before launching the full 5-model sweep.
"""
import argparse
import contextlib
import csv
import os
import sys
import threading
import time
import traceback

_ONSRAM_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_ONSRAM_DIR)
if _ONSRAM_DIR not in sys.path:
    sys.path.insert(0, _ONSRAM_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from run_onsram import (
    run_onsram, run_onsram_scale_sim, check_physical_validity,
    print_dram_savings, print_detailed_report, resolve_model_arg,
)
from onsram_helpers import scale_sim_runner

DEFAULT_CONFIG = os.path.join(_REPO_ROOT, 'configs', 'scale_onsram.cfg')
DEFAULT_SPM_MB = 2.0  # the paper's own headline SPM size (Sec 6)
DEFAULT_LOGS_DIR = os.path.join(_ONSRAM_DIR, 'logs')
DEFAULT_RESULTS_DIR = os.path.join(_ONSRAM_DIR, 'results')

# repo model arg (resolves under cosma/_exported/<name>/model.json) -> the
# paper's own name for that network, in the order Fig. 7 lists them.
PAPER_MODELS = {
    'VGG16': 'VGG16',
    'ResNeT50': 'ResNet-50',
    '_exported_inception_v3-tflite-float': 'Inception-v3',
    'MobileNet': 'MobileNetV1',
    'MobileNetV2': 'MobileNetV2',
    # NOT 'squeezenet' -- cosma/_exported/squeezenet/model.json is corrupt
    # (all 40 layers mislabeled 'ADD', zero real convolutions; confirmed
    # its own raw source .tflite is itself degenerate -- re-exporting it
    # hits "RuntimeError: multiple const inputs not supported"). This is
    # squeezenet1_1.tflite's own export instead (a different, valid raw
    # source under /home/george/trim/models/), confirmed to produce a real
    # SqueezeNet1.1 graph (CONV2D x26, CONCAT x8 for fire-module expand
    # branches, MAXPOOL x3; 4.94MB of params, matching SqueezeNet1.1's
    # known ~1.24M-parameter size almost exactly).
    'squeezenet1_1': 'SqueezeNet',
}

# Sec 7.1 / Fig. 7's own stated range for OnSRAM-Static vs No SPM Mgmt,
# printed as context -- never compared row-by-row against a specific
# model, since the paper doesn't publish per-model numeric labels (see
# module docstring).
PAPER_SPEEDUP_RANGE = (1.02, 4.8)

def _default_bandwidth_words_per_cycle(config_path: str) -> float:
    """Duplicated from run_onsram.py's own private helper of the same name
    -- see module docstring for why this file duplicates rather than
    imports private internals."""
    from scalesim.scale_config import scale_config
    config = scale_config()
    config.read_conf_file(config_path)
    if config.use_user_dram_bandwidth():
        return float(config.get_bandwidths_as_list()[0])
    _, arr_col = config.get_array_dims()
    return float(arr_col)


_HEARTBEAT_INTERVAL_SEC = 20


def _start_heartbeat(label: str) -> threading.Event:
    """Duplicated from run_onsram.py's own _start_heartbeat() -- see
    module docstring for why this file doesn't import it directly."""
    stop = threading.Event()
    start = time.monotonic()

    def _beat():
        while not stop.wait(_HEARTBEAT_INTERVAL_SEC):
            elapsed = time.monotonic() - start
            print(f"[heartbeat] still running: {label} elapsed={elapsed:.0f}s", file=sys.stderr)

    threading.Thread(target=_beat, daemon=True).start()
    return stop


def _run_one(model_arg: str, paper_name: str, model_json_path: str, config_path: str,
             memory_budget_bytes: int, logs_dir, run_scale_sim: bool) -> dict:
    """
    Runs one (model, budget) combination -- decision phase, physical-
    validity replay, and (unless run_scale_sim is False) the real
    SCALE-Sim baseline + OnSRAM-aware passes -- and returns a flat summary
    dict. Full verbose report goes to a per-combination log file (same
    '<model>_<budget>MB.log' convention as run_onsram.py's own
    _run_and_log(), duplicated rather than imported -- see module
    docstring), terminal gets one compact progress line.
    """
    budget_str = f"{memory_budget_bytes / 1024 / 1024:g}".replace('.', 'p')
    log_name = f"{model_arg}_{budget_str}MB_paper.log"
    log_path = os.path.join(logs_dir, log_name) if logs_dir else None
    log_file = open(log_path, 'w', buffering=1) if log_path else None
    try:
        cm = contextlib.redirect_stdout(log_file) if log_file is not None else contextlib.nullcontext()
        with cm:
            if log_file is not None:
                log_file.write(f"model: {model_json_path}\npaper_name: {paper_name}\n"
                                f"spm_budget_mb: {memory_budget_bytes / 1024 / 1024:.4f}\n"
                                f"config: {config_path}\n{'=' * 60}\n")
            result = run_onsram(model_json_path, memory_budget_bytes, verbose=True)
            stats = check_physical_validity(result, memory_budget_bytes, verbose=True)
            print_detailed_report(result, memory_budget_bytes)

            if run_scale_sim:
                heartbeat_stop = _start_heartbeat(
                    f"SCALE-Sim baseline+aware for {paper_name} @ "
                    f"{memory_budget_bytes / 1024 / 1024:.2f} MB")
                try:
                    layer_stats = scale_sim_runner.run_baseline(model_json_path, config_path, verbose=False)
                    bandwidth_bytes_per_cycle = _default_bandwidth_words_per_cycle(config_path)
                    scale_sim_result = run_onsram_scale_sim(
                        result, model_json_path, config_path, memory_budget_bytes,
                        layer_stats, bandwidth_bytes_per_cycle, verbose=True)
                finally:
                    heartbeat_stop.set()
                print_dram_savings(scale_sim_result)
                stats = {**stats, **scale_sim_result}
        return {'status': 'OK', **stats}
    except Exception as e:  # noqa: BLE001 -- one bad model shouldn't abort the sweep
        if log_file is not None:
            log_file.write(f"\n{traceback.format_exc()}\n")
        return {'status': 'ERROR', 'error': f"{type(e).__name__}: {e}"}
    finally:
        if log_file is not None:
            log_file.close()


def run_paper_reproduction(models: dict = None, spm_mb_list=None, config_path: str = DEFAULT_CONFIG,
                            run_scale_sim: bool = True, logs_dir=DEFAULT_LOGS_DIR,
                            verbose: bool = True) -> list:
    """
    Runs every (model, budget) combination in `models` x `spm_mb_list`
    through OnSRAM-Static + (unless run_scale_sim is False) real SCALE-Sim,
    using configs/scale_onsram.cfg's paper-matched hardware by default.
    Returns a list of flat row dicts, one per combination, in the same
    shape written to --out-csv.
    """
    models = models or PAPER_MODELS
    spm_mb_list = spm_mb_list or [DEFAULT_SPM_MB]
    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)

    rows = []
    for model_arg, paper_name in models.items():
        try:
            model_json_path = resolve_model_arg(model_arg)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED to resolve model={model_arg}: {type(e).__name__}: {e}", flush=True)
            rows.extend({'model': model_arg, 'paper_name': paper_name, 'spm_mb': mb,
                         'status': 'ERROR', 'error': f"{type(e).__name__}: {e}"}
                        for mb in spm_mb_list)
            continue

        for spm_mb in spm_mb_list:
            memory_budget_bytes = int(spm_mb * 1024 * 1024)
            if verbose:
                print(f"Running: paper_model={paper_name} ({model_arg}) "
                      f"spm={spm_mb:g}MB ...", file=sys.stderr, flush=True)
            stats = _run_one(model_arg, paper_name, model_json_path, config_path,
                              memory_budget_bytes, logs_dir, run_scale_sim)
            row = {'model': model_arg, 'paper_name': paper_name, 'spm_mb': spm_mb, **stats}
            rows.append(row)
            if stats['status'] == 'OK':
                dram_note = (f", DRAM -{stats['dram_traffic_reduction_pct']:.1f}%, "
                              f"speedup {stats['speedup']:.3f}x"
                              if 'dram_traffic_reduction_pct' in stats else "")
                print(f"  {paper_name}: {stats['pinned_count']}/{stats['total_tensors']} pinned, "
                      f"peak {stats['peak_bytes'] / 1024 / 1024:.4f} MB / {spm_mb:g} MB{dram_note}",
                      flush=True)
            else:
                print(f"  {paper_name}: FAILED: {stats['error']}", flush=True)

    return rows


def print_summary(rows: list) -> None:
    lo, hi = PAPER_SPEEDUP_RANGE
    print(f"\n--- OnSRAM paper's own reported range (Fig. 7, OnSRAM-Static vs No SPM Mgmt): "
          f"{lo}-{hi}x, across the paper's full 12-model suite ---")
    print(f"--- Our measured speedup, {len(set(r['model'] for r in rows))} of the paper's 12 models "
          f"(the ones with a real model.json export in this repo) ---\n")
    cols = ['paper_name', 'model', 'spm_mb', 'status', 'pinned', 'total_tensors',
            'dram_reduction_pct', 'speedup', 'error']
    print_rows = [{
        'paper_name': r['paper_name'], 'model': r['model'], 'spm_mb': r['spm_mb'],
        'status': r['status'],
        'pinned': r.get('pinned_count', ''), 'total_tensors': r.get('total_tensors', ''),
        'dram_reduction_pct': round(r['dram_traffic_reduction_pct'], 2) if 'dram_traffic_reduction_pct' in r else '',
        'speedup': round(r['speedup'], 4) if 'speedup' in r else '',
        'error': r.get('error', ''),
    } for r in rows]
    widths = {c: max(len(c), *(len(str(row[c])) for row in print_rows)) for c in cols}
    print('  '.join(c.ljust(widths[c]) for c in cols))
    print('  '.join('-' * widths[c] for c in cols))
    for row in print_rows:
        print('  '.join(str(row[c]).ljust(widths[c]) for c in cols))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', action='append', dest='models', default=None,
                         help="Restrict to one paper model (repeatable), by its repo model arg "
                              f"(one of {list(PAPER_MODELS)}). Defaults to all 5.")
    parser.add_argument('--spm-mb', type=float, nargs='+', default=[DEFAULT_SPM_MB],
                         help=f"SPM budget(s) in MB (default: the paper's own {DEFAULT_SPM_MB} MB).")
    parser.add_argument('--config', default=DEFAULT_CONFIG,
                         help=f"SCALE-Sim hardware config (default: {DEFAULT_CONFIG}, the "
                              "paper-matched 39x39/32GBps/384GBps hardware -- see module docstring).")
    parser.add_argument('--no-scale-sim', action='store_true',
                         help="Skip the real SCALE-Sim passes -- fast decision-only sanity check "
                              "(pinning %%, physical validity), no DRAM/speedup numbers.")
    parser.add_argument('--logs-dir', default=DEFAULT_LOGS_DIR)
    parser.add_argument('--out-csv', default=os.path.join(DEFAULT_RESULTS_DIR, 'paper_reproduction.csv'))
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        sys.exit(f"error: --config file not found: {args.config}")

    selected = ({m: PAPER_MODELS[m] for m in args.models} if args.models else PAPER_MODELS)
    unknown = [m for m in (args.models or []) if m not in PAPER_MODELS]
    if unknown:
        sys.exit(f"error: unknown --model {unknown}, must be one of {list(PAPER_MODELS)}")

    rows = run_paper_reproduction(
        models=selected, spm_mb_list=args.spm_mb, config_path=args.config,
        run_scale_sim=not args.no_scale_sim, logs_dir=args.logs_dir, verbose=True)
    print_summary(rows)

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = ['paper_name', 'model', 'spm_mb', 'status', 'pinned_count', 'total_tensors',
                  'peak_bytes', 'oversized_count', 'dram_traffic_reduction_pct', 'speedup', 'error']
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out_csv}")
