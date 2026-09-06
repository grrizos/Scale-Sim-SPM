# cosma/run_experiments.py
"""
Batch runner for run_cosma.py: runs the COSMA pipeline across one or more
models and one or more SPM budgets, and prints/saves a summary table.

A "model" input can be either:
  - a path to an already-exported model.json (used as-is), or
  - a path to a .tflite file, which gets exported to model.json first via
    the trim project's exporter (cached under --export-dir so repeat runs
    don't re-export).

Examples
--------
Single model, several budgets:
    python3 run_experiments.py --models model.json --budgets-kb 64 96 128 256

Several models (mix of .tflite and already-exported model.json), one budget:
    python3 run_experiments.py \\
        --models model.json \\
            /home/george/Desktop/trim/models/resnet20/cifar10/fp32.tflite \\
            /home/george/Desktop/trim/models/squeezenet_small/cifar10/fp32.tflite \\
        --budgets-kb 128 --out-csv results.csv

A failure on one (model, budget) pair (infeasible budget, unsupported op,
etc.) is recorded as an error row rather than aborting the whole batch.
"""
import argparse
import csv
import datetime
import os
import subprocess
import sys

import baseline
import cosma_Ilp
import graph_builder
import run_cosma

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(os.path.dirname(HERE), 'configs', 'scale.cfg')
DEFAULT_EXPORTER = '/home/george/Desktop/trim/python_scripts/export_model.py'
DEFAULT_EXPORT_DIR = os.path.join(HERE, '_exported')
DEFAULT_RESULTS_DIR = os.path.join(HERE, 'results')

RESULT_FIELDS = [
    'model', 'budget_kb', 'status',
    'total_ifmap_residency_credit_bytes', 'total_ofmap_residency_credit_bytes',
    'total_idealized_spill_bytes',
    'total_idealized_retrieve_bytes', 'total_real_retrieve_bytes',
    'baseline_dram_bytes', 'cosma_dram_bytes',
    'dram_traffic_reduction_pct', 'baseline_total_cycles', 'cosma_total_cycles',
    'speedup', 'error',
]


def resolve_model_json(model_input: str, exporter: str, export_dir: str,
                        force_export: bool) -> str:
    """Returns a path to a model.json, exporting from .tflite if needed."""
    if model_input.endswith('.json'):
        return model_input

    if not model_input.endswith('.tflite'):
        raise ValueError(f"Unrecognized model input (expected .json or "
                          f".tflite): {model_input}")

    # Derive a stable, collision-resistant cache name from the last two
    # path components (e.g. ".../mobilenet_v2_a035/cifar10/fp32.tflite"
    # -> "mobilenet_v2_a035_cifar10").
    parts = os.path.normpath(model_input).split(os.sep)
    name = '_'.join(parts[-3:-1]) if len(parts) >= 3 else parts[-2]
    out_dir = os.path.join(export_dir, name)
    model_json_path = os.path.join(out_dir, 'model.json')

    if force_export or not os.path.exists(model_json_path):
        if not os.path.exists(exporter):
            raise FileNotFoundError(
                f"Exporter not found at {exporter} -- pass --exporter to "
                f"point at trim/python_scripts/export_model.py, or export "
                f"{model_input} manually and pass the resulting model.json "
                f"directly instead."
            )
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(
            [sys.executable, exporter, '--model', model_input,
             '--out', out_dir, '--mode', 'fp32'],
            check=True, capture_output=True, text=True,
        )

    return model_json_path


def _error_row(model_input: str, budget_kb: float, exc: Exception) -> dict:
    row = {f: '' for f in RESULT_FIELDS}
    row['model'] = model_input
    row['budget_kb'] = budget_kb
    row['status'] = 'ERROR'
    row['error'] = f"{type(exc).__name__}: {exc}"
    return row


def _summary_row(model_input: str, budget_kb: float, summary: dict) -> dict:
    row = {f: '' for f in RESULT_FIELDS}
    row['model'] = model_input
    row['budget_kb'] = budget_kb
    row.update({
        'status': summary['status'],
        'total_ifmap_residency_credit_bytes': summary['total_ifmap_residency_credit_bytes'],
        'total_ofmap_residency_credit_bytes': summary['total_ofmap_residency_credit_bytes'],
        'total_idealized_spill_bytes': summary['total_idealized_spill_bytes'],
        'total_idealized_retrieve_bytes': summary['total_idealized_retrieve_bytes'],
        'total_real_retrieve_bytes': summary['total_real_retrieve_bytes'],
        'baseline_dram_bytes': summary['baseline_dram_bytes'],
        'cosma_dram_bytes': summary['cosma_dram_bytes'],
        'dram_traffic_reduction_pct':
            round(summary['dram_traffic_reduction_pct'], 2),
        'baseline_total_cycles': round(summary['baseline_total_cycles'], 1),
        'cosma_total_cycles': round(summary['cosma_total_cycles'], 1),
        'speedup': round(summary['speedup'], 4),
    })
    return row


def run_model_sweep(model_input: str, budgets_kb: list, config_path: str,
                     exporter: str, export_dir: str, force_export: bool,
                     time_limit_sec: float) -> list:
    """
    Runs every budget in budgets_kb for one model, running SCALE-Sim's
    baseline at most once (it doesn't depend on the budget, so it's
    reused across the whole sweep -- see run_cosma.py's layer_stats
    param) and skipping it entirely for budgets that fail the fast,
    baseline-independent tensor-fits-at-all check.
    """
    try:
        model_json_path = resolve_model_json(
            model_input, exporter, export_dir, force_export)
        _, tensors = graph_builder.load_graph(model_json_path)
    except Exception as e:  # noqa: BLE001 -- one bad model shouldn't abort the batch
        return [_error_row(model_input, b, e) for b in budgets_kb]

    rows = []
    runnable_budgets = []
    for budget_kb in budgets_kb:
        try:
            cosma_Ilp.assert_tensors_fit_budget(tensors, int(budget_kb * 1024))
            runnable_budgets.append(budget_kb)
        except AssertionError as e:
            rows.append(_error_row(model_input, budget_kb, e))

    if not runnable_budgets:
        return rows

    try:
        print(f"  running SCALE-Sim baseline for {model_input} ...", file=sys.stderr)
        layer_stats = baseline.run_baseline(model_json_path, config_path)
    except Exception as e:  # noqa: BLE001
        rows.extend(_error_row(model_input, b, e) for b in runnable_budgets)
        return rows

    for budget_kb in runnable_budgets:
        print(f"Running: model={model_input} budget={budget_kb}KB ...", file=sys.stderr)
        try:
            summary = run_cosma.run_cosma(
                model_json_path=model_json_path,
                config_path=config_path,
                memory_budget_bytes=int(budget_kb * 1024),
                ilp_time_limit_sec=time_limit_sec,
                layer_stats=layer_stats,
                verbose=False,
            )
            rows.append(_summary_row(model_input, budget_kb, summary))
        except Exception as e:  # noqa: BLE001
            rows.append(_error_row(model_input, budget_kb, e))

    return rows


def print_table(rows: list) -> None:
    cols = ['model', 'budget_kb', 'status',
            'total_ifmap_residency_credit_bytes', 'total_ofmap_residency_credit_bytes',
            'total_idealized_spill_bytes', 'total_real_retrieve_bytes',
            'dram_traffic_reduction_pct', 'speedup', 'error']
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = '  '.join(c.ljust(widths[c]) for c in cols)
    print(header)
    print('  '.join('-' * widths[c] for c in cols))
    for r in rows:
        print('  '.join(str(r[c]).ljust(widths[c]) for c in cols))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', required=True,
                         help='One or more paths to model.json or .tflite files.')
    parser.add_argument('--budgets-kb', nargs='+', type=float, default=[64, 128, 256],
                         help='One or more SPM budgets in KB (default: 64 128 256).')
    parser.add_argument('--config', default=DEFAULT_CONFIG,
                         help='SCALE-Sim hardware config file.')
    parser.add_argument('--exporter', default=DEFAULT_EXPORTER,
                         help='Path to trim/python_scripts/export_model.py '
                              '(only needed for .tflite inputs).')
    parser.add_argument('--export-dir', default=DEFAULT_EXPORT_DIR,
                         help='Cache directory for .tflite -> model.json exports.')
    parser.add_argument('--force-export', action='store_true',
                         help='Re-export even if a cached model.json exists.')
    parser.add_argument('--time-limit', type=float, default=120,
                         help='CBC solve time limit per (model, budget) run, seconds.')
    parser.add_argument('--out-csv', default=None,
                         help='Path to write the results table as CSV. Defaults to '
                              'a timestamped file under cosma/results/ -- results '
                              'are always saved, this only lets you pick where.')
    parser.add_argument('--no-save', action='store_true',
                         help='Skip writing a results file; print only.')
    args = parser.parse_args()

    rows = []
    for model_input in args.models:
        rows.extend(run_model_sweep(
            model_input, args.budgets_kb, args.config, args.exporter,
            args.export_dir, args.force_export, args.time_limit,
        ))

    print_table(rows)

    if not args.no_save:
        out_csv = args.out_csv
        if out_csv is None:
            os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            out_csv = os.path.join(DEFAULT_RESULTS_DIR, f'run_{timestamp}.csv')
        else:
            out_dir = os.path.dirname(out_csv)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_csv}")


if __name__ == '__main__':
    main()
