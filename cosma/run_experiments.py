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

Besides the summary CSV, a full run_cosma.py-style verbose report is also
saved per (model, budget) combination by default, under a timestamped
directory in cosma/logs/ (see --logs-dir/--no-logs) -- the CSV is the
compact cross-combination view, these logs are the same per-run detail
`run_cosma.py` alone would print for that one combination.
"""
import argparse
import contextlib
import csv
import datetime
import io
import os
import sys

from helpers import baseline
from helpers import cosma_Ilp
from helpers import graph_builder
from helpers import model_resolver
import run_cosma

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(os.path.dirname(HERE), 'configs', 'scale.cfg')
DEFAULT_EXPORTER = model_resolver.DEFAULT_EXPORTER
DEFAULT_EXPORT_DIR = model_resolver.DEFAULT_EXPORT_DIR
DEFAULT_RESULTS_DIR = os.path.join(HERE, 'results')
DEFAULT_LOGS_DIR = os.path.join(HERE, 'logs')

RESULT_FIELDS = [
    'model', 'budget_kb', 'status',
    'total_ifmap_residency_credit_bytes', 'total_ofmap_residency_credit_bytes',
    'total_idealized_spill_bytes',
    'total_idealized_retrieve_bytes', 'total_real_retrieve_bytes',
    'baseline_dram_bytes', 'cosma_dram_bytes',
    'dram_traffic_reduction_pct', 'baseline_total_cycles', 'cosma_total_cycles',
    'speedup', 'error',
]


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


def _array_dims_tag(config_path: str) -> str:
    """
    '<ArrayHeight>x<ArrayWidth>' from a scale.cfg, e.g. '64x64'. Included
    in log filenames specifically because array size is the config knob
    most likely to get swept for the same model/budget (see
    ITERATION_HISTORY.md's array-size discussion, and the "change the
    array size and watch the cycle numbers move" verification workflow)
    -- without it, two runs that only differ by array size would silently
    overwrite each other's log.
    """
    from scalesim.scale_config import scale_config
    config = scale_config()
    config.read_conf_file(config_path)
    arr_row, arr_col = config.get_array_dims()
    return f"{arr_row}x{arr_col}"


def _log_file_name(model_json_path: str, budget_kb: float, config_tag: str) -> str:
    """<model-parent-dir-name>_<budget>kb_<array-dims>.log -- same naming
    spirit as visualize_spm.default_out_path(), so a log and its matching
    plot are easy to pair up by eye, plus the array-dims tag so re-running
    the same (model, budget) at a different array size doesn't overwrite
    the previous log."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(model_json_path)))
    stem = os.path.splitext(os.path.basename(model_json_path))[0]
    name = parent if parent and parent != 'cosma' else stem
    budget_str = f"{budget_kb:.3f}".rstrip('0').rstrip('.').replace('.', 'p')
    return f"{name}_{budget_str}kb_{config_tag}.log"


def _run_and_log(logs_dir: str, log_name: str, model_input: str, budget_kb: float,
                  **run_cosma_kwargs) -> dict:
    """
    Calls run_cosma.run_cosma(verbose=True) with stdout captured instead
    of printed live (so a big sweep doesn't flood the terminal with every
    run's full report), and -- if logs_dir is set -- writes that captured
    text to logs_dir/log_name, the same detail level run_cosma.py alone
    would show for this exact (model, budget). Re-raises on failure after
    still saving whatever was captured (including the traceback), so a
    failing combination leaves a log to look at too.
    """
    buf = io.StringIO()
    header = f"model: {model_input}\nbudget: {budget_kb} KB\n{'=' * 60}\n"
    try:
        with contextlib.redirect_stdout(buf):
            summary = run_cosma.run_cosma(verbose=True, **run_cosma_kwargs)
        return summary
    except Exception as e:
        buf.write(f"\n{type(e).__name__}: {e}\n")
        raise
    finally:
        if logs_dir:
            os.makedirs(logs_dir, exist_ok=True)
            with open(os.path.join(logs_dir, log_name), 'w') as f:
                f.write(header + buf.getvalue())


def run_model_sweep(model_input: str, budgets_kb: list, config_path: str,
                     exporter: str, export_dir: str, force_export: bool,
                     time_limit_sec: float = None, logs_dir: str = None,
                     save_plots: bool = True) -> list:
    """
    Runs every budget in budgets_kb for one model, running SCALE-Sim's
    baseline at most once (it doesn't depend on the budget, so it's
    reused across the whole sweep -- see run_cosma.py's layer_stats
    param) and skipping it entirely for budgets that fail the fast,
    baseline-independent tensor-fits-at-all check.

    logs_dir: if set, saves run_cosma.py's full verbose report (the same
        detail `run_cosma.py` alone prints) for every (model, budget)
        combination -- one file per combination, including failures.

    save_plots: if True (default), also saves run_cosma.py's baseline-vs-
        COSMA occupancy PNG for every combination, to the same
        cosma/spm_plots/ directory a standalone run_cosma.py call would
        use (run_cosma()'s own default naming already tags it with this
        sweep's array size -- see run_cosma.py's _array_dims_tag()).
    """
    try:
        model_json_path = model_resolver.resolve_model_json(
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

    config_tag = _array_dims_tag(config_path)

    for budget_kb in runnable_budgets:
        print(f"Running: model={model_input} budget={budget_kb}KB ...", file=sys.stderr)
        try:
            summary = _run_and_log(
                logs_dir, _log_file_name(model_json_path, budget_kb, config_tag),
                model_input, budget_kb,
                model_json_path=model_json_path,
                config_path=config_path,
                memory_budget_bytes=int(budget_kb * 1024),
                ilp_time_limit_sec=time_limit_sec,
                layer_stats=layer_stats,
                save_plot=save_plots,
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
    parser.add_argument('--time-limit', type=float, default=None,
                         help='CBC solve time limit per (model, budget) run, seconds. ')
    parser.add_argument('--out-csv', default=None,
                         help='Path to write the results table as CSV. Defaults to '
                              'a timestamped file under cosma/results/ -- results '
                              'are always saved, this only lets you pick where.')
    parser.add_argument('--no-save', action='store_true',
                         help='Skip writing a results file; print only.')
    parser.add_argument('--logs-dir', default=None,
                         help='Directory to save one run_cosma.py-style verbose report '
                              'per (model, budget) combination. Defaults to a timestamped '
                              'directory under cosma/logs/ -- saved by default, this only '
                              'lets you pick where.')
    parser.add_argument('--no-logs', action='store_true',
                         help='Skip saving per-combination verbose logs.')
    parser.add_argument('--no-plots', action='store_true',
                         help="Skip saving each combination's baseline-vs-COSMA "
                              "occupancy PNG (saved to cosma/spm_plots/ by default).")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config_tag = _array_dims_tag(args.config)

    logs_dir = None
    if not args.no_logs:
        logs_dir = args.logs_dir or os.path.join(DEFAULT_LOGS_DIR, f'run_{timestamp}')

    rows = []
    for model_input in args.models:
        rows.extend(run_model_sweep(
            model_input, args.budgets_kb, args.config, args.exporter,
            args.export_dir, args.force_export, args.time_limit,
            logs_dir=logs_dir, save_plots=not args.no_plots,
        ))

    print_table(rows)

    if logs_dir and os.path.isdir(logs_dir):
        # The directory is only actually created (helpers via
        # _run_and_log()'s finally block) once at least one combination
        # got far enough to call run_cosma() -- e.g. a sweep where every
        # budget fails the fast pre-check never creates it, and this
        # message would otherwise falsely claim logs were written.
        print(f"\nWrote per-combination logs to {logs_dir}/")

    if not args.no_save:
        out_csv = args.out_csv
        if out_csv is None:
            os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
            # Same <timestamp>_<array> naming spirit as the logs/plots --
            # see _log_file_name()/run_cosma.py's _array_dims_tag().
            out_csv = os.path.join(DEFAULT_RESULTS_DIR, f'run_{timestamp}_{config_tag}.csv')
        else:
            out_dir = os.path.dirname(out_csv)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == '__main__':
    main()
