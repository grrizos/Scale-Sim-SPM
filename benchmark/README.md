# Vanilla vs. optimized SCALE-Sim benchmark

Compares this repo (`sim-opt` branch) against a vanilla upstream SCALE-Sim
clone across 10 models (googlenet, resnet18, resnet50, vit_b, dense_sparse,
vgg16, mobilenetv2, mnasnet, alexnet, squeezenet -- see models.py for
provenance: VGG16/MobileNetV2 built from cosma/_exported model.json,
MnasNet/SqueezeNet hand-derived from the real torchvision source since
their exports were unavailable/broken) and a 20-combo
parameter grid weighted toward USER interface bandwidth (15 array_size x
sram_kb corners under USER, 2 matched corners under CALC, plus the anchor
and 2 controls -- see combos.py) with a small dataflow addendum, a
cProfile-based function-time breakdown on a representative subset, and a
cycle-count correctness check. Each combo runs once per model/version (no
repeats) -- USER is where the vanilla-vs-optimized gap actually shows up
(see results.csv from the previous sweep), so CALC mostly stays at just the
anchor + control combos, with 2 grid corners added for a direct
CALC-vs-USER comparison at those specific array/sram points.

## Setup (do this first, on whichever machine runs the sweep)

Both repos must be `pip install -e .`-installed into **separate, dedicated
venvs** (`python3 -m venv`, not the system/default python, and not a venv
created with `--system-site-packages`). This isn't optional: a bare system
python can pick up an unrelated, possibly stale scalesim install from
`~/.local/lib/pythonX.Y/site-packages` ahead of the repo you actually meant
to run -- confirmed to happen on the machine this harness was built on.
`preflight.py` (below) checks for exactly this and will refuse to proceed
if a venv resolves to the wrong repo, but it's simplest to just always use
a fresh, isolated venv per repo:

```bash
python3 -m venv /path/to/vanilla_venv
/path/to/vanilla_venv/bin/pip install -e /path/to/vanilla/SCALE-Sim

python3 -m venv /path/to/optimized_venv
/path/to/optimized_venv/bin/pip install -e /path/to/this/SCALE-Sim   # this repo, sim-opt branch
```

## Run order

```bash
cd /path/to/this/SCALE-Sim   # this repo, sim-opt branch -- benchmark/ lives here

COMMON="--vanilla-repo /data/grizos/SCALE-Sim \
        --optimized-repo /data/grizos/Scale-Sim-SPM \
        --vanilla-venv-python /data/grizos/SCALE-Sim/venv/bin/python \
        --optimized-venv-python /data/grizos/Scale-Sim-SPM/venv/bin/python \
        --results-root /data/grizos/Scale-Sim-SPM/benchmarks/results"

# 1. Catches setup problems (bad paths, wrong venv, missing topology files,
#    insufficient disk space) before anything long-running starts.
python3 benchmark/preflight.py $COMMON

# 2. Runs the whole pipeline once end-to-end in ~2-4 minutes.
python3 benchmark/run_sweep.py $COMMON --smoke-test

# 3. Manually time the believed-worst corner (largest model, smallest SRAM,
#    largest array, vanilla) to sanity-check --timeout-s before committing
#    to the full sweep -- these corners are untested territory, the
#    documented numbers only cover the shipped-default config.
python3 benchmark/run_sweep.py $COMMON \
  --only model=googlenet,combo=grid_a64_s16_user,version=vanilla,repeat=1 \
  --timeout-s 3600

# 4. The full ~408-run sweep. Resumable -- safe to Ctrl-C and rerun the
#    same command; completed runs are skipped.
python3 benchmark/run_sweep.py $COMMON --timeout-s <set from step 3>

# 5. cProfile subset (8 runs) -- the "% of time per function" breakdown.
python3 benchmark/run_profile.py $COMMON

# 6. Pure post-processing, no re-running -- confirms vanilla and optimized
#    produced identical simulated cycle counts.
python3 benchmark/check_correctness.py --results-root /path/to/results

# 7. Static PNG charts from whatever's in results_raw.csv so far -- safe
#    to run mid-sweep too.
python3 benchmark/make_plots.py --results-root /path/to/results
```

## Output

- `results_raw.csv` -- one row per individual run (timing, cycle counts, full parameter columns).
- `profile_breakdown.csv` -- top-25 functions by cumulative time and by tottime, per profiled run.
- `correctness_check.csv` -- vanilla vs. optimized cycle-count diff per (model, combo).
- `plots/*.png` -- per-model speed bars, a speedup-% heatmap, the dataflow addendum, and profiling breakdowns.

See `/home/george/.claude/plans/i-need-to-have-agile-turing.md` for the full design rationale (why these parameters, why this grid size, what each script does).
