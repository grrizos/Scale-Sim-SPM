# spm_common/model_resolver.py
"""
Resolves a model input -- either an already-exported model.json, or a raw
.tflite file -- to a model.json path, auto-exporting via the trim
project's exporter and caching the result under export_dir when given a
.tflite. Used directly (not duplicated) by both cosma/run_cosma.py-family
scripts and onsram/run_onsram.py, since exporting/caching a .tflite has no
algorithm-specific logic at all -- see spm_common/__init__.py.
"""
import os
import subprocess
import sys

# The export cache itself stays physically under cosma/_exported/ (a large,
# gitignored, regenerable directory -- see .gitignore) rather than moving
# alongside this file to spm_common/_exported/: it predates this module's
# move out of cosma/helpers/, both papers already read/write it at this one
# location (onsram/run_onsram.py's resolve_model_arg() hardcodes the same
# path independently), and moving ~1.6GB of cached exports for a purely
# cosmetic match would trade a real, working shared cache for no benefit.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXPORTER = '/home/george/Desktop/trim/python_scripts/export_model.py'
DEFAULT_EXPORT_DIR = os.path.join(_REPO_ROOT, 'cosma', '_exported')


def resolve_model_json(model_input: str, exporter: str = DEFAULT_EXPORTER,
                        export_dir: str = DEFAULT_EXPORT_DIR,
                        force_export: bool = False) -> str:
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
