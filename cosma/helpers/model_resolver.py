# cosma/helpers/model_resolver.py
"""
Resolves a model input -- either an already-exported model.json, or a raw
.tflite file -- to a model.json path, auto-exporting via the trim
project's exporter and caching the result under export_dir when given a
.tflite. Shared by run_cosma.py and run_experiments.py so both accept
either input kind the same way (originally only run_experiments.py had
this; factored out here so run_cosma.py doesn't have to require an
already-exported model.json).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # cosma/
DEFAULT_EXPORTER = '/home/george/Desktop/trim/python_scripts/export_model.py'
DEFAULT_EXPORT_DIR = os.path.join(HERE, '_exported')


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
