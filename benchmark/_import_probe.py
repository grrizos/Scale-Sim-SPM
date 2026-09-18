"""Tiny helper invoked as a plain script (never via -c) by preflight.py
and run_sweep.py's static-meta capture, so scalesim resolution goes
through the same site-packages/editable-install path a real
`python3 <repo>/scalesim/scale.py` subprocess would use -- not the
cwd-priority shortcut that "-c" (and "-m") get, which can mask a venv
pointing at the wrong (or a stale, non-editable) scalesim install.
Confirmed empirically necessary: this machine has a stale non-editable
scalesim copy in ~/.local/lib/python3.10/site-packages that a plain
"python3 script.py" subprocess picks up ahead of an editable install
sitting in global site-packages, even though "python3 -c ..." from the
same cwd looked fine.
"""
import platform

import numpy
import scalesim

print(scalesim.__file__)
print(numpy.__version__)
print(platform.python_version())
print(platform.processor() or platform.machine())
