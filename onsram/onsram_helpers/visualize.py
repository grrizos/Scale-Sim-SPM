# onsram/onsram_helpers/visualize.py
"""
SPM occupancy-over-time visualization for OnSRAM's pinning plan.

Self-contained -- does NOT import cosma/visualize_spm.py's private
(leading-underscore) helpers, since that file is a top-level script, not
a helpers/ module with a stable public contract to depend on. But it
deliberately mirrors that file's visual grammar (address-vs-time
rectangles sized by real byte count, one color per tensor id, same
palette) so a COSMA plot and an OnSRAM plot read the same way side by
side -- this whole port exists to make the two algorithms comparable.
"""
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

_TENSOR_PALETTE = [
    '#4c78a8', '#54a24b', '#72b7b2', '#b279a2',
    '#9d755d', '#5b8c85', '#7a6fa6', '#3d6b8f',
]
_GRID_COLOR = '#e1e0d9'
_SPINE_COLOR = '#c3c2b7'
_TICK_COLOR = '#898781'
_LABEL_COLOR = '#57564f'
_BUDGET_COLOR = '#d03b3b'
_PEAK_COLOR = '#57564f'
_OCCUPIED_FILL = '#4c78a8'
_OCCUPIED_LINE = '#3d6b8f'


def _style_axes(ax) -> None:
    ax.grid(True, color=_GRID_COLOR, linewidth=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(_SPINE_COLOR)
    ax.tick_params(colors=_TICK_COLOR, labelsize=7)


def _spm_plan_to_runs(spm_plan: Dict[Tuple[int, int], int]) -> Dict[int, List[Tuple[int, int, int]]]:
    """
    tensor_id -> [(t_start, t_end_inclusive, address), ...]. A new run
    starts whenever the next timestep isn't exactly prev+1 or the address
    changes -- OnSRAM's build_resident_action() never re-addresses a
    pinned tensor mid-life, so in practice a gap and an address change
    always coincide, but both are checked for robustness.
    """
    by_tensor: Dict[int, List[int]] = {}
    for (a, t) in spm_plan:
        by_tensor.setdefault(a, []).append(t)

    runs: Dict[int, List[Tuple[int, int, int]]] = {}
    for a, ts in by_tensor.items():
        ts.sort()
        tensor_runs = []
        run_start = run_addr = prev_t = None
        for t in ts:
            addr = spm_plan[(a, t)]
            if run_start is None:
                run_start, run_addr, prev_t = t, addr, t
            elif t == prev_t + 1 and addr == run_addr:
                prev_t = t
            else:
                tensor_runs.append((run_start, prev_t, run_addr))
                run_start, run_addr, prev_t = t, addr, t
        tensor_runs.append((run_start, prev_t, run_addr))
        runs[a] = tensor_runs
    return runs


def _occupancy_trace(tensors, resident_action, spm_plan, memory_budget_bytes: int,
                      schedule_len: int) -> List[int]:
    """
    Replays a fresh SpmAllocator one timestep at a time (not just
    replay_all()) so occupied_bytes() can be read after every single
    step -- gives the full "SPM space through time" envelope, not just
    the overall peak. A second, independent physical-consistency check
    for free, on top of the one run_onsram.py's own validation already does.
    """
    from helpers.spm_allocator import SpmAllocator  # COSMA's, unmodified
    allocator = SpmAllocator(tensors, spm_plan, resident_action, memory_budget_bytes)
    occupied_by_ts = []
    for t in range(schedule_len):
        allocator.step(t)
        occupied_by_ts.append(allocator.occupied_bytes())
    return occupied_by_ts


def plot_spm_usage(tensors, resident_action: Dict[Tuple[int, int], str],
                    spm_plan: Dict[Tuple[int, int], int], memory_budget_bytes: int,
                    schedule_len: int, out_path: str, title: str = None) -> None:
    """
    Two stacked panels in one PNG:
      - top: total occupied bytes vs. timestep (the envelope), with the
        budget and the achieved peak drawn as reference lines -- answers
        "how much of the SPM is in use, and when."
      - bottom: per-tensor address occupancy over time (a Gantt-style
        view, one colored rectangle per pinned tensor's residency
        episode) -- answers "which tensor is where, for how long."
    Never calls plt.show() -- always saves to out_path.
    """
    occupied_by_ts = _occupancy_trace(tensors, resident_action, spm_plan,
                                       memory_budget_bytes, schedule_len)
    peak_bytes = max(occupied_by_ts) if occupied_by_ts else 0
    ylim_bytes = memory_budget_bytes * 1.08

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(max(8.0, schedule_len * 0.18), 7.0), sharex=True,
        gridspec_kw={'height_ratios': [1, 2.2]})

    ts = list(range(schedule_len))
    ax_top.fill_between(ts, occupied_by_ts, step='post', color=_OCCUPIED_FILL, alpha=0.45, zorder=2)
    ax_top.plot(ts, occupied_by_ts, drawstyle='steps-post', color=_OCCUPIED_LINE,
                linewidth=1.2, zorder=3)
    ax_top.axhline(memory_budget_bytes, color=_BUDGET_COLOR, linestyle='--', linewidth=1.2,
                    label=f'budget ({memory_budget_bytes / 1024 / 1024:.2f} MB)', zorder=4)
    ax_top.axhline(peak_bytes, color=_PEAK_COLOR, linestyle=':', linewidth=1.0,
                    label=f'peak ({peak_bytes / 1024 / 1024:.4f} MB)', zorder=4)
    ax_top.set_ylim(0, ylim_bytes)
    ax_top.set_ylabel('occupied bytes', fontsize=8)
    ax_top.legend(fontsize=7, loc='upper right')
    ax_top.set_title(title or 'SPM occupancy over time', fontsize=10, color=_LABEL_COLOR)
    _style_axes(ax_top)

    runs_by_tensor = _spm_plan_to_runs(spm_plan)
    for a, runs in runs_by_tensor.items():
        color = _TENSOR_PALETTE[a % len(_TENSOR_PALETTE)]
        size = tensors[a].size_bytes
        for (t_start, t_end, addr) in runs:
            width = t_end - t_start + 1
            rect = patches.Rectangle((t_start, addr), width, size, facecolor=color,
                                      alpha=0.75, edgecolor='#3a3a38', linewidth=0.6, zorder=2)
            ax_bottom.add_patch(rect)
            if size >= memory_budget_bytes * 0.03:
                ax_bottom.text(t_start + width / 2, addr + size / 2, f"t{a}",
                                ha='center', va='center', fontsize=6, color='white', zorder=3)

    ax_bottom.axhline(memory_budget_bytes, color=_BUDGET_COLOR, linestyle='--', linewidth=1.2, zorder=4)
    ax_bottom.set_xlim(0, schedule_len)
    ax_bottom.set_ylim(0, ylim_bytes)
    ax_bottom.set_xlabel('timestep')
    ax_bottom.set_ylabel('SPM address (bytes)', fontsize=8)
    ax_bottom.set_title('Per-tensor address occupancy (pinned tensors only)', fontsize=9,
                        color=_LABEL_COLOR)
    _style_axes(ax_bottom)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
