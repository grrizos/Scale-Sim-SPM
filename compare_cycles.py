#!/usr/bin/env python3
"""Compare total-cycle counts between two SCALE-Sim run logs, layer by layer.

Usage:
    python3 compare_cycles.py <baseline_output.txt> <smm_output.txt>
"""

import re
import sys


def parse_cycles(filepath):
    """Extract (layer_index, total_cycles) pairs from a scale.py stdout log."""
    layer_re = re.compile(r'^Running Layer (\d+)')
    cycles_re = re.compile(r'^Total cycles:\s*(\d+)')

    cycles = []
    current_layer = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            m = layer_re.match(line)
            if m:
                current_layer = int(m.group(1))
                continue
            m = cycles_re.match(line)
            if m and current_layer is not None:
                cycles.append((current_layer, int(m.group(1))))
                current_layer = None
    return cycles


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <baseline.txt> <smm.txt>")
        sys.exit(1)

    baseline_path, smm_path = sys.argv[1], sys.argv[2]
    baseline = parse_cycles(baseline_path)
    smm = parse_cycles(smm_path)

    if len(baseline) != len(smm):
        print(f"WARNING: layer count mismatch — baseline has {len(baseline)} layers, "
              f"SMM has {len(smm)} layers. Comparing by position up to the shorter one.\n")

    n = min(len(baseline), len(smm))
    header = f"{'Layer':>6} | {'Baseline':>12} | {'SMM':>12} | {'Saved':>12} | {'Saved %':>8}"
    print(header)
    print('-' * len(header))

    total_base = 0
    total_smm = 0
    for i in range(n):
        layer_idx, base_cycles = baseline[i]
        _, smm_cycles = smm[i]
        saved = base_cycles - smm_cycles
        pct = (saved / base_cycles * 100) if base_cycles else 0
        total_base += base_cycles
        total_smm += smm_cycles
        print(f"{layer_idx:>6} | {base_cycles:>12} | {smm_cycles:>12} | {saved:>12} | {pct:>7.2f}%")

    print('-' * len(header))
    total_saved = total_base - total_smm
    total_pct = (total_saved / total_base * 100) if total_base else 0
    print(f"{'TOTAL':>6} | {total_base:>12} | {total_smm:>12} | {total_saved:>12} | {total_pct:>7.2f}%")


if __name__ == '__main__':
    main()
