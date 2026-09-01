"""
smm_scalesim_runner.py
======================
Integrates the SMM scratchpad-management policies (Zouzoula et al., ICPP '24)
into SCALE-Sim so that every layer is simulated with the optimal GLB partition
chosen by Algorithm 1.

How it works
------------
1.  Before SCALE-Sim runs a layer, SMM's Algorithm 1 picks the best policy
    (P1–P5 / Intra) given the total GLB size you configure.
2.  The chosen policy determines EXACTLY how many bytes to allocate for
    ifmap, filter, and ofmap in the scratchpad (optionally doubled for
    double-buffering / prefetch).
3.  Those byte counts are injected into SCALE-Sim's `double_buffered_scratchpad`
    *before* `service_memory_requests` is called, so SCALE-Sim simulates the
    real access pattern under the SMM-managed memory.
4.  After all layers, a summary is printed showing the chosen policy,
    partition, accesses, and latency for every layer.

Usage
-----
    from smm_scalesim_runner import SMMScaleSimRunner

    runner = SMMScaleSimRunner(
        topology_file = "topologies/resnet18.csv",
        config_file   = "configs/scale.cfg",
        glb_size_kb   = 64,           # total on-chip Global Buffer
        objective     = "accesses",   # or "latency"
        homogeneous   = False,        # True = one policy for all layers
        allow_prefetch= True,
        output_dir    = "outputs/smm_run",
    )
    runner.run()
    runner.print_summary()
"""

from __future__ import annotations

import os
import math
from typing import List, Optional

import numpy as np

# SCALE-Sim imports
from scalesim.scale_config    import scale_config as ScaleConfig
from scalesim.topology_utils  import topologies   as Topologies
from scalesim.layout_utils    import layouts       as Layouts
from scalesim.single_layer_sim import single_layer_sim
from scalesim.memory.double_buffered_scratchpad_mem import double_buffered_scratchpad

# SMM imports (same directory)
from smm_policy_selector import (
    LayerSpec, HwParams, LayerPlan, Objective, Policy, POLICY_NAMES,
    layer_spec_from_scalesim_row, plan_network,
)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class SMMScaleSimRunner:
    """
    Drop-in replacement for scalesim.scale_sim that pre-plans every layer
    with the SMM policy engine before handing off to SCALE-Sim's memory model.
    """

    def __init__(
        self,
        topology_file:  str,
        config_file:    str,
        glb_size_kb:    int   = 64,
        objective:      str   = "accesses",   # "accesses" | "latency"
        homogeneous:    bool  = False,
        allow_prefetch: bool  = True,
        output_dir:     str   = "outputs/smm_run",
        verbose:        bool  = True,
        save_ifmap_trace: bool = True,
        save_filter_trace: bool = True,
        save_ofmap_trace: bool = True,
    ):
        self.topology_file  = topology_file
        self.config_file    = config_file
        self.glb_size_bytes = glb_size_kb * 1024
        self.objective      = Objective.Accesses if objective == "accesses" else Objective.Latency
        self.homogeneous    = homogeneous
        self.allow_prefetch = allow_prefetch
        self.output_dir     = output_dir
        self.verbose        = verbose
        self.save_ifmap_trace  = save_ifmap_trace
        self.save_filter_trace = save_filter_trace
        self.save_ofmap_trace  = save_ofmap_trace

        # SCALE-Sim objects
        self.config = ScaleConfig()
        self.config.read_conf_file(config_file)          # ← correct method name

        self.topo   = Topologies()
        self.topo.load_arrays(topofile=topology_file, mnk_inputs=False)

        self.layout = Layouts()

        # HwParams built from SCALE-Sim config
        arr_rows, arr_cols = self.config.get_array_dims()
        try:
            bw_user = self.config.use_user_dram_bandwidth()
        except Exception:
            bw_user = False
        bw = (float(self.config.get_bandwidths_as_list()[0])
              if bw_user else 16.0)
        self.hw = HwParams(
            bytes_per_elem     = 1,
            mac_per_cycle      = float(arr_rows * arr_cols),
            bw_bytes_per_cycle = bw,
        )

        # Results
        self.layer_specs:  List[LayerSpec]  = []
        self.smm_plans:    List[LayerPlan]  = []
        self.layer_sims:   List[single_layer_sim] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_layer_specs(self) -> List[LayerSpec]:
        """
        topo_arrays row layout (from SCALE-Sim load_arrays_conv):
          idx 0   : layer name (string)
          idx 1,2 : IFMAP H, W
          idx 3,4 : Filter H, W
          idx 5   : Channels (CI)
          idx 6   : Num Filters (Fn)
          idx 7   : Stride row (we use this; col stride at idx 8 is auto-added)
          idx 9+  : sparsity cols (ignored)
        SCALE-Sim topology CSV has no padding column; padding is handled
        by the operand matrix generator, not stored in the topology.
        """
        specs = []
        num_layers = self.topo.get_num_layers()
        for lid in range(num_layers):
            r = self.topo.get_layer_params(lid)   # list of strings/values
            name = str(r[0]).strip()
            spec = LayerSpec(
                name = name,
                IH   = int(r[1]),  IW = int(r[2]),
                FH   = int(r[3]),  FW = int(r[4]),
                CI   = int(r[5]),
                Fn   = int(r[6]),
                S    = int(r[7]),
                P    = 0,
            )
            specs.append(spec)
        return specs

    def _make_memory_system(self, plan: LayerPlan, layer_id: int) -> double_buffered_scratchpad:
        """
        Build and configure a SCALE-Sim double_buffered_scratchpad
        with buffer sizes dictated by the SMM plan.
        """
        mem = double_buffered_scratchpad()

        try:
            user_bw = self.config.use_user_dram_bandwidth()
        except Exception:
            user_bw = False

        bw_list = self.config.get_bandwidths_as_list()
        if user_bw:
            ifmap_bw  = getattr(self.config, 'ifmap_sram_bank_bandwidth',  10)
            filter_bw = getattr(self.config, 'filter_sram_bank_bandwidth', 10)
            ofmap_bw  = bw_list[0]
            est_bw    = False
        else:
            _, arr_c  = self.config.get_array_dims()
            ifmap_bw  = filter_bw = 16
            ofmap_bw  = arr_c
            est_bw    = True

        ifmap_bank_num  = getattr(self.config, 'ifmap_sram_bank_num',   1)
        ifmap_bank_port = getattr(self.config, 'ifmap_sram_bank_port',  2)
        filter_bank_num  = getattr(self.config, 'filter_sram_bank_num',  1)
        filter_bank_port = getattr(self.config, 'filter_sram_bank_port', 2)

        mem.set_params(
            layer_id              = layer_id,
            word_size             = self.hw.bytes_per_elem,
            # ---- KEY: SMM-chosen partition sizes ----
            ifmap_buf_size_bytes  = max(plan.ifmap_bytes,  self.hw.bytes_per_elem),
            filter_buf_size_bytes = max(plan.filter_bytes, self.hw.bytes_per_elem),
            ofmap_buf_size_bytes  = max(plan.ofmap_bytes,  self.hw.bytes_per_elem),
            # -----------------------------------------
            rd_buf_active_frac    = 0.5,
            wr_buf_active_frac    = 0.5,
            ifmap_backing_buf_bw  = ifmap_bw,
            filter_backing_buf_bw = filter_bw,
            ofmap_backing_buf_bw  = ofmap_bw,
            verbose               = self.verbose,
            estimate_bandwidth_mode = est_bw,
            ifmap_sram_bank_num   = ifmap_bank_num,
            ifmap_sram_bank_port  = ifmap_bank_port,
            filter_sram_bank_num  = filter_bank_num,
            filter_sram_bank_port = filter_bank_port,
            config                = self.config,
            topo                  = self.topo,
        )
        return mem

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        """Run all layers with SMM-optimal memory partitioning."""
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Build LayerSpecs from topology
        self.layer_specs = self._build_layer_specs()

        # 2. Run SMM Algorithm 1 across all layers
        self.smm_plans = plan_network(
            layers         = self.layer_specs,
            glb_bytes      = self.glb_size_bytes,
            hw             = self.hw,
            objective      = self.objective,
            allow_prefetch = self.allow_prefetch,
            homogeneous    = self.homogeneous,
        )

        # 3. Simulate each layer in SCALE-Sim with the SMM partition
        num_layers = self.topo.get_num_layers()
        for lid in range(num_layers):
            plan = self.smm_plans[lid]
            spec = self.layer_specs[lid]

            if self.verbose:
                print(f"\nRunning Layer {lid}")
                print(f"[SMM] Layer {lid:3d} ({spec.name:20s}): "
                      f"{POLICY_NAMES[plan.policy]:30s}  "
                      f"GLB {plan.memory/1024:6.1f} kB  "
                      f"off-chip {plan.accesses/1024:8.1f} kB")

            sim = single_layer_sim()
            sim.set_params(
                layer_id     = lid,
                config_obj   = self.config,
                topology_obj = self.topo,
                layout_obj   = self.layout,
                verbose      = self.verbose,
            )

            if plan.feasible:
                mem_sys = self._make_memory_system(plan, lid)
                sim.set_memory_system(mem_sys)
            # If infeasible (layer too big for any policy), SCALE-Sim uses its default sizing

            sim.run()

            if self.verbose:
                total_cycles, comp_cycles, stall_cycles, util, mapping_eff, compute_util = \
                    sim.get_compute_report_items()
                print('Total cycles: ' + str(total_cycles))
                print('Compute cycles: ' + str(comp_cycles))
                print('Stall cycles: ' + str(stall_cycles))
                print('Overall utilization: ' + "{:.2f}".format(util) + '%')
                print('Mapping efficiency: ' + "{:.2f}".format(mapping_eff) + '%')

                bw_items = sim.get_bandwidth_report_items()
                if self.config.sparsity_support is True:
                    (avg_ifmap_sram_bw, avg_filter_sram_bw, avg_filter_metadata_sram_bw,
                     avg_ofmap_sram_bw, avg_ifmap_dram_bw, avg_filter_dram_bw,
                     avg_ofmap_dram_bw) = bw_items
                else:
                    (avg_ifmap_sram_bw, avg_filter_sram_bw, avg_ofmap_sram_bw,
                     avg_ifmap_dram_bw, avg_filter_dram_bw, avg_ofmap_dram_bw) = bw_items

                print('Average IFMAP SRAM BW: ' + "{:.3f}".format(avg_ifmap_sram_bw) +
                      ' words/cycle')
                print('Average Filter SRAM BW: ' + "{:.3f}".format(avg_filter_sram_bw) +
                      ' words/cycle')
                if self.config.sparsity_support is True:
                    print('Average Filter Metadata SRAM BW: ' +
                          "{:.3f}".format(avg_filter_metadata_sram_bw) + ' words/cycle')
                print('Average OFMAP SRAM BW: ' + "{:.3f}".format(avg_ofmap_sram_bw) +
                      ' words/cycle')
                print('Average IFMAP DRAM BW: ' + "{:.3f}".format(avg_ifmap_dram_bw) +
                      ' words/cycle')
                print('Average Filter DRAM BW: ' + "{:.3f}".format(avg_filter_dram_bw) +
                      ' words/cycle')
                print('Average OFMAP DRAM BW: ' + "{:.3f}".format(avg_ofmap_dram_bw) +
                      ' words/cycle')

                print('Saving traces: ', end='')
            sim.save_traces(self.output_dir,
                            save_ifmap_trace=self.save_ifmap_trace,
                            save_filter_trace=self.save_filter_trace,
                            save_ofmap_trace=self.save_ofmap_trace)
            if self.verbose:
                print('Done!')

            self.layer_sims.append(sim)

        print(f"\n[SMM] Simulation complete. Traces in: {self.output_dir}\n")

    def print_summary(self):
        """Print a per-layer policy summary table."""
        KB = 1024.0
        MB = 1024.0 * 1024.0
        print("=" * 90)
        print(f"{'Layer':<22} {'Policy':<32} {'n':>4} {'pf':>3} "
              f"{'GLB(kB)':>9} {'OffChip(kB)':>12} {'Lat(Mc)':>10}")
        print("-" * 90)

        total_acc = 0
        total_lat = 0.0
        for spec, plan in zip(self.layer_specs, self.smm_plans):
            tag = "✓" if plan.feasible else "!"
            print(f"{spec.name:<22} {POLICY_NAMES[plan.policy]:<32} "
                  f"{plan.n if plan.n else '-':>4} "
                  f"{'+p' if plan.prefetch else '-':>3} "
                  f"{plan.memory/KB:>9.1f} "
                  f"{plan.accesses/KB:>12.1f} "
                  f"{plan.latency/1e6:>10.3f}  {tag}")
            total_acc += plan.accesses
            total_lat += plan.latency

        print("=" * 90)
        print(f"{'TOTAL':<22} {'':<32} {'':>4} {'':>3} "
              f"{'':>9} {total_acc/MB:>11.2f}M {total_lat/1e6:>10.3f}M")
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="SCALE-Sim + SMM memory management")
    p.add_argument("--topology",  required=True,  help="SCALE-Sim topology CSV")
    p.add_argument("--config",    required=True,  help="SCALE-Sim config file (.cfg)")
    p.add_argument("--glb_kb",    type=int, default=64, help="Total GLB size in kB (default 64)")
    p.add_argument("--objective", choices=["accesses","latency"], default="accesses")
    p.add_argument("--homogeneous", action="store_true", help="Use one policy for all layers")
    p.add_argument("--no_prefetch", action="store_true", help="Disable double-buffering candidates")
    p.add_argument("--output",    default="outputs/smm_run")
    p.add_argument("--no_ifmap_trace",  action="store_true", help="Skip writing IFMAP trace CSVs")
    p.add_argument("--no_filter_trace", action="store_true", help="Skip writing FILTER trace CSVs")
    p.add_argument("--no_ofmap_trace",  action="store_true", help="Skip writing OFMAP trace CSVs")
    args = p.parse_args()

    runner = SMMScaleSimRunner(
        topology_file  = args.topology,
        config_file    = args.config,
        glb_size_kb    = args.glb_kb,
        objective      = args.objective,
        homogeneous    = args.homogeneous,
        allow_prefetch = not args.no_prefetch,
        output_dir     = args.output,
        verbose        = True,
        save_ifmap_trace  = not args.no_ifmap_trace,
        save_filter_trace = not args.no_filter_trace,
        save_ofmap_trace  = not args.no_ofmap_trace,
    )
    runner.run()
    runner.print_summary()