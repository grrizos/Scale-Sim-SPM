# onsram/onsram_helpers/resident_buffers.py
"""
Two small subclasses used by scale_sim_runner.py's run_onsram_aware() to
give OnSRAM's own pinning plan a genuine effect on SCALE-Sim's own
simulation, instead of analytically adjusting the reported numbers after
the fact -- a self-contained duplicate of
scalesim/memory/cosma_resident_buffers.py's CosmaResidentReadBuffer/
CosmaResidentWriteBuffer, NOT an import of them. Despite living in the
shared scalesim/ package, that module is COSMA's own (named and documented
as such); OnSRAM keeps its own copy so a change to COSMA's version can
never change OnSRAM's simulated behavior, or vice versa. The underlying
logic is generic (it only reacts to a boolean flag the caller sets, with
no assumption about how that flag's value was decided -- ILP or greedy
heuristic), so a byte-for-byte duplicate is exact and safe.

Each subclass changes behavior *only* in the one case where a higher-level
system (OnSRAM's greedy pinning decision) has already determined the
answer: a tensor asserted resident needs no fetch; a tensor's own
freshly-computed output that stays on-chip needs no drain to DRAM. Every
other code path -- every genuine fetch, every genuine drain -- runs
through the exact same, unmodified logic as scalesim's own default memory
system.
"""
import math

import numpy as np

from scalesim.memory.read_buffer_estimate_bw import ReadBufferEstimateBw
from scalesim.memory.write_buffer import write_buffer


class OnsramResidentReadBuffer(ReadBufferEstimateBw):
    """
    Identical to ReadBufferEstimateBw unless fully_resident is set True --
    then every read is serviced as a pure on-chip hit (hit_latency added,
    exactly as SCALE-Sim already does for data it determines is in the
    active buffer) with zero backing-buffer (DRAM) traffic. Matches
    OnSRAM's own whole-tensor-or-nothing residency model (never partial
    credit -- see onsram/docs/onsram_integration_plan.md sec 2).
    """

    def __init__(self):
        super().__init__()
        self.fully_resident = False

    def service_reads(self, incoming_requests_arr_np, incoming_cycles_arr):
        if self.fully_resident:
            if not self.first_request_seen:
                self.first_request_rcvd_cycle = int(incoming_cycles_arr[0][0])
                self.first_request_seen = True
            return incoming_cycles_arr + self.hit_latency
        return super().service_reads(incoming_requests_arr_np, incoming_cycles_arr)

    def complete_all_prefetches(self):
        if self.fully_resident:
            self.trace_valid = True
            return
        super().complete_all_prefetches()

    def get_num_accesses(self):
        if self.fully_resident:
            return 0
        return super().get_num_accesses()

    def get_external_access_start_stop_cycles(self):
        if self.fully_resident:
            c = self.first_request_rcvd_cycle
            return c, c
        return super().get_external_access_start_stop_cycles()


class OnsramResidentWriteBuffer(write_buffer):
    """
    Identical to write_buffer unless stays_on_chip is set True -- then a
    freshly-computed output is still tracked through the normal on-chip
    SRAM write bookkeeping (so intra-layer buffer-full/free_space timing
    is unaffected), but when the buffer would normally drain to DRAM, that
    drain is treated as an instantaneous, zero-cost move within the chip
    instead: no backing_buffer.service_writes() call, no DRAM byte count.

    This matches the same architectural assumption COSMA's own model uses
    (Eq.3: a tensor can only be spilled after being resident for a prior
    timestep, never at its own creation instant) -- a freshly-created
    tensor that stays resident never needs a DRAM round-trip. OnSRAM's own
    resident_action never spills at all (whole-lifetime pinning is
    all-or-nothing), so for OnSRAM this is simply "every layer's own
    output is free at creation," unconditionally.
    """

    def __init__(self):
        super().__init__()
        self.stays_on_chip = False

    def empty_drain_buf(self, empty_start_cycle=0):
        if not self.stays_on_chip:
            return super().empty_drain_buf(empty_start_cycle)

        # Mirrors the parent's real bookkeeping (which rows are being
        # freed, how much data) exactly, so on-chip capacity/timing for
        # subsequent writes in this same layer behaves consistently --
        # only the backing_buffer (DRAM) call and num_access charge are
        # skipped, since this data never actually leaves the chip.
        lines_to_fill_dbuf = int(math.ceil(self.drain_buf_size / self.req_gen_bandwidth))
        self.drain_buf_end_line_id = self.drain_buf_start_line_id + lines_to_fill_dbuf
        self.drain_buf_end_line_id = min(self.drain_buf_end_line_id, self.trace_matrix.shape[0])

        requests_arr_np = \
            self.trace_matrix[self.drain_buf_start_line_id:self.drain_buf_end_line_id, :]
        num_lines = requests_arr_np.shape[0]
        data_sz_to_drain = num_lines * requests_arr_np.shape[1]
        for elem in requests_arr_np[-1, :]:
            if elem == -1:
                data_sz_to_drain -= 1

        cycles_this_drain = np.full((num_lines, 1), empty_start_cycle)
        if not self.trace_valid:
            self.cycles_vec = cycles_this_drain
            self.trace_valid = True
        else:
            self.cycles_vec = np.concatenate((self.cycles_vec, cycles_this_drain), axis=0)

        self.free_space += data_sz_to_drain
        self.drain_buf_start_line_id = self.drain_buf_end_line_id
        return empty_start_cycle

    def get_num_accesses(self):
        if self.stays_on_chip:
            return 0
        return super().get_num_accesses()

    def get_external_access_start_stop_cycles(self):
        if self.stays_on_chip:
            return 0, 0
        return super().get_external_access_start_stop_cycles()
