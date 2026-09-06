"""
Two small subclasses used by cosma/baseline.py's run_cosma_aware() to give
COSMA's memory-allocation plan a genuine effect on SCALE-Sim's own
simulation, instead of analytically adjusting the reported numbers after
the fact.

Each subclass changes behavior *only* in the one case where a higher-level
system (COSMA's ILP) has already determined the answer: a tensor asserted
resident needs no fetch; a tensor's own freshly-computed output that stays
on-chip needs no drain to DRAM. Every other code path -- every genuine
fetch, every genuine drain -- runs through the exact same, unmodified
logic as scalesim's own default memory system. See
cosma/ITERATION_HISTORY.md for the investigation that grounded this
design (which of SCALE-Sim's memory classes scale.cfg's CALC bandwidth
mode actually exercises, and exactly which counters feed the final
per-layer DRAM report).
"""
import math

import numpy as np

from scalesim.memory.read_buffer_estimate_bw import ReadBufferEstimateBw
from scalesim.memory.write_buffer import write_buffer


class CosmaResidentReadBuffer(ReadBufferEstimateBw):
    """
    Identical to ReadBufferEstimateBw unless fully_resident is set True --
    then every read is serviced as a pure on-chip hit (hit_latency added,
    exactly as SCALE-Sim already does for data it determines is in the
    active buffer) with zero backing-buffer (DRAM) traffic. Matches
    COSMA's whole-tensor-or-nothing residency model (never partial credit
    -- see cosma_Ilp.py's docstring on tiling).
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


class CosmaResidentWriteBuffer(write_buffer):
    """
    Identical to write_buffer unless stays_on_chip is set True -- then a
    freshly-computed output is still tracked through the normal on-chip
    SRAM write bookkeeping (so intra-layer buffer-full/free_space timing
    is unaffected), but when the buffer would normally drain to DRAM, that
    drain is treated as an instantaneous, zero-cost move within the chip
    instead: no backing_buffer.service_writes() call, no DRAM byte count.

    This matches COSMA's model exactly: Eq.3 of cosma_Ilp.py
    (`S[a,t] <= C(a,t-1) + P[a,t-1]`) proves a tensor can only be spilled
    *after* being resident for a prior timestep, never at its own
    creation instant, so a freshly-created tensor that stays resident
    never needs a DRAM round-trip.
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
