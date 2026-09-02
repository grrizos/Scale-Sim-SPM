"""
The `write_buffer` class simulates the memory operations of the OFMAP SRAM for a double-buffered
write system in systolic array-based computations.
"""
# TODO: Verification Pending
import time
import math
import numpy as np
# import matplotlib.pyplot as plt
from scalesim.memory.write_port import write_port


class write_buffer:
    """
    Class which runs the memory simulation of the OFMAP SRAM.
    """
    #
    def __init__(self):
        """
        __init__ method.
        """
        # Buffer properties: User specified
        self.total_size_bytes = 128
        self.word_size = 1
        self.active_buf_frac = 0.9

        # Buffer properties: Calculated
        self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
        self.active_buf_size = int(math.ceil(self.total_size_elems * self.active_buf_frac))
        # A buffer this tiny (e.g. an SMM policy sizing it down to ~1 element) can round its entire
        # capacity into the active half, leaving nothing to drain -- which deadlocks the very first
        # write. Guarantee at least 1 element of drain room whenever there's any capacity at all.
        self.drain_buf_size = max(1, self.total_size_elems - self.active_buf_size)

        # Backing interface properties
        self.backing_buffer = write_port()
        self.req_gen_bandwidth = 100

        # Status of the buffer
        self.free_space = self.total_size_elems
        self.drain_buf_start_line_id = 0
        self.drain_buf_end_line_id = 0

        # Helper data structures for faster execution
        self.line_idx = 0
        self.current_line = np.ones((1, 1)) * -1
        self.max_cache_lines = 2 ** 10  # TODO: This is arbitrary, check if this can be tuned
        self.trace_matrix_cache = np.zeros((1, 1))

        # Access counts
        self.num_access = 0

        # Trace matrix
        self.trace_matrix = np.zeros((1, 1))
        self.cycles_vec = np.zeros((1, 1))
        # Backing storage for the amortized-growable trace_matrix (see _append_chunk_to_trace_matrix).
        # trace_matrix is kept as a live view (_trace_buf[:_trace_len]) after every append, since
        # empty_drain_buf() reads trace_matrix mid-simulation (every cycle the buffer drains) --
        # unlike the read-buffer classes, this one can't defer building the trace to the very end.
        self._trace_buf = None
        self._trace_len = 0

        # Flags
        # This variable determines where the new requests should be buffered
        # 0: Directly in the drain buffer
        # 1: In the active buffer, while the drain buffer is flushed
        self.state = 0
        self.drain_end_cycle = 0

        self.trace_valid = False
        # Fixing ISSUE #10
        self.trace_matrix_cache_empty = True
        self.trace_matrix_empty = True

    #
    def set_params(self, backing_buf_obj,
                   total_size_bytes=128, word_size=1, active_buf_frac=0.9,
                   backing_buf_bw=100
                   ):
        """
        Method to set the ofmap memory simulation parameters for housekeeping.
        """
        self.total_size_bytes = total_size_bytes
        self.word_size = word_size

        assert 0.5 <= active_buf_frac < 1, "Valid active buf frac [0.5,1)"
        self.active_buf_frac = active_buf_frac

        self.backing_buffer = backing_buf_obj
        self.req_gen_bandwidth = backing_buf_bw

        self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
        self.active_buf_size = int(math.ceil(self.total_size_elems * self.active_buf_frac))
        # See __init__ for why this needs a floor of 1.
        self.drain_buf_size = max(1, self.total_size_elems - self.active_buf_size)
        self.free_space = self.total_size_elems

    #
    def reset(self):
        """
        Method to reset the write buffer parameters.
        """
        self.total_size_bytes = 128
        self.word_size = 1
        self.active_buf_frac = 0.9

        self.backing_buffer = write_buffer()
        self.req_gen_bandwidth = 100

        self.free_space = self.total_size_elems
        self.drain_end_cycle = 0

        self.trace_matrix = np.zeros((1, 1))
        self._trace_buf = None
        self._trace_len = 0

        self.num_access = 0
        self.state = 0

        self.trace_valid = False
        # Fixing ISSUE #10
        self.trace_matrix_cache_empty = True
        self.trace_matrix_empty = True

    #
    def store_to_trace_mat_cache(self, elem):
        """
        Method to add the incoming element to the trace matrix cache.
        """
        if elem == -1:
            return

        if self.current_line.shape == (1,1):    # This line is empty
            self.current_line = np.ones((1, self.req_gen_bandwidth)) * -1

        self.current_line[0, self.line_idx] = elem
        self.line_idx += 1
        self.free_space -= 1

        if not self.line_idx < self.req_gen_bandwidth:
            # Store to the cache matrix
            # Fixing ISSUE #10
            # if self.trace_matrix_cache.shape == (1,1):
            if self.trace_matrix_cache_empty:
                self.trace_matrix_cache = self.current_line
                self.trace_matrix_cache_empty = False
            else:
                self.trace_matrix_cache = np.concatenate((
                                                          self.trace_matrix_cache,
                                                          self.current_line
                                                         ),
                                                         axis=0)

            self.current_line = np.ones((1,1)) * -1
            self.line_idx = 0

            if not self.trace_matrix_cache.shape[0] < self.max_cache_lines:
                self.append_to_trace_mat()

    #
    def append_to_trace_mat(self, force=False):
        """
        Method to append to the trace matrix cache.
        """
        if force:
            # This forces the contents for self.current_line and self.trace_matrix
            # cache to be dumped
            if not self.line_idx == 0:
                #if self.trace_matrix_cache.shape == (1,1):
                if self.trace_matrix_cache_empty:
                    self.trace_matrix_cache = self.current_line
                    self.trace_matrix_cache_empty = False
                else:
                    self.trace_matrix_cache = np.concatenate((
                                                              self.trace_matrix_cache,
                                                              self.current_line
                                                             ),
                                                             axis=0)

                self.current_line = np.ones((1,1)) * -1
                self.line_idx = 0
        # Fixing ISSUE #10
        # if self.trace_matrix_cache.shape == (1,1):
        if self.trace_matrix_cache_empty:
            return

        #if self.trace_matrix.shape == (1,1):
        if self.trace_matrix_empty:
            self.drain_buf_start_line_id = 0
            self.trace_matrix_empty = False

        self._append_chunk_to_trace_matrix(self.trace_matrix_cache)

        self.trace_matrix_cache = np.zeros((1,1))
        # Fixing ISSUE #10
        self.trace_matrix_cache_empty = True

    #
    def _append_chunk_to_trace_matrix(self, chunk):
        """
        Method to append a chunk of rows to trace_matrix using an amortized-growable backing array
        instead of concatenating onto trace_matrix directly. Concatenating on every single drain
        event copies the entire trace built so far every time (O(n^2) total across n drains) --
        small on-chip buffers (e.g. from SMM policies) drain far more often than SCALE-Sim's default
        buffer sizes, which makes that cost dominate. Growing this backing array geometrically instead
        means reallocation+copy only happens O(log n) times, amortizing to O(n) total.

        trace_matrix is reassigned to a fresh view of the backing array after every call, so it stays
        a valid, correctly-sized, up-to-date array at all times -- empty_drain_buf()/empty_all_buffers()
        keep reading it exactly as before, since they always read trace_matrix fresh rather than
        holding on to a reference from before this call.
        """
        num_rows, num_cols = chunk.shape

        if self._trace_buf is None:
            init_capacity = max(num_rows, self.max_cache_lines)
            self._trace_buf = np.empty((init_capacity, num_cols))
        elif self._trace_len + num_rows > self._trace_buf.shape[0]:
            new_capacity = self._trace_buf.shape[0]
            while new_capacity < self._trace_len + num_rows:
                new_capacity *= 2
            grown = np.empty((new_capacity, num_cols))
            grown[:self._trace_len] = self._trace_buf[:self._trace_len]
            self._trace_buf = grown

        self._trace_buf[self._trace_len:self._trace_len + num_rows] = chunk
        self._trace_len += num_rows
        self.trace_matrix = self._trace_buf[:self._trace_len]

    #
    def service_writes(self, incoming_requests_arr_np, incoming_cycles_arr_np):
        """
        Method to service write requests coming from systolic array. Logic: Assuming no miss, keep
        adding to the active buffer. Once the active buffer is full, drain a part of it to the DRAM.
        """
        assert incoming_cycles_arr_np.shape[0] == incoming_requests_arr_np.shape[0], \
                                                  'Cycles and requests do not match'
        out_cycles_arr = []
        offset = 0

        # DEBUG_num_drains = 0
        # DEBUG_append_to_trace_times = []

        for i in range(incoming_requests_arr_np.shape[0]):
            row = incoming_requests_arr_np[i]
            cycle = incoming_cycles_arr_np[i]
            current_cycle = cycle[0] + offset

            for elem in row:
                # Pay no attention to empty requests
                if elem == -1:
                    continue

                self.store_to_trace_mat_cache(elem)

                if current_cycle < self.drain_end_cycle:
                    if not self.free_space > 0:
                        offset += max(self.drain_end_cycle - current_cycle, 0)
                        current_cycle = self.drain_end_cycle

                elif self.free_space < (self.total_size_elems - self.drain_buf_size):
                    self.append_to_trace_mat(force=True)
                    self.drain_end_cycle = self.empty_drain_buf(empty_start_cycle=current_cycle)
                    # TODO sarbartha
                    #current_cycle = self.drain_end_cycle

            out_cycles_arr.append(current_cycle)

        num_lines = incoming_requests_arr_np.shape[0]
        out_cycles_arr_np = np.asarray(out_cycles_arr).reshape((num_lines, 1))

        #print('DEBUG: Num Drains = ' + str(DEBUG_num_drains))
        #print('DEBUG: Num appeneds = ' + str(len(DEBUG_append_to_trace_times)))
        #plt.plot(DEBUG_append_to_trace_times)
        #plt.show()

        return out_cycles_arr_np

    #
    def empty_drain_buf(self, empty_start_cycle=0):
        """
        Method to drain the drain buffer once the active buffer is full.
        """

        lines_to_fill_dbuf = int(math.ceil(self.drain_buf_size / self.req_gen_bandwidth))
        self.drain_buf_end_line_id = self.drain_buf_start_line_id + lines_to_fill_dbuf
        self.drain_buf_end_line_id = min(self.drain_buf_end_line_id, self.trace_matrix.shape[0])

        requests_arr_np = \
                    self.trace_matrix[self.drain_buf_start_line_id: self.drain_buf_end_line_id, :]
        num_lines = requests_arr_np.shape[0]

        data_sz_to_drain = num_lines * requests_arr_np.shape[1]
        # Adjust for -1
        for elem in requests_arr_np[-1,:]:
            if elem == -1:
                data_sz_to_drain -= 1
        self.num_access += data_sz_to_drain

        cycles_arr = [x+empty_start_cycle for x in range(num_lines)]
        cycles_arr_np = np.asarray(cycles_arr).reshape((num_lines, 1))
        serviced_cycles_arr = self.backing_buffer.service_writes(requests_arr_np, cycles_arr_np)

        # Assign the cycles vector which will be used to generate the complete trace
        if not self.trace_valid:
            self.cycles_vec = serviced_cycles_arr
            self.trace_valid = True
        else:
            self.cycles_vec = np.concatenate((self.cycles_vec, serviced_cycles_arr), axis=0)

        service_end_cycle = np.amax(serviced_cycles_arr)
        self.free_space += data_sz_to_drain

        self.drain_buf_start_line_id = self.drain_buf_end_line_id
        return service_end_cycle

    #
    def empty_all_buffers(self, cycle):
        """
        Method to drain all of the active buffer.
        """
        self.append_to_trace_mat(force=True)

        if self.trace_matrix_empty:
            return

        while self.drain_buf_start_line_id < self.trace_matrix.shape[0]:
            self.drain_end_cycle = self.empty_drain_buf(empty_start_cycle=cycle)
            cycle = self.drain_end_cycle + 1

    #
    def get_trace_matrix(self):
        """
        Method to get the write buffer trace matrix. It contains addresses requsted by the systolic
        array and the cycles (first column) at which the requests are made.
        """
        if not self.trace_valid:
            print('No trace has been generated yet')
            return

        trace_matrix = np.column_stack((self.cycles_vec, self.trace_matrix))

        return trace_matrix

    #
    def get_free_space(self):
        """
        Method to get free space of the write buffer.
        """
        return self.free_space

    #
    def get_num_accesses(self):
        """
        Method to get number of accesses of the write buffer if trace_valid flag is set.
        """
        assert self.trace_valid, 'Traces not ready yet'
        return self.num_access

    #
    def get_external_access_start_stop_cycles(self):
        """
        Method to get start and stop cycles of the write buffer if trace_valid flag is set.
        """
        assert self.trace_valid, 'Traces not ready yet'
        start_cycle = np.amin(self.cycles_vec)
        end_cycle = np.amax(self.cycles_vec)
        return start_cycle, end_cycle

    #
    def print_trace(self, filename):
        """
        Method to write the write buffer trace matrix to a file.
        """
        if not self.trace_valid:
            print('No trace has been generated yet')
            return
        trace_matrix = self.get_trace_matrix()
        np.savetxt(filename, trace_matrix, fmt='%s', delimiter=",")
