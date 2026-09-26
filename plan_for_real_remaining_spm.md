# Constrain the ifmap buffer to real remaining SPM room

## Context

`SpmAllocator` (`spm_common/spm_allocator.py`) independently verifies that
resident activation tensors never exceed `memory_budget_bytes` *against each
other*. Separately, for every layer, `_make_memory_system()` in
`cosma/helpers/baseline.py` and `onsram/onsram_helpers/scale_sim_runner.py`
sizes SCALE-Sim's own real per-layer buffers to that layer's exact raw
tensor byte counts — as if the whole budget were always free for whichever
layer happens to be running, with zero awareness of what's already resident.
A diagnostic added in the previous session (`budget_overflow_events`,
`[COSMA/OnSRAM SPM] WARNING`) already proved this gap is real, not
hypothetical: COSMA ResNet-20-CIFAR10 @ 200KB shows 4/32 timesteps over
budget (worst case 8,448 bytes); OnSRAM MobileNet @ 2MB shows 3/30 timesteps
over (worst case 2,502,656 bytes — one layer's own weights alone are 4.2MB
against a 2MB budget). That diagnostic is detection-only; it never changes
what gets simulated.

This plan makes the *ifmap* buffer's size genuinely reflect how much SPM
room is actually left, instead of just reporting when it wouldn't have been.
The user has already decided the scope: **only ifmap is constrained.**
Filter/weight bytes are never tracked by either paper's own algorithm
(confirmed against the COSMA paper's primary text this session; both
papers' "activation tensors only" convention), so filter keeps its current
full-size treatment unchanged. Ofmap has its own separate `stays_on_chip`
handling and is out of scope here too. There is deliberately no
"splitting a shortfall across three buffers" logic — this only ever
constrains one buffer.

Why this is the right fix and not scope creep: SCALE-Sim's own read-buffer
classes are already genuine streaming models (active + prefetch windows,
real hit/miss and re-fetch bookkeeping) — currently never exercised under
real pressure because the buffer is always sized to exactly fit the data.
Shrinking it to the real remaining room makes that *existing* machinery
simulate real re-fetch traffic under pressure — a physically real
consequence, not an invented eviction policy. This complies with the
session's own governing principle (`CONTINUE_HERE.md`): no new "live
capacity-check-and-decide" logic is added anywhere; SCALE-Sim keeps playing
back a sizing input the same way it always has, just now a more honest one.

## The one real correctness subtlety — and why the fix for it is genuinely shared, not per-paper

`SpmAllocator.occupied_bytes()` read immediately after `allocator.step(t)`
already includes any tensor whose action at that exact `t` was `'C'` or
`'R'` (allocated within that same `step()` call before `occupied_bytes()`
is read — confirmed by direct trace of `step()`'s body). COSMA's own
ifmap-residency check is specifically `resident_action.get((ifmap_id, t))
== 'P'` — so a genuinely-being-fetched ifmap whose action is `'R'`
(retrieved, not preserved) is *already* counted in `occupied_bytes(t)`,
because SpmAllocator considers it resident the instant it's retrieved. If
"remaining budget for ifmap" were computed as
`memory_budget_bytes - occupied_bytes(t)` naively, that tensor would be
double-charged: once as "already occupying room," again as "the thing
about to be fetched into that same room." The fix must exclude the current
layer's own ifmap tensor's bytes from `occupied_bytes(t)` when its action
at `(ifmap_id, t)` is `'C'` or `'R'`.

**This logic has zero paper-specific content and belongs in `spm_common/`,
not duplicated into `cosma/helpers/baseline.py` and
`onsram/onsram_helpers/scale_sim_runner.py` separately** (this is the
correction from the plan's first draft, which had drafted near-identical
arithmetic independently into both files — same mistake in spirit as
`graph_builder.py`/`model_resolver.py`/`spm_allocator.py` itself being
stuck under `cosma/helpers/` before this project's own earlier
`spm_common/` split). It operates purely on `SpmAllocator`'s own state,
`resident_action`, and a tensor id/timestep — nothing COSMA- or
OnSRAM-specific. OnSRAM never needs the exclusion in *practice* (its
`resident_action` only ever contains `'C'`/`'P'`, never `'R'` — a
non-pinned tensor gets zero entries at all, so it was never added to
`_occupants` to begin with), but that's a property of what data OnSRAM's
plan ever contains, not a reason to fork the code — the exact same
generic method is correct, unmodified, for both, and simply never
triggers its exclusion branch on the OnSRAM side. Implementing this twice
would be re-deriving the same 4 lines of arithmetic in two places for no
isolation benefit, since it makes no simulation-driving decision at all
(compare with `topology_builder.py`/`resident_buffers.py`, which *are*
correctly duplicated, because those genuinely do drive each paper's own
simulation).

**New method on `SpmAllocator`, `spm_common/spm_allocator.py`:**

```python
def remaining_budget_for(self, tensor_id: int, t: int) -> int:
    """
    How much SPM room is left for tensor_id's own fetch at timestep t,
    given everything else currently resident (call only after step(t)
    for this same t). Excludes tensor_id's own contribution if its own
    action at t was 'C' or 'R' -- step(t) already added it to occupied
    bytes by the time this is called, so without this exclusion a
    tensor being fetched right now would be double-charged: once as
    "already occupying room," again as "the thing that needs room
    fetched into." Generic across any plan shape -- a harmless no-op
    whenever tensor_id's own action at t isn't 'C'/'R' (true for every
    OnSRAM tensor, which never retrieves, and whose own 'C' only ever
    lands at a tensor's PRODUCER timestep, never a later consumer's t --
    not a paper-specific branch, just what the data there looks like).
    """
    occupied = self.occupied_bytes()
    own_action = next((act for a, act in self._actions_by_t.get(t, [])
                        if a == tensor_id), None)
    if own_action in ('C', 'R'):
        occupied -= self._tensors[tensor_id].size_bytes
    return max(self._budget - occupied, 0)
```

Verified directly against the current class body: `self._budget` already
stores `memory_budget_bytes` (set in `__init__`), `self._tensors` already
stores the tensor dict, `self._actions_by_t` already groups actions by
timestep — this method needs no new state, only reads what the
constructor already keeps.

## Required prerequisite: a real crash risk in `read_buffer.py`

Verified directly (not just theorized) by reading the actual code:
`configs/scale.cfg` — the actual `DEFAULT_CONFIG` for `run_cosma.py`,
`run_experiments.py`, and `run_onsram.py` — has `InterfaceBandwidth: USER`,
not CALC. In `_make_memory_system()`, a non-resident ifmap fetch with
`ifmap_buf_class=None` takes `double_buffered_scratchpad`'s plain
`read_buffer` path (bank/port model, `scalesim/memory/read_buffer.py`), not
`ReadBufferEstimateBw`. `read_buffer.set_params()` (lines 93-94) computes:

```python
self.active_buf_size = int(math.ceil(self.total_size_elems * self.active_buf_frac))
self.prefetch_buf_size = self.total_size_elems - self.active_buf_size
```

`_make_memory_system()` always passes `rd_buf_active_frac=0.5`. At
`total_size_elems == 1` (exactly what the *existing* `max(ifmap_buf_size_bytes,
1)` floor already in `_make_memory_system()` produces once our new ceiling
drives the natural size down that far — and our own OnSRAM repro case, 2.5MB
over a 2MB budget, will very plausibly hit this): `active_buf_size =
ceil(0.5) = 1`, `prefetch_buf_size = 1 - 1 = 0`. Traced forward into
`new_prefetch()` (line 479): `num_lines = ceil(0 / req_gen_bandwidth) = 0`,
producing a zero-row array that reaches `np.amax()` at line 520 —
`ValueError: zero-size array to reduction operation maximum which has no
identity`. A hard crash, not silently-wrong output.

This is the same bug class as two already-fixed siblings in this codebase
(`write_buffer.py`'s `drain_buf_size = max(1, ...)`,
`read_buffer_estimate_bw.py`'s `num_items_per_set = max(1, ...)`) — just
never caught in this third file because nothing had driven its buffer size
this small before now. Fix it the same way, same file, same precedent:

```python
# scalesim/memory/read_buffer.py, set_params() (~line 94)
self.prefetch_buf_size = max(1, self.total_size_elems - self.active_buf_size)
```

One-line, floor-only, same disclosed tradeoff as the existing precedents
(active + prefetch can exceed `total_size_elems` by 1 in this edge case; a
no-op for any normally-sized buffer). This must land *before* the ifmap
ceiling changes are tested, or the very first tight-budget verification run
will crash instead of demonstrating the fix.

## Implementation

### 1. `scalesim/memory/read_buffer.py`
Apply the one-line floor above in `set_params()`. Leave `__init__()`'s and
`reset()`'s own copies of this computation alone — both use hardcoded
defaults (`total_size_bytes=128`, `active_buf_frac=0.9`) that never reach
the degenerate case, so touching them is unnecessary.

### 2. `cosma/helpers/baseline.py`

**`_run_layers()`** — extend the existing `if allocator is not None:` block
inside the per-timestep loop (right alongside the existing
`budget_overflow_events` computation) to call the new shared
`allocator.remaining_budget_for()` and pass the result into
`_simulate_layer()`:

```python
budget_overflow_events = []
ifmap_ceiling_events = []  # new: for the optional visibility print, see below

...
for t, lid in schedule:
    layer = layer_by_id[lid]
    if allocator is not None:
        allocator.step(t)
    if lid not in layer_id_to_row:
        continue

    ifmap_buf_ceiling_bytes = None
    if allocator is not None:
        ifmap_bytes, _, filter_bytes = _layer_operand_bytes(layer, tensor_shapes)
        combined = allocator.occupied_bytes() + filter_bytes
        if combined > memory_budget_bytes:
            budget_overflow_events.append((t, lid, combined - memory_budget_bytes))

        ifmap_id = _activation_input_tensor_id(layer)
        if ifmap_id is not None:
            ifmap_buf_ceiling_bytes = allocator.remaining_budget_for(ifmap_id, t)
            if (resident_action.get((ifmap_id, t)) != 'P'
                    and ifmap_buf_ceiling_bytes < ifmap_bytes):
                ifmap_ceiling_events.append((t, lid, ifmap_bytes - ifmap_buf_ceiling_bytes))

    row_to_stats[layer_id_to_row[lid]] = _simulate_layer(
        config, topo, layout, layer_id_to_row[lid], layer, t, tensor_shapes,
        verbose, resident_action=resident_action,
        ifmap_buf_ceiling_bytes=ifmap_buf_ceiling_bytes)
```

All of the double-counting reasoning now lives in one place
(`SpmAllocator.remaining_budget_for()`, above) — this call site just uses
it, no arithmetic duplicated here.

**`_simulate_layer()`** — one new parameter, default `None` so nothing
changes when the caller doesn't pass it (i.e. `run_baseline()`'s plain
pass, which never touches this):

```python
def _simulate_layer(config, topo, layout, row: int, layer: dict, t: int,
                     tensor_shapes: Dict[int, dict], verbose: bool,
                     resident_action: dict = None,
                     ifmap_buf_ceiling_bytes: int = None) -> dict:
    ifmap_bytes, ofmap_bytes, filter_bytes = _layer_operand_bytes(layer, tensor_shapes)

    ifmap_resident = False
    if resident_action is not None:
        ifmap_id = _activation_input_tensor_id(layer)
        if ifmap_id is not None:
            ifmap_resident = resident_action.get((ifmap_id, t)) == 'P'

    ifmap_buf_bytes = ifmap_bytes
    if not ifmap_resident and ifmap_buf_ceiling_bytes is not None:
        ifmap_buf_bytes = min(ifmap_bytes, ifmap_buf_ceiling_bytes)

    mem_sys = _make_memory_system(
        config, topo, row, ifmap_buf_bytes, filter_bytes, ofmap_bytes, verbose,
        ifmap_resident=ifmap_resident,
        ofmap_stays_on_chip=resident_action is not None,
    )
    # ... unchanged from here ...
```

**`_make_memory_system()`** — no changes at all. It keeps taking a plain
`ifmap_buf_size_bytes` int; the existing `max(ifmap_buf_size_bytes, 1)`
keeps applying on top of the new ceiling automatically. Smallest possible
surface change.

**Optional but recommended** — a third summary print (unconditional, same
style as the existing two), so the fix's effect is actually observable in
output, not just inferable from DRAM-byte totals moving:

```python
if ifmap_ceiling_events:
    worst_t, worst_lid, worst_short = max(ifmap_ceiling_events, key=lambda e: e[2])
    print(f"[COSMA SPM] ifmap buffer constrained below natural size on "
          f"{len(ifmap_ceiling_events)} of {len(schedule)} timestep(s) "
          f"(worst case {worst_short} bytes short of natural fetch size "
          f"at t={worst_t}, layer {worst_lid}) -- expect extra simulated "
          f"re-fetch traffic there, not a bug")
else:
    print(f"[COSMA SPM] ifmap buffer never constrained below its natural "
          f"size across all {len(schedule)} timesteps")
```

**Existing `budget_overflow_events` diagnostic** — logic unchanged (it
never included ifmap bytes to begin with, so it's unaffected by this fix);
update its header comment, which currently frames the ifmap exclusion as
an open ambiguity, to instead say plainly that filter/weight is the only
thing this specific check still adds on top of residency now that ifmap's
own sizing is handled directly via `ifmap_buf_ceiling_bytes`.

### 3. `onsram/onsram_helpers/scale_sim_runner.py`
Mirror the `_run_layers()`/`_simulate_layer()` structural changes exactly
(same `ifmap_buf_ceiling_bytes` threading, same `[OnSRAM SPM]`-prefixed
print) — **and call the exact same `allocator.remaining_budget_for()`**
from `spm_common.spm_allocator`, not a re-derived copy. This file already
imports `SpmAllocator` from `spm_common.spm_allocator` (confirmed:
`from spm_common.spm_allocator import SpmAllocator`), so this needs no new
import, just using a method that already exists on the object it already
holds. This is the one piece of this change that is *not* duplicated
between the two files — same as `graph_builder`/`model_resolver`/
`SpmAllocator` itself already aren't. Everything else specific to driving
SCALE-Sim (`_make_memory_system()`, the resident-buffer class choice,
`_simulate_layer()`'s own body) stays a separate, self-contained copy per
the established convention, unchanged by this refinement.

## Verification

1. **`read_buffer.py` floor fix in isolation first** — confirm the
   one-line change doesn't alter any existing, already-trusted output
   (it's a no-op except at `total_size_elems == 1`, which nothing hit
   before this session's own diagnostic work).
2. **COSMA ResNet-20-CIFAR10 @ 200KB** (`cosma/_exported/resnet20_cifar10/model.json`,
   matches the existing repro used to find this gap):
   ```
   python3 cosma/run_cosma.py --model-json cosma/_exported/resnet20_cifar10/model.json --budget-kb 200 --config configs/scale.cfg
   ```
   Expect: no crash; `[COSMA SPM] verified 32 timesteps... 0 violations`
   and the existing filter-only WARNING print byte-identical to before
   (both come from `SpmAllocator`'s own replay, untouched by this fix);
   the plain baseline pass's numbers unchanged (proves the gating on
   `resident_action is not None` held); the aware pass's total DRAM
   bytes/cycles **≥** the previous run's numbers (genuine extra re-fetch
   cost on the timesteps that were already known to be under pressure —
   a lower reported speedup than previously published here is expected
   and correct, not a regression).
3. **OnSRAM MobileNet @ 2MB** (the project's literal default):
   ```
   python3 onsram/run_onsram.py
   ```
   Same expectations, `[OnSRAM SPM]`-prefixed; this is the case most
   likely to have actually needed the `read_buffer.py` floor fix, given
   its 2.5MB overflow dwarfs the 2MB budget.
4. **Regression check at a comfortable, non-adversarial budget** — re-run
   at least one budget already logged under `cosma/logs/` that previously
   printed "0 of N timesteps exceed the budget," confirm every number is
   byte-identical to the pre-fix log. This is the most important
   non-regression guard: it proves the fix is a true no-op whenever the
   ceiling never actually binds, which is the common case for this
   project's existing, already-trusted results.

## Critical files
- `spm_common/spm_allocator.py` — new shared `remaining_budget_for()` method (the only genuinely shared piece of this change)
- `scalesim/memory/read_buffer.py` — one-line floor fix, required prerequisite
- `cosma/helpers/baseline.py` — `_run_layers()`/`_simulate_layer()` threading, COSMA's own file
- `onsram/onsram_helpers/scale_sim_runner.py` — same threading, OnSRAM's own file, calls the same shared method

---

# Phase 2: Extend budget-awareness to filter and ofmap

## Context

Phase 1 above (already implemented and verified) constrained only the ifmap
buffer. Filter/weight bytes were left deliberately unconstrained — always
passed to `_make_memory_system()` at full natural size — with a print-only
`budget_overflow_events`/WARNING diagnostic reporting when this was
physically impossible, never correcting it. Ofmap had no diagnostic and no
constraint at all.

The user asked for every operand type to know how much real SPM room is
left, accounting for what's currently pinned. Approved design: **ifmap and
ofmap get priority (they're the activations both COSMA's ILP and OnSRAM's
FoM/greedy algorithm actually optimize); filter absorbs whatever room is
left after both, worst case down to the existing 1-byte floor**. Rationale
(user's own): some SPM-management papers model weights as living in a
wholly separate, dedicated scratchpad — OnSRAM's own paper studies exactly
this as a Fig. 10 variant. Treating filter as lowest-priority-within-the-
shared-budget is a reasonable middle ground consistent with that intuition,
without building a whole separate-SPM architecture. Same governing
principle as Phase 1: this is engine simulation-fidelity, not an algorithm
change — neither paper's own decision logic (ILP / FoM+greedy) is touched
or made aware of any of this; SCALE-Sim just plays back a more physically
honest sizing input.

This does **not** contradict `CONTINUE_HERE.md`'s prior note to hold off on
"filter-budget modeling" for OnSRAM's own algorithm — that note was about
giving OnSRAM's *pinning decision* awareness of weight competition (an
algorithm-scope change, still held off, no primary-source confirmation
it belongs in OnSRAM's own model). This phase never touches either
algorithm's decision logic — it's purely how faithfully SCALE-Sim
simulates buffer sizes for decisions the algorithms already made, the same
category `CONTINUE_HERE.md`'s own text says is always fair game. Worth a
short addendum note in that file when this lands, so a future reader
doesn't see a contradiction that isn't there.

Verified directly (two independent passes: fresh code reads plus a design
review) before finalizing this approach:
- `SpmAllocator.remaining_budget_for()` is already fully generic (works for
  any tensor id) and is called in exactly two places today, both with
  `ifmap_id` — extending it to accept a *group* of tensor ids is a
  backward-compatible generalization, not a behavior change to existing
  callers passing a single id.
- Filter's SCALE-Sim buffer always takes the same plain `read_buffer.py`
  class ifmap's own non-resident path uses (confirmed: `_make_memory_system()`
  never passes SCALE-Sim's own `filter_buf_class` override, and
  `configs/scale.cfg`'s `InterfaceBandwidth: USER` makes
  `estimate_bandwidth_mode=False`) — so filter is already protected by this
  session's earlier `read_buffer.py` crash-floor fix once it starts being
  shrunk too. No new crash-floor work needed.
- Ofmap's write buffer (`write_buffer.py`, and its `CosmaResidentWriteBuffer`/
  `OnsramResidentWriteBuffer` subclasses) already has its own pre-existing
  `max(1, ...)` floor, and neither subclass overrides `set_params()` — also
  already safe.
- **Important, counterintuitive finding**: clamping ofmap's own buffer size
  does **not** change `ofmap_dram_bytes` or `compute_cycles` today, at any
  value. `ofmap_stays_on_chip=resident_action is not None` (unconditional
  for every layer in the aware pass, untouched by this phase) already makes
  the resident write-buffer classes report zero DRAM accesses and an
  instant, zero-cost drain regardless of buffer capacity — confirmed by
  tracing `empty_drain_buf()`/`get_num_accesses()`/
  `get_external_access_start_stop_cycles()`'s overrides, and matches this
  session's own `onsram/logs/MobileNet_2MB_paper.log` line describing ofmap
  credit as "unconditional in this mode, not specific to pinning choices."
  Implementing ofmap's clamp anyway is still correct and necessary: filter's
  own leftover room has to be computed *after* subtracting what ofmap's
  natural footprint actually claims, since ofmap genuinely occupies real
  space concurrently with filter even though its *drain cost* is waived.
  Expect `ofmap_dram_bytes`/`compute_cycles` to stay bit-identical
  before/after this phase, even on timesteps where the new ofmap diagnostic
  fires — that's the correct, expected outcome, not a sign the fix is inert
  or broken.
- **A real, disclosed side effect on ifmap's own numbers**: grouping ifmap
  and ofmap into one shared-room calculation (`remaining_budget_for((ifmap_id,
  ofmap_id), t)`) can only ever make ifmap's ceiling *more generous* than
  Phase 1's ifmap-only version, never tighter (Phase 1's version excluded
  only ifmap's own same-timestep contribution; this version excludes ifmap's
  *and* ofmap's, so strictly less gets left counted as "occupied"). This is
  common for COSMA specifically: COSMA's ILP creates *every* tracked tensor
  exactly once, at its own producer's timestep (Eq.7) — so a layer's own
  ofmap essentially always has a fresh `'C'` at that layer's own `t`,
  meaning `room_all` differs from Phase 1's number on most timesteps where
  ifmap needs a ceiling at all. For OnSRAM it's conditional — only on
  timesteps where that layer's own output specifically wins the FoM pinning
  competition, since OnSRAM only records `'C'` for tensors it decides to
  pin. This is a genuine correctness improvement riding along with this
  phase (Phase 1's ifmap-only ceiling had a latent inaccuracy: it implicitly
  treated a layer's own about-to-be-created ofmap as already competing for
  room against that same layer's own ifmap fetch, when in real per-layer
  execution ifmap is fetched *before* ofmap is written) — not scope creep,
  but it means COSMA's previously-reported `ifmap_ceiling_events` numbers
  from Phase 1 testing will likely shift (fewer/smaller events) once this
  phase lands. Expected and correct, not a regression — call this out
  explicitly when reporting results so it doesn't look like unexplained
  drift.
- The existing `budget_overflow_events`/WARNING check will **not** drop to
  zero after this phase — it deliberately compares against raw, unclamped
  `filter_bytes` (not the new `filter_buf_ceiling_bytes`), so it stays a
  frozen "would this have violated the budget under the OLD, fully-
  unconstrained filter treatment" historical benchmark, unchanged by this
  phase by design. The actual "does the new fix work" check is a new
  always-true-by-construction `assert` (below), not this diagnostic.

## Implementation

### 1. `spm_common/spm_allocator.py` — generalize `remaining_budget_for()`

Replace the existing single-tensor method with a group-accepting version
(a bare int is still accepted, treated as a one-element group — fully
backward compatible with Phase 1's two existing call sites):

```python
def remaining_budget_for(self, tensor_ids, t: int) -> int:
    """
    How much SPM room is left for one or more tensors' own fresh
    allocation need at timestep t, given everything else currently
    resident (call only after step(t) for this same t). Generalizes the
    original single-tensor version (a bare int is treated as a
    one-element group) to a group of ids sharing the same timestep's
    fresh-allocation need -- e.g. a layer's own ifmap AND ofmap at that
    layer's own t. Excludes each id's own contribution from
    occupied_bytes() if THAT id's own action at t was 'C' or 'R' -- see
    the single-tensor version's original docstring for the double-
    counting reasoning, which applies per-id here. Every OTHER tensor's
    contribution -- including another id in the same group, if it
    doesn't also have a 'C'/'R' action at this exact t -- stays fully
    counted, since it's a genuine, currently-resident competitor for the
    same budget. Duplicate ids excluded only once.
    """
    if isinstance(tensor_ids, int):
        tensor_ids = (tensor_ids,)
    occupied = self.occupied_bytes()
    actions_at_t = dict(self._actions_by_t.get(t, []))
    for tid in dict.fromkeys(tensor_ids):
        if actions_at_t.get(tid) in ('C', 'R'):
            occupied -= self._tensors[tid].size_bytes
    return max(self._budget - occupied, 0)
```

### 2. `cosma/helpers/baseline.py`, mirrored exactly into `onsram/onsram_helpers/scale_sim_runner.py`

**New helper**, next to `_activation_input_tensor_id()`:

```python
def _ofmap_tensor_id(layer: dict):
    """The tensor id of this layer's own freshly-produced output -- mirrors
    _activation_input_tensor_id()'s role for inputs. Only looks at
    outputs[0] -- the same pre-existing simplification _layer_operand_bytes()'s
    own ofmap_bytes already makes."""
    outputs = layer.get('outputs', [])
    return outputs[0] if outputs else None
```

**`_simulate_layer()`** — two new params, both default `None` (so
`run_baseline()`'s untouched call keeps behaving exactly as before),
applied unconditionally (no residency gate like ifmap's — ofmap can never
be `'P'` at its own creation `t`, and filter has no residency concept at
all):

```python
def _simulate_layer(config, topo, layout, row: int, layer: dict, t: int,
                     tensor_shapes: Dict[int, dict], verbose: bool,
                     resident_action: dict = None,
                     ifmap_buf_ceiling_bytes: int = None,
                     ofmap_buf_ceiling_bytes: int = None,
                     filter_buf_ceiling_bytes: int = None) -> dict:
    ifmap_bytes, ofmap_bytes, filter_bytes = _layer_operand_bytes(layer, tensor_shapes)

    ifmap_resident = False
    if resident_action is not None:
        ifmap_id = _activation_input_tensor_id(layer)
        if ifmap_id is not None:
            ifmap_resident = resident_action.get((ifmap_id, t)) == 'P'

    ifmap_buf_bytes = ifmap_bytes
    if not ifmap_resident and ifmap_buf_ceiling_bytes is not None:
        ifmap_buf_bytes = min(ifmap_bytes, ifmap_buf_ceiling_bytes)

    ofmap_buf_bytes = ofmap_bytes
    if ofmap_buf_ceiling_bytes is not None:
        ofmap_buf_bytes = min(ofmap_bytes, ofmap_buf_ceiling_bytes)

    filter_buf_bytes = filter_bytes
    if filter_buf_ceiling_bytes is not None:
        filter_buf_bytes = min(filter_bytes, filter_buf_ceiling_bytes)

    mem_sys = _make_memory_system(
        config, topo, row, ifmap_buf_bytes, filter_buf_bytes, ofmap_buf_bytes, verbose,
        ifmap_resident=ifmap_resident,
        ofmap_stays_on_chip=resident_action is not None,
    )
    # ... rest unchanged ...
```

Note the positional order into `_make_memory_system()` is `(ifmap_buf_bytes,
filter_buf_bytes, ofmap_buf_bytes)` — filter before ofmap, matching its
existing signature (`ifmap_buf_size_bytes, filter_buf_size_bytes,
ofmap_buf_size_bytes`). Easiest spot to introduce a silent transcription
bug — double check this order when implementing.

**`_run_layers()`** — replace the `if allocator is not None:` body inside
the per-timestep loop with sequential depletion (ifmap first, unchanged
priority from Phase 1; then ofmap; filter absorbs the rest):

```python
budget_overflow_events = []
ifmap_ceiling_events = []
ofmap_ceiling_events = []
filter_ceiling_events = []

...
for t, lid in schedule:
    layer = layer_by_id[lid]
    if allocator is not None:
        allocator.step(t)
    if lid not in layer_id_to_row:
        continue

    ifmap_buf_ceiling_bytes = None
    ofmap_buf_ceiling_bytes = None
    filter_buf_ceiling_bytes = None
    if allocator is not None:
        ifmap_bytes, ofmap_bytes, filter_bytes = _layer_operand_bytes(layer, tensor_shapes)
        combined = allocator.occupied_bytes() + filter_bytes
        if combined > memory_budget_bytes:
            budget_overflow_events.append((t, lid, combined - memory_budget_bytes))

        ifmap_id = _activation_input_tensor_id(layer)
        ofmap_id = _ofmap_tensor_id(layer)
        claim_ids = tuple(tid for tid in (ifmap_id, ofmap_id) if tid is not None)
        room_all = allocator.remaining_budget_for(claim_ids, t)

        # 1. ifmap -- same priority as Phase 1; room_all's basis now also
        #    excludes this layer's own ofmap (if it has a fresh 'C' this
        #    t), which can only make this MORE generous than Phase 1,
        #    never tighter (see Context).
        ifmap_actual_claim = 0
        if ifmap_id is not None:
            ifmap_buf_ceiling_bytes = room_all
            if resident_action.get((ifmap_id, t)) != 'P':
                ifmap_actual_claim = min(ifmap_bytes, room_all)
                if room_all < ifmap_bytes:
                    ifmap_ceiling_events.append((t, lid, ifmap_bytes - room_all))

        # 2. ofmap -- whatever's left after ifmap's actual claim.
        ofmap_room = max(room_all - ifmap_actual_claim, 0)
        ofmap_actual_claim = 0
        if ofmap_id is not None:
            ofmap_buf_ceiling_bytes = ofmap_room
            ofmap_actual_claim = min(ofmap_bytes, ofmap_room)
            if ofmap_room < ofmap_bytes:
                ofmap_ceiling_events.append((t, lid, ofmap_bytes - ofmap_room))

        # 3. filter -- absorbs whatever's left, worst case to the 1-byte
        #    floor _make_memory_system() already applies. No tensor id of
        #    its own (never part of resident_action), so this is pure
        #    leftover arithmetic, not another remaining_budget_for() call.
        filter_buf_ceiling_bytes = max(ofmap_room - ofmap_actual_claim, 0)
        if filter_buf_ceiling_bytes < filter_bytes:
            filter_ceiling_events.append(
                (t, lid, filter_bytes - filter_buf_ceiling_bytes))

        assert (ifmap_actual_claim + ofmap_actual_claim
                + min(filter_bytes, filter_buf_ceiling_bytes)) <= room_all, (
            f"sequential depletion arithmetic bug at t={t}, layer {lid}")

    row_to_stats[layer_id_to_row[lid]] = _simulate_layer(
        config, topo, layout, layer_id_to_row[lid], layer, t, tensor_shapes,
        verbose, resident_action=resident_action,
        ifmap_buf_ceiling_bytes=ifmap_buf_ceiling_bytes,
        ofmap_buf_ceiling_bytes=ofmap_buf_ceiling_bytes,
        filter_buf_ceiling_bytes=filter_buf_ceiling_bytes)
```

The `assert` is the real "the fix guarantees no violation" invariant —
always true by construction given the sequential-depletion arithmetic; if
it ever fires, that's a real bug in this logic, not a modeling edge case.

**New diagnostic prints**, appended after the existing `ifmap_ceiling_events`
block (mirror `[COSMA SPM]` → `[OnSRAM SPM]` in the other file):

```python
if ofmap_ceiling_events:
    worst_t, worst_lid, worst_short = max(ofmap_ceiling_events, key=lambda e: e[2])
    print(f"[COSMA SPM] ofmap buffer's natural size exceeds real remaining "
          f"SPM room (after ifmap's own claim) on {len(ofmap_ceiling_events)} of "
          f"{len(schedule)} timestep(s) (worst case {worst_short} bytes short at "
          f"t={worst_t}, layer {worst_lid}) -- descriptive only: ofmap_stays_on_chip's "
          f"existing blanket waiver already treats every layer's own output write as "
          f"a free, instantaneous on-chip move regardless of buffer capacity, so this "
          f"does not change ofmap_dram_bytes or compute_cycles, not a bug")
else:
    print(f"[COSMA SPM] ofmap buffer's natural size never exceeded real remaining "
          f"SPM room across all {len(schedule)} timesteps")
if filter_ceiling_events:
    worst_t, worst_lid, worst_short = max(filter_ceiling_events, key=lambda e: e[2])
    print(f"[COSMA SPM] filter buffer constrained below natural size on "
          f"{len(filter_ceiling_events)} of {len(schedule)} timestep(s) "
          f"(worst case {worst_short} bytes short of natural fetch size "
          f"at t={worst_t}, layer {worst_lid}) -- expect extra simulated "
          f"re-fetch traffic there, not a bug")
else:
    print(f"[COSMA SPM] filter buffer never constrained below its natural "
          f"size across all {len(schedule)} timesteps")
```

Also reword the existing `budget_overflow_events` WARNING print (same
change in both files) so it can't be mistaken for a live check post-fix:
swap "exceed the budget" for "would exceed the budget under the OLD,
fully-unconstrained filter/weight treatment."

### 3. `onsram/onsram_helpers/scale_sim_runner.py`
Mirror all of the above exactly — same `_ofmap_tensor_id()` helper, same
`_simulate_layer()`/`_run_layers()` threading, same `[OnSRAM SPM]`-prefixed
prints, same call into the one shared `allocator.remaining_budget_for()`.
`_make_memory_system()` needs **zero changes** in either file — it already
applies `max(x, 1)` to all three buffer sizes unconditionally and already
takes plain ints.

### 4. `CONTINUE_HERE.md`
Short addendum near the existing "hold off on filter-budget modeling for
OnSRAM" note, distinguishing that (OnSRAM's own *algorithm* scope, still
held off) from this phase (engine buffer-sizing fidelity, unrelated to
either algorithm's decision logic).

## What does not change
- `ofmap_stays_on_chip=resident_action is not None` — untouched.
- `ifmap_resident` logic — untouched.
- `_make_memory_system()` — zero changes, in either file.
- COSMA's ILP / OnSRAM's FoM+greedy pinning — untouched; nothing here feeds
  back into either algorithm's own decision.
- No new buffer classes — filter never gets a resident-style override
  (correctly: it has no residency concept), it just gets a smaller plain
  `read_buffer.py` instance.

## Verification

1. **Regression (ceiling never binds)** — re-run an already-logged
   comfortable-budget case from `cosma/logs/`/`onsram/logs/`, confirm every
   number byte-identical, both new diagnostics print "never
   constrained"/"never exceeded."
2. **COSMA ResNet-20-CIFAR10 @ 200KB** (known filter-overflow repro,
   `cosma/_exported/resnet20_cifar10/model.json`):
   ```
   python3 cosma/run_cosma.py --model-json cosma/_exported/resnet20_cifar10/model.json --budget-kb 200 --config configs/scale.cfg
   ```
   Expect: `budget_overflow_events` WARNING fires with the *same* 4/32,
   worst case 8,448 bytes as Phase 1 (frozen benchmark, see Context); new
   `filter_ceiling_events`/`ofmap_ceiling_events` diagnostics engage;
   `ifmap_ceiling_events` may show fewer/smaller events than Phase 1's own
   result (never more — see Context's side-effect note); DRAM bytes/cycles
   change (filter's clamp is what actually moves numbers this time); a
   further-reduced reported speedup vs. Phase 1 is expected and correct.
3. **OnSRAM MobileNet @ 2MB** (`python3 onsram/run_onsram.py`, already
   reproduced in `onsram/logs/MobileNet_2MB_paper.log`): same expectations,
   `[OnSRAM SPM]`-prefixed; `ifmap_ceiling_events` should shift less than
   COSMA's (only on timesteps where that layer's own output wins FoM
   pinning).
4. **Ofmap-inertness check, both cases above**: `ofmap_dram_bytes` totals
   and `compute_cycles` must be **bit-identical** to Phase 1's own numbers,
   even on timesteps where `ofmap_ceiling_events` fires — confirms the
   expected-inert case from Context; if not bit-identical, something
   unrelated to this phase's intent changed (most likely
   `ofmap_stays_on_chip` accidentally became conditional somewhere).

## Critical files (Phase 2)
- `spm_common/spm_allocator.py` — generalize `remaining_budget_for()` to accept a group of tensor ids (only genuinely shared piece, same as Phase 1)
- `cosma/helpers/baseline.py` — `_ofmap_tensor_id()`, `_simulate_layer()`/`_run_layers()` sequential-depletion threading, new diagnostics
- `onsram/onsram_helpers/scale_sim_runner.py` — identical mirrored changes
- `CONTINUE_HERE.md` — short addendum distinguishing algorithm scope (still held off) from engine fidelity (this phase)
