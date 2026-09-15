"""
Code genuinely shared between cosma/ and onsram/: parsing model.json into
plain nodes/tensors dicts (graph_builder.py), resolving/exporting a raw
.tflite to model.json (model_resolver.py), and independently replaying a
solved SPM plan to verify it's physically realizable (spm_allocator.py).

These three carry zero algorithm-specific logic -- no ILP, no FoM scoring,
no notion of "which paper decided this plan" -- so both papers import this
package directly, unmodified, rather than each keeping a copy.

This is deliberately NOT where the two papers' SCALE-Sim-driving code
lives, even though that code is structurally similar between them (see
cosma/helpers/topology_builder.py vs. onsram/onsram_helpers/topology.py,
or scalesim/memory/cosma_resident_buffers.py vs. onsram/onsram_helpers/
resident_buffers.py). That code is duplicated on purpose: a future change
to one paper's ILP/heuristic or simulation wiring must never be able to
move the other paper's simulated numbers. Only the parts with no
algorithm baked in at all -- graph parsing, model export/caching, and
plan validation -- are safe to genuinely share, and those live here.
"""
