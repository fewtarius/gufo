# Laguna S 2.1 quality

## Reference

The CPU reference implementation lives in `src/models/laguna/reference.cpp`.
It computes the same forward pass on the host, without the GPU, from the same
GGUF. It is the oracle the GPU engine is checked against.

## Greedy output

Greedy decoding on the GPU matches the CPU reference for the cases in
`tests/models/laguna/`. The model family is otherwise unchanged from the
upstream fork this work started from.

## Accepted divergences

These are deliberate, and each is a property of the implementation rather than
a defect:

- **Block-sparse attention.** Above about 18.5k tokens of context the
  selection is approximate: a query reads 2 sink blocks, a 32-block local
  window, and the 256 best-scoring older 64-token blocks. Below that every
  candidate fits the budget and the result is exact. Needle-in-a-haystack
  matched dense at every depth tested.
- **The q8_0 KV cache.** Lossy by construction. See
  [KV-Q8.md](KV-Q8.md). F16 is the default.
- **DFlash2 speculative decoding.** The drafter's proposals are verified
  against the target's own logits, so an accepted draft is the target's
  token. A rejected draft costs a verification pass, not correctness.
- **The NPU path.** Not present in this tree. Upstream removed the NPU stack,
  and the Laguna NPU path was worth +0.2% to +2.4%.

## Replay guarantees

A snapshot restored into a session reproduces the cold session's output. The
session snapshot carries the attention K/V cache, the sparse pooled block
bounds and the drafter's injected-context frontier, so a restore does not
re-derive any of them from the restored state.

## Unresolved gaps

- The CPU reference covers the forward pass, not the speculative decode loop.
  A restored session and a cold session are compared by their generated text
  rather than by logits.
- The q8_0 KV cache has no measured KL divergence against F16 for this model.
  The contract is shared with the reference codec, which is cross-checked
  against its own upstream; the per-model quality cost is unmeasured.
