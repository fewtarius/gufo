# Laguna S 2.1 quality

## Reference

The CPU reference implementation lives in `src/models/laguna/reference.cpp`. It
computes the forward pass on the host, one token at a time, from the same GGUF.

It is the intended oracle, and nothing in the tree calls it: no test compares GPU
logits against `ReferenceModel`, so that comparison runs by hand when a kernel
changes and is not a gate. Wiring it into a test is the largest missing piece of
this model's verification.

## Automated checks

- `laguna.chat_template` compares the chat template with golden cases built from
  the artifact's own Jinja template.
- `laguna.tool_calls` drives the native `<tool_call>` dialect scanner.
- `laguna.kv_q8` checks the device q8_0 store and dequantizer against the host
  codec, asserting byte-for-byte equality.

Prefill also checks against itself. `gufo bench --validate-prefill N` compares
one multi-token `Append` with the same N tokens appended one at a time, and on a
top-1 mismatch it prints both argmax tokens and their logit margins. Every width
tested during the port matched except **n=16**, which diverges with WMMA enabled
and with it disabled. That makes it a near-tie in the model rather than a kernel
defect, and it stays open.

## Accepted divergences

These are deliberate, and each is a property of the implementation rather than a
defect:

- **Block-sparse attention.** Above about 18.5k tokens of context the selection
  is approximate: a query reads 2 sink blocks, a 32-block local window and the
  256 best-scoring older 64-token blocks. Below that every candidate fits the
  budget and the result is exact. Needle-in-a-haystack matched dense at every
  depth tested.
- **The q8_0 KV cache.** Lossy by construction. See [KV-Q8.md](KV-Q8.md). F16 is
  the default.
- **DFlash2 speculative decoding.** The drafter's proposals are verified against
  the target's own logits, so an accepted draft is the target's token. A rejected
  draft costs a verification pass, not correctness.

The NPU path is not in this list because it no longer exists: upstream removed the
NPU stack, and the Laguna offload was worth +0.2% to +2.4%, so it was not
re-vendored.

## Replay guarantees

A snapshot restored into a session reproduces the cold session's output. The
session snapshot carries the attention K/V cache, the sparse pooled block bounds
and the drafter's injected-context frontier, so a restore does not re-derive any
of them from the restored state.

## Unresolved gaps

- No test compares GPU logits against the CPU reference; see above.
- The batched-versus-sequential divergence at n=16 has no root cause yet.
- The q8_0 KV cache has no measured KL divergence against F16 for this model. The
  codec contract is shared and cross-checked against its own upstream, but the
  per-model quality cost is unmeasured.
