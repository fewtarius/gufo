# Qwen35B-A3B quality

**The q8_0 K/V cache reproduces the dequantized values exactly, and the F16
path is unchanged.** These are execution-consistency checks; original-target
and GGUF-conversion parity remain unqualified.

[Model identities and dated evidence](../../models/qwen3.8-27b/artifacts/model-identities.json)
covers the shared family artifacts; this model has no separate identity
capture.

| Check | Result |
| --- | --- |
| Device q8_0 encoder against the host codec, byte for byte | Identical: same binary16 scale, same 32 codes, same tie-breaking |
| q8_0 attention against the FP16 attention over the same rows | Max difference 4.6e-4 on an output of magnitude 1.3e-2 |
| q8_0 read path against the FP16 plane holding the dequantized values | Max difference 0 |
| F16 KV storage against the FP32 reference plane, decode and split-K | Unchanged from the pre-q8_0 tree |
| Greedy completion, F16 against q8_0, same request | Identical text |

The first three are checked by
[the K/V storage test](../../../../tests/models/qwen/hip/attention/kv_storage_ops_test.cpp)
and run on every build. The fourth is the existing FP16 coverage, which the
q8_0 work did not touch. The fifth is a single-request control, not a
sweep.

## What the q8_0 cache costs

A lossy cache moves the logits, so a near-tie in the argmax can flip. The
format's own loss is the 4.6e-4 difference above, measured over 64
positions of one head.

The device encoder is the host codec's device half, and it matches byte for
byte. That matters: a scale computed differently, or a tie broken the other
way, would put a systematic error into every cached row, and the error
would compound over a context instead of staying at the format's precision.

## Unqualified

- **Original-target parity.** No comparison against an independent
  implementation of this model on identical weights. A llama.cpp Vulkan
  build runs on this host and serves these weights, so an engine
  comparison is available; the one in
  [BENCHMARKS.md](BENCHMARKS.md) is withheld until prefill is fixed.
- **GGUF-conversion parity.** The Unsloth UD-Q4_K_XL conversion is not
  compared against the original BF16 weights.
- **A greedy sweep.** The F16-versus-q8_0 completion control is one
  request. A sweep over the model card's workloads, as the
  [Qwen3.8 27B card](../qwen3.8-27b/QUALITY.md) does for DFlash2, is not
  run.
- **Multi-user behaviour.** One session is the whole memory budget at a
  131072-token context on this host, so no concurrency row is measured.
- **Speculative decoding.** The model carries an embedded MTP head, but
  the Qwen Flash-Next runner serves it, not this one. Nothing about
  speculation is qualified here.
