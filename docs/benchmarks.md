# Benchmarks

The speeds on the model pages come from one method and one harness. This page
explains what the numbers mean and how to measure your own.

## What is measured

Every number comes from a release-mode build on an AMD Strix Halo with 128 GB
of memory, and each one carries the date it was measured.

- **Prefill** (pp) is how fast Gufo reads the prompt, in tokens per second.
  Text models use a 2,048-token prompt.
- **Decode** (tg) is how fast it writes the answer: 128 generated tokens per
  request, with thinking off and greedy decoding.
- **Concurrent** rows run several requests at once and add up their decode
  speeds.
- **Depth** rows start from a long conversation already in the context, because
  decoding slows as the context grows.

Measurements go through the HTTP server, as a client would see them. The tables
report Gufo's own throughput, not a comparison against another engine: a
comparison has to be re-measured whenever either engine changes, so each table
carries Gufo's results and the revision that produced them.

Each table names the device and the revision it was measured on, and the
artifact beside it under `docs/internal/models/<model>/artifacts/` records the
exact command. That matters because the instruments differ: a row measured with
`gufo serve` is a request against a cached prefix, while `gufo prompt` or
`gufo bench` may prefill the whole prompt at once. Those report different rates
for the same weights, so do not compare one against the other.

Speculative modes are faster on text that is easy to predict. The tables say
when a row includes repetitive output, so a peak isn't mistaken for a typical
rate.

## Quality comes first

A speedup counts only if the output stays correct. Each model has a quality
report that compares Gufo with an independent implementation of the original
model, token by token, and checks that speculative decoding with greedy
sampling produces exactly the autoregressive output. The model pages link to
these reports.

## Measure your own

`gufo bench` measures a model without a server, in the style of llama.cpp's
`llama-bench`:

```sh
gufo bench --model "$MODEL" --n-prompt 2048 --n-gen 128
gufo bench --model "$MODEL" --n-prompt 2048 --n-gen 128 --n-depth 32768
```

For results comparable with the published ones, use a release build, close
other GPU work, and let the machine cool between runs, because sustained load
lowers its clock speeds. Run each case a few times and compare medians.

The full method, the benchmark driver and the reference builds are in the
[benchmark methodology](internal/benchmarks.md).
