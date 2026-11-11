# Benchmark methodology

How every number under `docs/internal/models/` is produced. Each per-model page
renders its tables from measurement artifacts, so a figure on a page always has
a JSON file behind it.

## Gufo is compared against itself

These pages report Gufo throughput, not Gufo against another engine. A
comparison table has to be re-measured whenever either engine changes, and the
reference build has to be kept in step by hand. That cost buys a number nobody
acts on, so the pages carry Gufo's own results and the release the results
describe.

A `bench.json` with no `reference` entry renders Gufo columns only. The renderer
drops the reference, gain and exactness columns instead of filling them with
placeholders. Add a `reference` entry to a config if a comparison is ever wanted
back for one model.

## Harness

`tools/bench/model-bench.py` measures a model's tables over HTTP, writes one
artifact per table and target under `docs/internal/models/<model>/artifacts/`,
and renders the tables between the `<!-- bench:<id> -->` markers in
`BENCHMARKS.md`. The `run` subcommand measures, `render` rewrites the page from
the artifacts it finds, and `tables` lists what a config declares.

Workloads live in `docs/internal/models/<model>/artifacts/bench.json`:

- `variants` names each weight set the page covers.
- `tables` declares the sweeps, their contexts and their depths.
- `gufo.llm` carries the server arguments a measurement uses, so a run does not
  depend on a shell alias.

The harness prints the server command it launched into the artifact, with the
model path reduced to a basename. Re-running a render without re-measuring
leaves the numbers alone.

## Protocol

Every throughput row comes from the same shape of request:

- `gufo serve llm`, one session, `/v1/chat/completions`
- temperature 0, fixed seed, thinking off
- 2,048 new prompt tokens, 128 generated tokens
- one fresh request per repetition, prompt cache a miss on each
- `prefill_tps` and `decode_tps` taken from the server's own completed-request
  log line, not from client wall time
- two repetitions per row, reported as mean ± standard deviation

Depth is the length of the cached prefix a row is measured against. It is
separate from the 2,048 new tokens: depth says how much context the attention
has to read, the prompt says how much work the pass does.

Prompts come from the harness's `synthetic-paragraphs` generator, which builds
random sentences of varying length from a fixed vocabulary and a fixed seed.
Consecutive prompts share no prefix. A generator that pads by repeating one
sentence is not usable here: repetition engages prompt-lookup paths and inflates
decode, and both engines cache a repeated prefix, so the measured rate
describes the new tail only.

## Devices

A result names the device by its hardware so it can be matched without knowing
which machine produced it.

| | Nimo Axis N161 | AYANEO Flip KB |
| --- | --- | --- |
| APU | Ryzen AI Max+ 395 | Ryzen 7 7840U |
| GPU | Radeon 8060S, `gfx1151`, 40 CUs | Radeon 780M, `gfx1103`, 8 CUs |
| RAM seen by the OS | 124.9 GiB | 25.2 GiB |
| Firmware VRAM carveout | 512 MiB | 6,144 MiB |
| GTT window | 114,688 MiB | 20,480 MiB |
| Pool HIP reports to a loader | 114,688 MiB | 20,480 MiB |

Both are unified-memory parts, so device memory and system RAM are the same
DRAM reached two ways. The pool a loader sees and that `mem_info_gtt_total`
reports is the GTT window; the firmware carveout is a separate, small
reservation. A model larger than the whole pool can be admitted and then read
pages that are not backed, which surfaces as non-finite logits rather than an
allocation error.

Each device measures its own depth ladder, because the pool bounds the context
each one can hold. The handheld tables (`single-ar-handheld` and its loading and
memory companions) use the contexts and depths the 20 GiB pool supports. A page
carries both devices by naming an artifact directory in the marker:
`<!-- bench:single-ar@artifacts-flip -->` renders the handheld measurement of
the same table.

## Between-run settling

Loading a model that occupies most of the pool, then immediately loading
another, can produce a broken first forward even when the second configuration
fits. The kernel needs time to reclaim the previous process's pages, and a
server that starts before it has done so gets an allocation that does not match
reality.

Two symptoms were measured on the Nimo Axis N161, both from the same cause:

- Non-finite logits at position 1. Laguna S 2.1 UD-Q5_K_XL at 262,144 tokens
  with q8_0 K/V failed this way when started 70 seconds after Laguna UD-Q4_K_XL
  released 90 GiB, then loaded unchanged after a longer wait and used 109,195
  MiB of the 114,688 MiB pool.
- A GPU queue error rather than a numerical one. Qwen3.8-Flash-Next at its
  native 262,144 aborted with `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT: The AQL
  packet is malformed` and `fatal_signal signal=6` on its first request, then
  completed three consecutive full sweeps at the same context once the device
  had settled.

Neither is a result. Space consecutive large loads by 150 seconds on the Nimo
and 90 on the handheld, and treat a first-forward NaN or a queue abort right
after a previous server exited as unsettled until it reproduces after the wait.
A model at 95% of the pool makes this more likely, not impossible to avoid.

## Coverage

Which model has measured results on which device. A device that cannot hold a
model is recorded as such rather than left blank.

| Model | Nimo Axis N161 | AYANEO Flip KB |
| --- | --- | --- |
| Qwen3.6-35B-A3B UD-Q4_K_XL | measured | measured, q8_0 K/V |
| Qwen3.8-27B UD-Q4_K_XL | measured | measured, q8_0 K/V |
| Qwen3.8-Flash-Next UD-Q4_K_XL | measured | not present |
| Laguna S 2.1 UD-Q4_K_XL | measured | not present |
| Laguna S 2.1 UD-Q5_K_XL | measured, q8_0 K/V | not present |

The other model pages describe models whose weights are not installed on either
device. They carry identity and method only. Measure one and its tables fill in
from the same harness.

## Adding a model

1. Write `docs/internal/models/<model>/artifacts/bench.json`. Copy an existing
   config and change `model`, `files`, `variants` and `tables`. Leave out
   `reference` to keep the page Gufo-only.
2. Add `<!-- bench:<table-id> -->` and `<!-- /bench -->` markers to
   `BENCHMARKS.md` where each table belongs. Use
   `<!-- bench:<table-id>@artifacts-flip -->` for a second device.
3. Measure, from the gufo tree:

   ```sh
   python3 tools/bench/model-bench.py --model <model> \
     --gufo "$PWD/build/release/gufo" \
     --gguf /path/to/model.gguf \
     run --target gufo --table single-ar,loading,memory --context 131072
   ```

   `--gufo` takes any Gufo binary; `result/bin/gufo` and `build/release/gufo`
   are the two a checkout produces. Pass `--draft`, `--mtp`, `--dspark` or
   `--mmproj` when a table's variant declares those files.

4. Render: `python3 tools/bench/model-bench.py --model <model> render`.

Paths given to `--gufo`, `--log-dir` and `--artifacts-dir` are resolved against
the gufo tree when relative, so pass absolute paths when running from elsewhere.

## Reading a table

`pp` is prompt-processing throughput in tok/s: how fast a new prompt is
consumed. `tg` is token generation in tok/s: how fast output is produced. Depth
is the cached prefix the row ran against.

Decode is bandwidth bound. It reads every active weight once per token, so it
falls slowly with depth and barely moves when the batch or the cache format
changes. Prefill is compute bound, so it is what batch, ubatch and cache format
actually move. A large decode swing between two runs of the same model points at
a different configuration or an unstable device, not at tuning.
