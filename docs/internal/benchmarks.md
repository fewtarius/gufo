# Benchmark methodology

How benchmark numbers are produced, compared and retained. The public summary
is [benchmarks](../benchmarks.md). Measure release binaries (`nix build`, then
`result/bin`). Per-model results and qualification gaps live in:

- [Laguna S 2.1](models/laguna/BENCHMARKS.md)
- [Qwen3.6-35B-A3B](models/qwen3.6-35b-a3b/BENCHMARKS.md)

- [DeepSeek V4 Flash](models/deepseek-v4-flash/BENCHMARKS.md)
- [EmbeddingGemma 2](models/embeddinggemma-2/BENCHMARKS.md)
- [MiniMax H3](models/minimax-h3/BENCHMARKS.md)
- [Qwen-Image-2.1](models/qwen-image-2.1/BENCHMARKS.md)
- [Qwen3.8 27B](models/qwen3.8-27b/BENCHMARKS.md)
- [Qwen3.8 Flash-Next](models/qwen3.8-flash-next/BENCHMARKS.md)
- [Qwen3-ASR](models/qwen3-asr/BENCHMARKS.md)
- [Qwen3-TTS](models/qwen3-tts/BENCHMARKS.md)

## Coverage by device

Which models have measured results on which device. "Does not fit" means the
weights alone exceed the device's GPU-visible pool, so no context makes it
usable there. A device is named by its product and APU, so a result can be
matched to hardware without knowing which machine it ran on.

| | Nimo Axis N161 | AYANEO FLIP KB |
| --- | --- | --- |
| APU | Ryzen AI Max+ 395 | Ryzen 7 7840U |
| GPU | Radeon 8060S, `gfx1151`, 40 CUs | Radeon 780M, `gfx1103`, 8 CUs |
| RAM visible to the OS | 124.9 GiB | 25.2 GiB |
| GPU-visible pool | 112.5 GiB (512 MiB carveout + 112 GiB GTT) | 26 GiB (6 GiB carveout + 20 GiB GTT) |

| Model | Nimo Axis N161 | AYANEO FLIP KB |
| --- | --- | --- |
| Laguna S 2.1 UD-Q4_K_XL (68.35 GiB) | measured, 128k | does not fit |
| Laguna S 2.1 UD-Q5_K_XL (82.02 GiB) | measured, 128k F16 / native 262k q8_0 | does not fit |
| Qwen3.6-35B-A3B UD-Q4_K_XL (21.28 GiB) | measured, 128k | measured, 128k q8_0 |
| Qwen3.8-27B UD-Q4_K_XL (16.35 GiB) | measured, up to 131k | spills to host, 3.8 tok/s |
| Qwen3.8-Flash-Next UD-Q4_K_XL (103.7 GiB) | measured, AR and MTP | does not fit |
| DeepSeek V4 Flash, MiniMax H3, Qwen-Image-2.1, EmbeddingGemma 2, Qwen3-ASR, Qwen3-TTS | see each model page | not run |

Peak device memory at the tested context, for the rows above:

| Model | Device | Context | K/V | Peak device | Pool |
| --- | --- | ---: | --- | ---: | ---: |
| Laguna S 2.1 Q4 | Nimo Axis N161 | 131,072 | F16 | 90,446 MiB | 115,200 MiB |
| Laguna S 2.1 Q4 | Nimo Axis N161 | 131,072 | q8_0 | 83,563 MiB | 115,200 MiB |
| Laguna S 2.1 Q5 | Nimo Axis N161 | 131,072 | F16 | 109,963 MiB | 115,200 MiB |
| Laguna S 2.1 Q5 | Nimo Axis N161 | 262,144 | q8_0 | 109,195 MiB | 115,200 MiB |
| Qwen3.6-35B-A3B | Nimo Axis N161 | 131,072 | F16 | 29,928 MiB | 115,200 MiB |
| Qwen3.6-35B-A3B | Nimo Axis N161 | 131,072 | q8_0 | 28,728 MiB | 115,200 MiB |
| Qwen3.6-35B-A3B | AYANEO FLIP KB | 131,072 | q8_0 | 26,351 MiB | 26,624 MiB |
| Qwen3.8-Flash-Next | Nimo Axis N161 | 133,121 | F16 | 84,774 MiB AR / 88,931 MiB MTP | 115,200 MiB |

Two rows are at the edge. Laguna Q5 with q8_0 fills 95% of the Nimo pool at
native context, which is why the cache format is what makes that context
reachable. Qwen3.6 on the FLIP KB uses 99% of the pool at 128k, so any larger
context needs the format change, not a tuning change.

Both machines are unified-memory parts, so "device memory" and "system RAM" are
the same DRAM reached two ways. The kernel's GTT window is the pool a loader
sees and what `mem_info_gtt_total` reports; the BIOS carveout is a separate,
small reservation. A model larger than the whole pool can be admitted and then
read unbacked pages, which surfaces as non-finite logits rather than an
allocation error - which is why the loaders carry budget checks.

## Between-run settling

Loading a model that occupies most of the pool, then immediately loading
another, can produce non-finite logits on the first forward even when the second
configuration fits. The kernel needs roughly a minute to reclaim the previous
process's pages. Measured repeatedly on the Nimo Axis N161: a run that failed
this way passed unchanged after a wait.

Allow about 75 seconds between back-to-back large loads, and treat a
first-forward NaN right after a previous server exited as a device state, not a
result, until it reproduces after the wait.


## Model benchmark driver

`tools/bench/model-bench.py` measures the tables of a model's `BENCHMARKS.md`
over HTTP for Gufo and for the reference project of its category (llama.cpp
for Qwen; antirez/ds4 for DeepSeek), writes per-table artifacts, and renders the tables
and SVG charts between the `<!-- bench:<id> -->` markers. Workloads and table
layouts are declared in `docs/internal/models/<model>/artifacts/bench.json`; the
`benchmark-model` skill in `.agents/skills` describes the procedure.

Reference runtimes are explicitly selected Nix packages, excluded from Gufo,
the normal development shell and hosted checks:

| Reference | Package | Executable |
| --- | --- | --- |
| Qwen AR / DFlash2 | `.#llama-cpp-reference` | `llama-server` |
| Flash-Next MTP | `.#llama-cpp-mtp-reference` | `llama-server-mtp` |
| DeepSeek AR / DSpark | `.#ds4-reference` | `ds4-server`, `ds4-bench` |

For example, enter the development shell and add only the needed baseline:

```sh
nix develop
nix shell .#llama-cpp-reference .#llama-cpp-mtp-reference
python3 tools/bench/model-bench.py --model qwen3.8-flash-next --gguf "$MODEL" \
  --mtp "$MTP" run --target reference --table single-mtp
```

DeepSeek's HTTP payload omits pp/tg durations. The driver reads the pinned
server's existing stage timers from its log and checks them against HTTP token
counts. Native `ds4-bench` also supports single-session AR/DSpark, but not
concurrency. The pinned ROCm server disables DSpark when batching, so C>1
DSpark comparisons are **N/A**. See the
[DS4 method](models/deepseek-v4-flash/QUALITY.md#benchmark-method).

New or refreshed text-model cards use the same pp2048 prose/copying prompts
for single-user d0 and concurrency, with tg128. The driver shares their prompt
generator and records prompt hashes. C1 should cross-check d0; investigate
input, cache, controller and timing differences before attributing a gap to
batching. Short repeated-word corpora remain useful quality probes, but their
rates cannot replace these matched workloads.

For decode comparisons, set `prefill_first: true` in the concurrency table.
Prepare every session with the full prompt and one output token, wait for all
preparations, then time identical tg128 requests with prefix reuse. The driver
pins llama.cpp slots and rejects replay exceeding four prompt tokens. This keeps
peers' long prefills outside llama.cpp's elapsed generation timer; Gufo reports
active decode time. Preserve a separate fresh AR completion-hash control.

## Direct and serving measurements

`gufo bench` measures the model path. Keep model artifact, prompt length,
generation length, context depth, speculative mode, and sampling controls
fixed when comparing runs. `pp2048` means 2,048 new prompt tokens; context
depth is separate from the amount of new prefill work.

`tools/serving/gufo-serving-bench.py` measures HTTP serving with synchronized
requests. It distinguishes server-stage prefill/decode throughput, scheduler
and client TTFT, token ITL, whole-request throughput, and aggregate concurrency
throughput. C1 serving and a direct single-user benchmark are comparable only
when prompt, cache state, sampling, speculation, and timed scope match.

Model tables report the **sum of individual request decode rates** in each
concurrent group, averaged across measured groups. Prefill and queue time
remain in the latency diagnostics. Pending measurements are **TODO**;
comparisons prevented by missing reference features are **N/A**, with a reason.
Whole-request throughput is not a substitute for decode timings.

`--endpoint-profile openai` supports other OpenAI-compatible servers. Missing
server-stage metrics remain null. `--reference-report` compares completion
hashes against an existing C1 reference; category filters select a focused
subset. Qualify acceptance with a fresh server so prompt-cache hits do not
hide work. See [serving benchmark commands](performance.md) for invocation.

`tools/bench/speculative-corpus.py` compares direct autoregressive and
speculative token traces. Request correctness, acceptance, and end-to-end
speed are separate requirements.

## Quality method (matched-token, per position)

Use the independent references and model-owned limits described in
[testing](testing.md). Tokenize once and feed teacher and candidate the same
predetermined history. Compare full next-token distributions before appending
the next evaluation token. Free-running text cannot isolate numerical error
after the first differing choice.

```text
p_t = softmax(teacher_logits)
p_c = softmax(candidate_logits)
KL  = sum_v p_t[v] * (log p_t[v] - log p_c[v])
NLL = -log(p_c[target_token])
```

Record non-finite counts, normalized-logit error, KL, teacher-token
NLL/perplexity, top-1 agreement, and top-k overlap as applicable. Aggregates
are over scored positions. A changed tokenizer, template, suite, or history
changes the comparison identity.

Capability evaluation is separate: [gufo eval](../cli.md#measuring) grades free-running
answers through the OpenAI-compatible server.

## Focused kernel measurements

Build development tools with `cmake --preset gpu-test` and
`cmake --build --preset gpu-test --target gufo-kernel-bench tune_hipblaslt`.
They are omitted from the production package.

`gufo-kernel-bench` exercises production HIP entry points without loading a
model. Correctness sentinels run outside the timed interval:

```sh
./build/gpu-test/gufo-kernel-bench --help
./build/gpu-test/gufo-kernel-bench \
  --case decode-attention --context 4096,8192,12288,16384 \
  --warmup 5 --repetitions 30 --json
```

Standalone microbenchmarks under `tools/bench` and model-specific tool folders
report component correctness alongside timing. Their numbers do not replace
end-to-end model measurements. Collect profiler traces separately and use
[the performance guide](performance.md) to identify the limiting operation.

## Reporting and artifact retention

Record model/engine revisions, artifact hashes, hardware/software fingerprint,
power profile and CPU/GPU temperatures, input identity, context, output count,
sampling, concurrency, cache state, oracle identity, and the exact timed scope.
A numerical mismatch or device failure is not a passing result. Add repetitions
when timing noise or a suspected regression requires them.

Checked-in JSON retains fixtures, model provenance, controller calibration,
and bounded result summaries. Serving summaries preserve aggregate metrics,
per-case completion hashes, and the original report digest. Raw samples,
profiler traces, generated tuning databases, and full-logit captures stay in
ignored `artifacts/`, except for independent reference data needed by a
maintained correctness test.

Serving reports exclude endpoint hosts, prompts, generated text, model paths,
raw token IDs, and timestamps. The canonical machine fingerprint excludes
hostnames, usernames, secrets, timestamps, and private paths.
`gufo diagnose --validate-artifact` validates its supported diagnostic schema;
each model/tool owns the validation of its own result schema.

Downloaded tokenizer files and `.direnv` shell state are local artifacts.
Count tracked source, including HIP translation units, with:

```sh
nix run nixpkgs#cloc -- --vcs=git --force-lang=C++,hip
```
