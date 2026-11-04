# gufo-laguna

This is a fork of [gufo-org/gufo](https://github.com/gufo-org/gufo) that adds
the **Laguna S 2.1** model.

It is not a replacement for upstream Gufo. Upstream remains the engine; this
fork carries the Laguna model family, the pieces of shared infrastructure
Laguna needs that upstream does not have, and the changes to upstream's own
files that Laguna's presence requires.

## What this fork adds

| | |
|---|---|
| **Laguna S 2.1** | A MoE text model on gfx1151: 48 layers, hidden 3,072, 8 KV heads of 128, 256 experts with 10 routed plus a shared expert, sliding-window attention of 512 on 3 of every 4 layers. See [docs/models/laguna/](docs/models/laguna/README.md). |
| **Block-sparse attention** | Quest-style block selection on the 12 full-attention layers. A prefill win that grows with depth: +52% at 141k tokens. |
| **DFlash2 speculative decoding** | The drafter predicts a block of tokens from the target's tapped hidden states; each is verified against the target's own logits. |
| **Native tool calls** | Laguna's `<tool_call>{name}<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` dialect, compiled into the sampler as a grammar and read back by a scanner. |
| **KV row codec** | A model-agnostic q8_0/q4_0 row codec in `src/core/kv_codec/`, with the contract in [docs/kv-codec.md](docs/kv-codec.md). The Laguna engine and the Qwen runner can store their K/V cache as q8_0. See [Laguna](docs/models/laguna/KV-Q8.md) and [Qwen](docs/models/qwen3.8-27b/KV-Q8.md). |
| **Sampled-request speculation** | The server drafts for sampled requests as well as greedy ones, using the prompt CLI's acceptance rule. |

## What this fork does not carry

- **The NPU stack.** Upstream removed it. The Laguna NPU path was worth
  +0.2% to +2.4%, and re-vendoring the XCLBIN blobs to keep that was not
  worth it.
- **Flint and GLM-5.3.** Models from the fork this work started from, which
  upstream does not have and this fork does not need.
- **The Flash-Next performance series.** Upstream took its own.

## How to build

```sh
source ../scripts/env.sh rocm
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON \
  -DENGINE_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build --parallel
```

Or through the fork's own wrapper, if present, which sets the ROCm paths:

```sh
../scripts/env.sh rocm
cmake --build build --parallel
```

Otherwise use the standard build documented in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md):

```sh
nix build .#release
```

## How to run

```sh
gufo serve llm --host 0.0.0.0 --port 9090 \
  --model Laguna-S-2.1-UD-Q4_K_M.gguf \
  --speculative dflash2 --dflash-model Laguna-S-2.1-DFlash-BF16.gguf -d 2
```

`LAGUNA_KV_CACHE_TYPE=q8_0` stores the attention K/V cache as q8_0 instead of
F16. See [docs/models/laguna/KV-Q8.md](docs/models/laguna/KV-Q8.md).

## Layout

The Laguna model follows upstream's extension contract, so it is laid out the
way every other model is:

```
src/models/laguna/
  extension.json        the manifest: providers, targets, options, docs
  extension.cmake       the build definition
  provider.cpp          registers the provider with ext::Registry
  probe.cpp             recognises a Laguna GGUF
  runner.cpp            the ext::TextModelRunner implementation
  runner.hpp            the runner header
  drivers.cpp           the bench, prompt and chat CLI drivers
  runtime/config.cpp    the GGUF config parser
  chat_template.cpp     the chat template
  tool_calls.cpp        the native tool-call scanner
  reference.cpp         the CPU reference implementation
  gpu/                  the GPU engine and its kernels
```

## Changes to upstream's own files

These are the edits upstream's tree needs for Laguna to exist. Each one is
small and self-contained:

| File | Why |
|---|---|
| `src/core/json_constraint.{hpp,cpp}` | The native tool-call dialect, and a `qwen_like` generalization upstream does not have. |
| `src/ext/text.hpp` | A `prefill_grid_tokens` capability, so a model whose prefill chunks are wider than the default aligns its checkpoints to them. Upstream hardcodes 2048. The dialect contract also carries the per-argument and per-call scan offsets the recovery path needs. |
| `src/cli/serve/text_model_runner.cpp` | Consumes that capability. |
| `src/cli/serve/openai_chat.cpp` | The Laguna parser reaches the model's scanner through the dialect contract instead of naming the model. |
| `src/models/qwen/hip/dflash.hpp`, `src/models/qwen/hip/dflash/{executor.cpp,model.cpp}` | Device-side target-context injection, a rewind of the injected frontier, a persistent snapshot of that frontier, and a creation overload for a target that is not a `QwenGpuModel`. |
| `src/models/qwen/dflash_weights.{hpp,cpp}` | DFlash v1 support, which upstream dropped. The Laguna drafter is a v1 model. |
| `src/families/qwen/kernels/dflash_kernels.{hpp,hip}` | The v1 attention gate and the per-tap RMSNorm. |
| `src/families/qwen/text/tokenizer.{hpp,cpp}` | The Laguna control tokens and the Laguna pre-tokenizer, both of which upstream's tokenizer rewrite dropped. |
| `src/models/qwen38_flash_next/kernels/rocm/**` | The richer MMQ surface the Laguna engine uses, the paired routed gate/up GEMM with a selectable tile width, and the routed GEMM tile widths the Laguna experts ask for. |
| `CMakeLists.txt` | The KV codec in `gufo_core`, its test, the Laguna tests, and fixes to upstream's own link errors (see below). |
| `tests/models/deepseek_v4_flash/CMakeLists.txt` | A fix to upstream's own link error. |
| `tests/models/qwen27b/serving_test.cpp` | A fix to upstream's own API drift. |

### Upstream's own breakage this fork fixes

Besides the two link errors below, upstream's extension refactor left two
tests calling APIs it had removed: the DeepSeek V4 Flash serving test called
a model-specific `load` overload that moved into the model's runner factory,
and the Qwen3.8-27B GPU serving test called a `QwenDFlashGpuModel::Open`
that no longer exists and defined an incomplete `SpeculativeOptions`. Both
are fixed here.

### Two upstream link errors this fork fixes

`src/cli/extension_diagnostics.cpp` was moved out of the library that
`src/cli/diagnose/diagnose.cpp` lives in, so every target linking the
diagnostics library failed to link with an undefined `ExtensionDiagnostics`.
And the DeepSeek V4 Flash tests linked the CPU tier while calling into the
HIP tier, so they failed with undefined `ds4_*` symbols. Both are fixed
here, which makes this fork's test suite complete where upstream's is not.

## Keeping up with upstream

Upstream moves fast. The merge base when this fork was created was
`d0f7c3a0`; upstream is hundreds of commits ahead.

- Rebase onto `upstream/main` before starting anything that touches shared
  infrastructure. The Laguna-specific files conflict rarely.
- The shared-infra edits above are where conflicts land. Keep them as
  separate commits from the Laguna work, so a conflict resolution is a
  reviewable diff rather than a merge.
- Upstream deleted the NPU stack and refactored the serve layer into
  `src/ext/`. A future upstream refactor of that layer will hit
  `runner_body.inc` and the DFlash executor first.

## Provenance

The Laguna model support was ported from
[mikealanni/gufo](https://github.com/mikealanni/gufo) (MIT), where the model
was first brought up. The KV row codec's numeric contract was derived from
the [Gufetto](https://github.com/fewtarius/gufetto) fork's Flash-Next q8 KV
work; the implementation here is independent. See `NOTICE` and
`THIRD_PARTY_NOTICES.md`.
