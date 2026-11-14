# Plan: the fit of a second model (Qwen 27B) beside DeepSeek

**Status:** 2026-11-13: the fast-model window is done (§5.5). Nemotron-3.5-Lightning-30B-A3B is the fastest and the
smallest candidate: 74 tok/s in 25.5 GiB with no draft, 127 tok/s in 28.8 GiB with its DSpark draft, at a 256k
context. DeepSeek decodes at 35.3 tok/s, and about 29 GiB is available. With no draft it fits today with a margin of
about 3.5 GiB. Qwen 27B does not fit (§5.1). Next: your choice of the setup, then Phases C and D ·
**Author:** colin-spark · **Date:** 2026-11-13
**Related:** `PLAN_MODEL_ROUTER.md` in the bot's repo (§3 the memory budget, §7 the fast model, §9 the run of both
servers), `PLAN_MD_DEPLOY.md` (the live server), `../serve-tune/PLAN_SERVE_TUNE.md` (§6 the KV pin, §8 the wide
drafts, §11 the memory budget), `PLAN_HEAD_VOCAB.md` (the vocabulary trim), `PLAN_FEWER_LAYERS.md` (the end of the
lossy memory track), `~/Projects/model-benchmarks` (the Qwen benchmark runs)
**Scope:** one measure script and its dry run in this repo, one short test window, and a decision table. The
savings on the DeepSeek side (§6) and the run of both servers (§7) each get their own pull requests. No change to the
live server in Phases A and B.

| Step | Status |
|---|---|
| §4 Phase A: measure Qwen alone (about 45 min downtime) | ✅ 2026-11-13, 16:44 to 17:33 (§4.4) |
| §5 Phase B: the arithmetic and your decisions (no downtime) | ✅ 2026-11-13: the arithmetic (§5.1); your decision is open (§5.2) |
| §6 Phase C: the savings on the DeepSeek side (one window each) | ⬜ |
| §7 Phase D: both servers at the same time | ⬜ |

---

## 1. Short answer

- **The goal:** `nvidia/Qwen3.8-27B-NVFP4` with a context of 256,000 tokens, beside the live DeepSeek checkpoint (the
  MD prune), on one DGX Spark with 121 GiB of shared memory. You accept a free margin below 10 GiB.
- **It does not fit today.** Qwen needs 20.4 GiB of weights and 8.0 GiB of KV cache: 28.4 GiB before its runtime
  overhead. The live server leaves about 29 GiB available. So the margin is below 1 GiB before the overhead counts.
- **One number is missing:** the runtime overhead of Qwen at one sequence. Your benchmark runs used up to 13.5 GiB,
  but with 16 sequences. Phase A measures it, in about 30 minutes of downtime.
- **The gap** is the overhead plus the margin minus 0.9 GiB. With an overhead of 4 GiB and a margin of 4 GiB, DeepSeek
  must free about 7 GiB.
- **The savings with no output change** on the DeepSeek side give about 6 GiB, or about 8 GiB without the draft
  model. The largest one has a known crash risk. So the fit is possible, but it is tight.
- **The largest single lever is on the Qwen side:** a context of 128,000 tokens needs 4.0 GiB less KV cache.
- **The lossy track is closed.** DeepSeek sits on the replay gate (`PLAN_FEWER_LAYERS.md` §8.5). No more experts or
  layers can go.

## 2. The numbers (measured)

### 2.1 DeepSeek, the live server (2026-11-13)

| Part | GiB | Source |
|---|---|---|
| Weights of the target | 66.54 | The probe boot with no draft (`PLAN_FEWER_LAYERS.md` §7.7) |
| Draft model (64 experts) | 3.25 | 69.79 (the live boot) minus 66.54 |
| KV cache, pinned, 256k context | 7.25 | `compose-md149t5.yml` |
| Runtime memory of the GPU process beyond these | 5.03 | `nvidia-smi`: 84,036 MiB for the process |
| Host memory of the server processes | 3.9 | `docker stats` |
| **Available beside the idle server** | **28 to 29** | `free`, with one Claude Code session open |

### 2.2 Qwen 27B NVFP4

| Part | Value | Source |
|---|---|---|
| Weights on disk | 20.42 GiB | The safetensors files: 18.1 GiB language model, 0.86 GiB vision encoder, 0.79 GiB MTP layers, 0.66 GiB head |
| Weights loaded | 19.83 to 20.75 GiB | "Model loading took" in the benchmark logs (with and without the MTP draft) |
| KV cache | 33.6 KB per token (FP8) | 16 GiB held 512,023 tokens. Only 16 of the 64 layers use full attention |
| KV cache for 256,000 tokens | 8.00 GiB | 256,000 × 33.6 KB |
| KV cache for 128,000 tokens | 4.00 GiB | The same |
| Runtime overhead | At most 13.5 GiB | `memory_samples.csv` of run `20261109_143107`: 49.9 GiB used in all, with 16 sequences, a prefill budget of 8,192 tokens and a 16 GiB KV pin |
| Server | vLLM 0.33.0 in `~/venvs/vllm033` | `docs/dgx-spark.md` of `model-benchmarks` |

The overhead at one sequence is not measured. It is the unknown X of this plan.

### 2.3 The fit

Qwen needs 28.4 GiB + X at 256k. The available memory is about 29.3 GiB. So:

**The saving that DeepSeek must give = X + the margin − 0.9 GiB.**

| Overhead X | Margin 4 GiB | Margin 6 GiB | Margin 4 GiB, Qwen at 128k |
|---|---|---|---|
| 3 GiB | 6.1 | 8.1 | 2.1 |
| 4 GiB | 7.1 | 9.1 | 3.1 |
| 6 GiB | 9.1 | 11.1 | 5.1 |
| 8 GiB | 11.1 | 13.1 | 7.1 |

## 3. The levers

### 3.1 On the DeepSeek side, with no output change

| # | Lever | Saving (GiB) | Cost or risk | State |
|---|---|---|---|---|
| D1 | A context cap of 160k (now 256k), with the KV pin at 1.04x of the cap | About 1.1 | A prompt above 160k fails. The largest calibration prompt has 133k tokens | Estimate of `PLAN_SERVE_TUNE.md` §3; no boot yet |
| D2 | A narrower draft: 32 experts (now 64) | About 1.2 | A slower decode. The 128-expert draft gained 4.2%, so the loss is likely of that size | Not tested |
| D3 | No draft (`MODE=mtp0`) | 3.25 (in place of D2) | A much slower decode: the draft gets about 2.2 tokens accepted per step | The probe boot ran this way |
| D4 | A smaller prefill budget: `MAX_NUM_BATCHED_TOKENS` 2048 (now 8224) | Up to 3.9 | A trial in the server repo crashed under real traffic: "Workspace is locked" (issue #4 there). It needs a patch that reserves the right MLA scratch at the warmup | A study, then a patch |

D1, D2 and D4 together give about 6.2 GiB. With D3 in place of D2, they give about 8.3 GiB.

### 3.2 On the DeepSeek side, with an output change

| # | Lever | Saving (GiB) | State |
|---|---|---|---|
| D5 | A smaller head vocabulary (`PLAN_HEAD_VOCAB.md`, variant T) | 0.62, and about 5% decode speed | Planned. The logits of the kept tokens do not change, so the cost can be near zero. The offline study needs no downtime |

Every other lossy lever needs a budget that the replay gate does not have: more cold experts, a layer half, 2-bit
experts, attention in EXL3.

### 3.3 On the Qwen side

| # | Lever | Saving (GiB) | Cost |
|---|---|---|---|
| Q1 | One sequence and a small prefill budget (`--max-num-seqs 1`, `--max-num-batched-tokens 2048`) | Unknown: Phase A measures it | A slower prefill of long prompts |
| Q2 | No speculative decoding (no MTP draft) | About 0.8, and its graphs | A slower decode |
| Q3 | Text only, if vLLM 0.33 then skips the vision encoder | Up to 0.86 | No images. Phase A checks whether the load size falls |
| Q4 | A context of 128k | 4.0 | Your decision. You plan 256k |

## 4. Phase A: measure Qwen alone (about 45 min downtime)

### 4.1 The script: `qwen_fit.sh`

It uses `run_lib.sh` for the deadlines, the stop of the live server, the memory watch and the restore. It boots Qwen
with the `vllm` of `~/venvs/vllm033` as a plain process on port 8889, with no restart. Before each boot, no other
`vllm serve` process runs.

For each point, it records:

- the available memory before the boot, after the health check, and the lowest value in each test (one sample per
  5 seconds);
- the memory of the process: `nvidia-smi` and the RSS;
- the lines "Model loading took" and "GPU KV cache size" of the server log;
- the free memory that `torch.cuda.mem_get_info` gives before the boot (for §7.2).

The tests of a point:

1. One short request, and one tool call.
2. One prompt of about 250,000 tokens with 500 output tokens: the peak of a full context.
3. Five bot turns from the replay data, for the decode speed.

**The overhead X** of a point is the available memory before the boot, minus the lowest value in the tests, minus
the loaded weights and the KV pin.

### 4.2 The points

All points: `--max-model-len 256000`, a KV pin of 8.3 GiB (1.04x of the need, as on the live server),
`--kv-cache-dtype fp8_e4m3`, `--max-num-seqs 1`, no prefix cache change, and the parsers of the benchmark config.

| Point | Prefill budget | Draft | Vision | Question |
|---|---|---|---|---|
| F1 | 2,048 | Off | Off (`--limit-mm-per-prompt` with zero images, if 0.33 accepts it) | The smallest setup: X at its floor |
| F2 | 8,192 | Off | Off | What the prefill budget costs |
| F3 | 2,048 | On (MTP, K 3) | Off | What the draft costs, and what it gains |

If the KV pin is too small for 256,000 tokens, vLLM stops the boot and prints the need. The script then boots once
more with that value plus 4%.

### 4.3 The tests and the gate

- `bash -n` passes. A dry run with stub boots (`test_qwen_fit.py`, as `test_skip_scripts.py`) covers a normal
  window, a failed boot, two failed boots in a row, and a pin that is too small. Two more tests check the report
  arithmetic and the size of the long prompt. Result 2026-11-13: 6 tests pass.
- The bot decode test is `model_eval.speed` of the bot's repo with 5 sessions and 1 run, so its rate compares with
  the DeepSeek numbers.
- The window boots use `--gpu-memory-utilization 0.30`. With a pinned KV cache, this value sets only the boot check
  of §7.2.
- A start with a boot deadline in the past ends before any server action.

**Gate.** Each point gives its four memory values, and the live server is healthy after the window.

CAUTION: The window stops the bot's model server for about 45 minutes (three points of about 15 minutes; the script
skips a point that cannot end before `RESTORE_AT`). Do not start it between 05:00 and 10:00 on a
weekday while the cron jobs are active.

### 4.4 Result (2026-11-13, 16:44 to 17:33, on your "start the qwen window")

The live server stopped at 16:44 and was healthy again at 17:33:03 (reply "OK.", gateway active). The downtime was
49 minutes. A first start at 16:41 ended at once and cost 2 more minutes: the process check of the script matched the
text of a shell command, not a server (fixed: the check now matches a real vLLM process only). Files:
`logs/qwen-fit.log`, `results/qwen-fit-f{1,2,3}*` (not in git).

| Point | Weights | KV pin (tokens) | Used after the boot | Used at the peak | Overhead X | GPU process at the end | Bot decode | 250k prompt |
|---|---|---|---|---|---|---|---|---|
| F1: budget 2,048, no draft | 18.96 | 8.32 (257,514) | 33.2 | **33.95** | **6.67** | 29.4 GiB | 11.09 tok/s | 327 s |
| F2: budget 8,192, no draft | 18.96 | 8.32 (257,514) | 33.6 | 35.67 | 8.39 | 31.1 GiB | 11.12 tok/s | 324 s |
| F3: budget 2,048, MTP draft | 19.75 | 9.60 (266,067) | 36.1 | About 37 (estimate) | About 7.5 (estimate) | 31.9 GiB | 30.75 tok/s | 325 s |

All values are GiB unless a unit is given. Every smoke test passes (a short request and a tool call).

- **The no-images flag works.** The load is 18.96 GiB, 0.87 GiB less than in the benchmark runs: vLLM skips the
  vision encoder.
- **The overhead at one sequence is 6.7 GiB**, not the 3 to 4 GiB that the fit needs. The larger prefill budget adds
  1.7 GiB and gives no prefill speed on one long prompt (324 s against 327 s, about 765 tokens per second).
- **The draft needs a larger KV cache.** With the first pin, F3 did not boot: vLLM asked for 9.23 GiB for 256,000
  tokens. The script booted once more with 9.60 GiB, as designed.
- **The F3 peak is an estimate.** The boot retry stopped the memory sampler (a fault of the script, fixed now), so
  F3 has no samples from its tests. The estimate is its use after the boot plus the rise of F1 (0.76 GiB).
- **The load has a short, larger peak.** About 15 seconds into each boot, the available memory fell far below the
  later level: 41.7 GiB used in F3 and 32.3 GiB in F2, before the KV cache existed. The samples are 5 seconds apart,
  so the real peak can be higher. The cause is most likely the loader (`--load-format fastsafetensors` holds the
  file bytes in host memory while it copies them). Beside DeepSeek this peak alone is larger than the available
  memory, so a pair needs another loader. Not tested.
- **The speed.** `model_eval.speed` failed in its summary for F1 and F2 (it needs the counters of a draft). Its rows
  were saved, and the rates come from them: 5 sessions, 1 run each. With the draft, Qwen accepts 2.59 tokens per
  draft step. The live DeepSeek server decodes the same test at about 35.3 tok/s.

## 5. Phase B: the arithmetic and your decisions (no downtime)

### 5.1 The arithmetic (2026-11-13)

About 29 GiB is available beside the live server. The saving that DeepSeek must give is the use of Qwen at the peak,
plus the margin, minus 29 GiB:

| Qwen setup | Use at the peak | DeepSeek must free (margin 4 GiB) | Decode |
|---|---|---|---|
| 256k, no draft | 34.0 | 9.0 | 11.1 tok/s |
| 256k, with the draft | About 37 | About 12 | 30.8 tok/s |
| 128k, no draft (KV 4.16) | About 29.8 | About 4.8 | 11.1 tok/s |
| 128k, with the draft (KV 4.80) | About 32.2 | About 7.2 | 30.8 tok/s |

The 128k rows are computed from the measured KV size per token, not booted.

The DeepSeek savings with no output change give at most 6.2 GiB, or 8.3 GiB with no DeepSeek draft (§3.1). So:

1. **Qwen at 256k does not fit beside this DeepSeek checkpoint,** with or without its draft. This is the last row
   of the table below.
2. **Qwen at 128k with no draft fits on paper** if D1, D2 and D4 all work. It then decodes at 11 tok/s, a third of
   the DeepSeek rate.
3. **Qwen at 128k with its draft** needs 7.2 GiB: every DeepSeek saving, and no DeepSeek draft. Both models are then
   slow.
4. Each case also needs a loader with no load peak, and the boot check of §7.2.

### 5.2 Your decision

The router plan asks for a fast model: at least 2x the decode speed of DeepSeek (§7.1 there). Qwen 27B is slower
than DeepSeek on this machine, with its draft (30.8 against 35.3 tok/s) and much slower without it. So the question
is what the second model is for.

| Option | What it means |
|---|---|
| A. A swap, not a pair | One model in memory at a time. Qwen boots in about 70 s and DeepSeek in about 150 to 180 s. Qwen then gets its full 256k context and its draft. This is fallback 1 of the router plan (§4 there) |
| B. A smaller second model | A model that needs about 20 GiB in all, for example a MoE with few active parameters. It can be faster than DeepSeek, which Qwen 27B is not |
| C. Qwen at 128k with no draft, beside DeepSeek | Phase C with D1, D2 and D4 (three windows, one with a known crash risk), then Phase D. The result is a second model at 11 tok/s with a margin of about 4 GiB |
| D. Stop | DeepSeek alone, as now |

### 5.4 The fast-model candidates (2026-11-13)

**Your goal (2026-11-13):** a routing layer that sends a request to a strong reasoning model (DeepSeek) or to a model
that answers very fast. Priority: the decode speed, with enough context. This is option B of §5.2.

**Why a MoE with few active parameters.** The decode on this machine is bound by the bytes that one token reads.
The dense Qwen 27B (19 GiB) decodes at 11 to 12 tok/s. A dense model then needs under about 5B parameters to pass
70 tok/s. A MoE with about 3B active parameters reads a small part of its weights per token.

| Candidate | Weights on disk | Decode on this machine | Context | KV cache at 256k (estimate) | State |
|---|---|---|---|---|---|
| `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` | 22.0 GiB, with a vision encoder | 85.6 tok/s with the MTP draft, 74.9 without (July 2026, vLLM 0.25.1, short prompts). The nvidia build: 77.7 to 79.0 | 262k | 2.7 GiB: 10 of 40 layers use full attention, 10.5 KB per token | Measured here in July, with 16 sequences |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 20.1 GiB, text only; its DSpark draft: 1.26 GiB | Not measured here. Its card recommends the DSpark draft on the DGX Spark | 1M | 0.8 GiB: 6 of 52 blocks are attention, 3.1 KB per token | Not measured |

DeepSeek decodes the bot test at 35.3 tok/s, so the first candidate is at 2.1x to 2.4x. The router plan asks for 2x
(§7.1 there).

**The fit on paper.** Qwen 35B-A3B: about 21 GiB of weights with no images, 2.7 GiB of KV cache, and an overhead of
5 to 7 GiB (the 27B had 6.7): 29 to 31 GiB, against about 29 GiB available. So the margin is near zero today, and a
margin of 4 GiB needs 4 to 6 GiB from DeepSeek (§3.1). Nemotron needs about 1 to 2 GiB less on paper. The context
is not the lever for these models: 128k saves 1.3 GiB at most.

**What is not known.** The quality of both models on the bot's skills (the gate of the router plan §7.3), the speed
on the bot's long prompts, and the memory at one sequence. The tool accuracy of the July suite (39% to 41% for the
35B-A3B) used a scoring that was fixed later, so it does not compare with the 27B.

**The points** (`qwen_fit.sh`, `POINTS='q1 n1 q2 n2'`, about 40 minutes of downtime):

| Point | Model | Draft | Flags that differ |
|---|---|---|---|
| q1 | Qwen 35B-A3B | None | `--moe-backend flashinfer_cutlass`, `--tool-call-parser qwen3_xml`, `--language-model-only` (the July run with the best speed) |
| q2 | Qwen 35B-A3B | MTP, K 3 | The same |
| n1 | Nemotron | None | `--moe-backend marlin`, `--mamba-backend flashinfer`, `--reasoning-parser nemotron_v3`, `--tool-call-parser qwen3_coder` (the DGX Spark recipe of the model card) |
| n2 | Nemotron | DSpark, K 3 | The same |

All four use a 256k context, one sequence, a prefill budget of 2,048 tokens and the default loader. The order
q1, n1, q2, n2 keeps one bad model from the rule of two failed boots in a row. Each point gives the memory at the
peak, the peak of the load, and the bot decode rate. vLLM 0.33 accepts both command lines (its argument parser, no
boot). The gate of the window: the model with the best bot decode rate whose use at the peak leaves a margin of 4 GiB
after the DeepSeek savings that you accept.

### 5.5 Result of the fast-model window (2026-11-13, 18:27 to 19:14, on your "start the window")

The live server stopped at 18:27 and was healthy again at 19:13:58 (reply "OK.", gateway active). The downtime was
46 minutes. Files: `logs/fit-fast.log`, `results/qwen-fit-{q1,n1,q2,n2}*` (not in git). All four points booted,
passed the smoke test (a short request and a tool call) and took the 250,000-token prompt.

| Point | Weights | KV pin for 256k | Used at the peak | Peak of the load | Bot decode | Against DeepSeek (35.3) | 250k prompt |
|---|---|---|---|---|---|---|---|
| q1: Qwen 35B-A3B, no draft | 19.65 | 2.75 | 28.81 | 28.17 | 57.9 tok/s | 1.6x | 138 s |
| q2: Qwen 35B-A3B, MTP draft | 21.22 | 3.22 | 32.29 | 32.58 | 99.3 tok/s | 2.8x | 144 s |
| n1: Nemotron, no draft | 17.82 | 0.92 | **25.51** | 26.37 | **74.1 tok/s** | 2.1x | 82 s |
| n2: Nemotron, DSpark draft | 19.37 | 2.09 | **28.78** | 29.92 | **127.2 tok/s** | 3.6x | 77 s |

Memory values are GiB. The bot decode is `model_eval.speed` with 5 sessions and 1 run (prompts of 30,000 to 73,000
tokens), so each rate is one measurement.

- **Nemotron wins on every axis:** it is faster than the Qwen 35B-A3B with and without a draft, it needs 3.3 to
  3.5 GiB less, and its prefill is faster (about 3,100 against 1,900 tokens per second on the long prompt).
- **The drafts work.** The MTP draft of Qwen accepts 2.28 tokens per step, and the DSpark draft of Nemotron 2.10.
  Each costs memory: 3.5 GiB for Qwen and 3.3 GiB for Nemotron (weights, a larger KV need and overhead).
- **The first KV pins were too small at all four points** (the needs: 2.64, 3.10, 0.88 and 2.01 GiB). The script
  booted once more each time, as designed. A draft raises the KV need.
- **The default loader has no large peak at the load:** at most 1.1 GiB over the steady use. The peak of the 27B
  boots came from `--load-format fastsafetensors`. A boot with the default loader takes about 3.5 minutes.
- **The overhead is 6.4 to 7.9 GiB at every point,** as for the 27B (6.7). So the overhead is a cost of the vLLM
  process on this machine more than of the model.
- **The bot prompts are slower than short prompts:** the Qwen 35B-A3B gave 57.9 tok/s here and 74.9 in July.

**The fit** (about 29 GiB available beside DeepSeek; the saving that DeepSeek must give for a margin of 4 GiB, by the
steady peak and by the peak of the load):

| Setup | Margin today | DeepSeek must free | Which savings (§3.1) |
|---|---|---|---|
| Nemotron, no draft (74 tok/s) | About 3.5 GiB | 0.5 to 1.4 GiB | D1, the context cap (1.1 GiB), or nothing with a margin of 3 GiB |
| Nemotron, DSpark draft (127 tok/s) | About 0.2 GiB | 3.8 to 4.9 GiB | D1, D2 and D4; or D1 and D3 (no DeepSeek draft) |
| Qwen 35B-A3B, no draft (58 tok/s) | About 0.2 GiB | 3.8 GiB | As above, for less speed |
| Qwen 35B-A3B, MTP draft (99 tok/s) | None: 3.3 GiB over | 7.3 GiB | D1, D3 and D4 |

**The choice.** Nemotron-3.5-Lightning-30B-A3B is the second model. With no draft it fits now. With its DSpark draft
it is 1.7x faster again, and it needs about 4 to 5 GiB from DeepSeek.

**What is not measured:** the quality of Nemotron on the bot's skills (the gate of the router plan §7.3). The window
proves only that it answers and calls a tool. Each speed is one run.

### 5.3 The decision table (written before Phase A)

1. Put X of F1 into the formula of §2.3. Done (§5.1).
2. Read vLLM 0.33 for the boot check of §7.2. No downtime. Done 2026-11-13 (§7.2).
3. You decide three values:
   - the margin (the proposal: at least 4 GiB, the small margin of the router plan);
   - the context of Qwen (256k, or 128k for 4.0 GiB);
   - the context cap of DeepSeek (256k, or 160k for 1.1 GiB).
4. The table below then gives the work of Phase C.

| The saving that DeepSeek must give | Work |
|---|---|
| At most 0 | None. Go to Phase D |
| At most 2.3 GiB | D1 and D2 |
| At most 6.2 GiB | D1, D2 and D4 |
| At most 8.3 GiB | D1, D3 and D4. The decode gets much slower: your decision |
| More | Qwen at 128k (Q4), a smaller second model, or the swap fallback of the router plan (§4 there) |

D5 (0.62 GiB) can replace a small part of any row, after its offline study.

## 6. Phase C: the savings on the DeepSeek side

Each lever is a change of the live server, so each gets its own pull requests, its test window on port 8890 and your
go for the deploy. None changes the output, so the gate is the smoke test and the bot decode × 5 against a reference
of the same window (noise floor 4.2%), not the replay.

| Lever | Work | Gate |
|---|---|---|
| D1 context cap | One test boot with the new `MAX_MODEL_LEN` and pin | The boot log gives 1.02x or more for the cap |
| D2 narrower draft | Build drafts of 32 and 48 experts with the method of `PLAN_SERVE_TUNE.md` §8. One window: 64, 48, 32 | Your decision on the speed loss. Report the accepted tokens per step |
| D3 no draft | One point of the D2 window with `MODE=mtp0` | Your decision on the speed loss |
| D4 prefill budget | Read the reserve code of the MLA scratch and issue #4. Patch the warmup so that it reserves the largest scratch that a real prompt needs. One window with a 250k prompt, a deep warm turn and the smoke test | No "Workspace is locked" assertion, and the measured fall of the process memory |

Do D4 first if Phase B needs it: it is the largest lever and the least certain one.

## 7. Phase D: both servers at the same time

### 7.1 The setup

- DeepSeek stays on port 8888 with its compose file. Qwen runs on port 8889 with its own compose file or unit, with
  `restart: on-failure:1` or no restart.
- Start DeepSeek first. Start Qwen only when DeepSeek is healthy and idle.
- A memory watch runs for the whole test. Below 2 GiB available, it stops Qwen, not DeepSeek.

CAUTION: Out of memory, the Spark can do a hard reset. Do not start Qwen beside DeepSeek before Phase B shows a
margin of at least 4 GiB on paper.

### 7.2 The boot check

vLLM 0.33 checks the free GPU memory at boot, also with a pinned KV cache (`vllm/v1/worker/utils.py`,
`request_memory`): the boot fails if the free memory is below `--gpu-memory-utilization` × 121.7 GiB. On this machine
the free value leaves out the page cache. Beside the live server, torch saw 1.1 GiB free while 28 GiB was available
(2026-11-13). So the boot of Qwen beside DeepSeek needs one of these, and Phase D tests which one works:

1. A very small `--gpu-memory-utilization`, for example 0.005 (0.6 GiB). With a pinned KV cache, vLLM uses the value
   for this check only. The kernel then frees the page cache when Qwen allocates.
2. A drop of the page cache before the boot. This needs `sudo`, so you run it:
   `sudo sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches'`. Then set the utilization below the free share.

### 7.3 The tests

1. Both idle: the available memory.
2. Qwen with a prompt of 250,000 tokens, while DeepSeek decodes a bot turn.
3. DeepSeek with a prompt of 128,000 tokens, while Qwen decodes.
4. One full bot session on each model, with Chromium open (§9.3 of the router plan).
5. The decode speed of each model alone and of both at the same time.

**Gate.** The lowest available memory in every test is at least your margin. Then the 24-hour gate of the router
plan (§9.3 there): no out-of-memory event with the cron jobs and normal use.

## 8. Risks

| Risk | Effect | Guard |
|---|---|---|
| X is large (8 GiB or more) | No set of levers gives the fit at 256k | Phase A comes first and costs 30 minutes. The last row of §5 |
| D4 crashes under real traffic, as in the server repo | The bot fails on a long prompt | The 250k prompt and the deep warm turn in its window. A rollback is one compose file |
| Both servers peak at the same time | Out of memory, a hard reset | The margin, the memory watch that stops Qwen, no automatic restart loop |
| The host uses more memory than today (Chromium, CLIs) | The margin shrinks | Measure the margin with Chromium open. The router plan counts 5 GiB for the host |
| The boot check of vLLM refuses Qwen beside DeepSeek | Qwen does not boot | §7.2 |
| The page cache drop slows the next DeepSeek boot | A slower restore | Only the boot time changes: the weights are read from the disk again |
| Both models share the memory bandwidth | Both decode slower together | Test 5 of §7.3 gives the numbers |

## 9. Timeline

| Phase | Work | Downtime |
|---|---|---|
| A | Done 2026-11-13: the script, its dry run, and the window (16:44 to 17:33) | 51 min |
| B | The arithmetic is done (§5.1). Your decision is open (§5.2) | None |
| C | One to three windows, by the table of §5 | About 1 to 2 h each |
| D | One evening for the tests, then 24 hours of normal use | None, if the margin holds |

Phase A decides whether the rest is worth the work. If X is 8 GiB or more, the plan ends there, or goes on with Qwen
at 128k.
