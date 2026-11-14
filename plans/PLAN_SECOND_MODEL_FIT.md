# Plan: the fit of a second model (Qwen 27B) beside DeepSeek

**Status:** 2026-11-13: plan written; the numbers of §2 are measured; nothing built ·
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
| §4 Phase A: measure Qwen alone (about 30 min downtime) | ⬜ |
| §5 Phase B: the arithmetic and your decisions (no downtime) | ⬜ |
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

## 4. Phase A: measure Qwen alone (about 30 min downtime)

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

- `bash -n` passes. A dry run with stub boots (as `test_skip_scripts.py`) covers a normal window, a failed boot, a
  pin that is too small, and a memory stop.
- A start with a boot deadline in the past ends before any server action.

**Gate.** Each point gives its four memory values, and the live server is healthy after the window.

CAUTION: The window stops the bot's model server for about 30 minutes. Do not start it between 05:00 and 10:00 on a
weekday while the cron jobs are active.

## 5. Phase B: the arithmetic and your decisions (no downtime)

1. Put X of F1 into the formula of §2.3.
2. Read vLLM 0.33 for the boot check of §7.2. No downtime.
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

vLLM checks the free GPU memory at boot. On this machine that value leaves out the page cache: beside the live server
it showed about 2 GiB while 23 GiB was available (measured for a small GPU test on 2026-11-11, not for a second
server). If the boot of Qwen fails on this check, drop the page cache first. This needs `sudo`, so you run it:
`sudo sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches'`.

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
| A | The script and its dry run: about 2 hours. The window: about 30 minutes | About 30 min |
| B | About 1 hour, and your three decisions | None |
| C | One to three windows, by the table of §5 | About 1 to 2 h each |
| D | One evening for the tests, then 24 hours of normal use | None, if the margin holds |

Phase A decides whether the rest is worth the work. If X is 8 GiB or more, the plan ends there, or goes on with Qwen
at 128k.
