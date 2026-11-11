# Serve tune: a faster decode and less memory from the serving settings

On 2026-11-10 we tested 11 ideas for the speed and the memory of a live LLM server, without a change to the model's
weights. One idea is live now: a smaller KV cache pin saves 0.75 GiB and changes nothing else. One more idea passes
its gate: a wider draft model gives +4.2% decode speed for 2.39 GiB of memory, and it waits for a later decision. The
other ideas fail their gates or cannot run on this stack. This folder holds the plan, the tools, the run scripts, the
data and this write-up.

- The plan, with each test, gate and result: `PLAN_SERVE_TUNE.md` (§5 to §9).
- The data of every test point: `data/` (see "Data" below).

## At a glance

| # | Idea | Output change | Result |
|---|---|---|---|
| 1 | A shorter draft (3 or 4 tokens, not 5) | No | **Cannot run.** The draft is a block drafter with a block of 5 tokens; the image refuses a smaller depth |
| — | An FP8 copy of the head for the draft | No | **Fails.** −2.1%: the draft passes are faster, but fewer draft tokens pass the check |
| 2 | A smaller KV cache pin (8 → 7.25 GiB) | No | **Passes, live.** −0.75 GiB, 1.04x of a 256k request still fits, no speed loss |
| 3 | Larger prefill chunks (1,024 → 8,224 tokens) | No | **Fails.** +1.7% to +4.7% prefill speed (the gate is 10%), and 3.4 GiB more memory after a long prompt |
| 4 | An FP8 head for the target model | Yes | **Fails.** −0.49 GiB, but +0.91% NLL against the live model (the gate is +0.5%) |
| 5 | A different number of experts (K) in each layer | Yes | **Fails.** Not better than a uniform K at mean K 144; the decisions fail at mean K 152 |
| 6 | Suffix matches as draft tokens | No | **Fails the study.** +8.5% at best (the gate is 10%); reasoning text does not repeat |
| 7 | A wider draft (96 or 128 experts) | No | **128 experts pass, at the edge.** +4.21% bot decode for +2.39 GiB; not deployed yet |
| 8 | A KV tier on the NVMe disk | No | **Blocked.** It needs one week of metrics |
| 9–11 | EXL3 for attention, 2-bit cold experts, research items | — | Not in this round (`PLAN_SERVE_TUNE.md` §10) |

## The setup

- **Machine:** one NVIDIA DGX Spark (GB10). The CPU and the GPU share 121 GiB of memory.
- **Model:** DeepSeek V4 Flash, pruned for one bot: 160 of 216 routed experts per layer, 5 experts per token, 74.09
  GiB loaded (`../PRUNE.md`). The KV cache uses DeepSeek's 432-byte NVFP4 record.
- **Server:** a vLLM fork in Docker, port 8888, one sequence at a time, a context of 256,000 tokens.
- **Speculative decoding:** DSpark, a draft model cut from the model's own multi-token-prediction layer. It keeps 64
  of the 216 experts of that layer, chosen by the bot's routing. The draft proposes 5 tokens; the target checks 6
  rows in one pass, inside a full CUDA graph. About 2.2 draft tokens pass per step.
- **Workload:** a stock-research agent (Hermes) with long prompts, tool calls and long reasoning.
- **Memory before this round:** weights 74.09 GiB, KV cache 8.00 GiB (pinned), peak activations 5.08 GiB, other
  runtime memory 4.47 GiB; about 23 GiB stayed free.

## Method

**Test windows.** The machine holds one model at a time. So each test stops the live server, boots test points on
port 8890 one after the other, and restores the live server. Three windows ran back to back:

| Window | Time (2026-11-10) | Points | Run script |
|---|---|---|---|
| Step 1: lossless settings | 14:13–16:03 | T0, T3, T4 | `tune_night.sh` |
| Step 2: changes to the output | 16:04–19:46 | H1, N1, N2, N3, H0 | `tune_replay.sh` |
| Step 3: wider drafts | 19:46–22:40 | D0, W1–W4 | `tune_draft.sh` |

`../run_lib.sh` gives every window the same safety rules: one model in memory, a memory floor before each boot,
deadlines, a memory watch that stops a client below 1 GiB, and the restore. After each window, port 8888 answered.

**Speed.** The bot decode test sends recorded bot requests and measures the decode rate. Each point has 5 runs; its
rate is the median of the runs. A change counts only above the noise floor of 4.2% of the bot decode rate, and only
against a reference point in the same window, because the speed moves between boots. The code decode test (4 short
runs) is for information only.

**Quality.** The replay screen sends 60 recorded bot turns and reads the model's NLL of the recorded output, its
decisions and its tool names. The gates, against the unpruned model (K 216): the summed NLL rises at most 3%, the tool
names reach at least 98%, and the correct decisions fall by at most 3 (the noise). #4 also had to stay within +0.5%
NLL of the live model (the run noise is about 0.22%).

**Checks per point.** The graph check reads the boot log: the engine took the CUDA graph sizes, and the target and
the draft captured full graphs. The smoke requests send a JSON schema and two tool calls. The prefill probe
(`prefill_probe.py`) sends cold prompts of 8k, 32k and 128k tokens (a nonce stops cache hits) and a warm turn of 8k
new tokens on a cached 200k prefix.

## Results

### Step 1: settings that do not change the output

| Point | Settings | Weights | KV tokens | Bot decode | Draft tokens passed per step |
|---|---|---|---|---|---|
| T0 | The live settings | 74.09 GiB | 292,952 (1.14x of 256k) | 35.00 tok/s | 2.20 |
| T3 | + FP8 draft head | 74.58 GiB | 292,952 | 34.27 tok/s | 2.10 |
| T4 | KV pin 7.25 GiB, prefill chunk 8,224 | 74.09 GiB | 265,498 (1.04x) | 35.36 tok/s | 2.21 |

| Prefill probe | T0 (chunk 1,024) | T4 (chunk 8,224) |
|---|---|---|
| Cold 8k | 6.0 s, 1,338 tok/s | 5.8 s, 1,378 tok/s |
| Cold 32k | 23.9 s, 1,338 tok/s | 22.8 s, 1,400 tok/s |
| Cold 128k | 103.8 s, 1,233 tok/s | 102.1 s, 1,254 tok/s |
| Warm 8k on a 200k prefix | 12.2 s, 196,608 tokens from the cache | 12.5 s |
| Free memory: idle, lowest | 22.6 GiB, 22.6 GiB | 24.3 GiB, 20.8 GiB |

### Step 2: changes to the output

| Point | What | Summed NLL against K 216 | Decisions (K 216: 41) | Tool names (K 216: 77 of 83) |
|---|---|---|---|---|
| Live | Prune, K 160 | +1.89% | 42 | 78 |
| H1 | FP8 head of the target (73.60 GiB) | +2.82% (+0.91% against the live model) | 39 | 80 |
| N1 | K per layer, global rule, mean K 144 | +3.79% | 39 | 79 |
| N2 | K per layer, even rule, mean K 144 | +4.26% | 38 | 79 |
| N3 | K per layer, global rule, mean K 152 | +2.74% | 37 | 77 |

H1 against H0 (the live settings in the same window): bot decode 35.70 against 34.64 tok/s (+3.1%, inside the
noise), code decode 43.51 against 40.58 tok/s (+7.2%).

### Step 3: offline studies and a wider draft

The suffix study (`suffix_match.py`) read 66 recorded sessions: 675,623 output tokens and a history of 3.16 million.

| Match length m | Positions with a match of at least m | Correct of the next 5 | Gate estimate |
|---|---|---|---|
| 4 | 44.0% | 2.36 | +3.1% |
| 8 | 20.2% | 3.45 | +8.5% |
| 16 | 10.5% | 3.98 | +6.2% |

| Point | Draft, depth | Weights | Bot decode | Draft tokens passed per step | Against D0 |
|---|---|---|---|---|---|
| D0 | 64 experts (live), 5 | 74.09 GiB | 35.64 tok/s | 2.23 | — |
| W1 | 96 experts, 5 | 75.29 GiB | 35.82 tok/s | 2.28 | +0.5% |
| W2 | 96 experts, 6 | 75.29 GiB | 34.93 tok/s | 2.37 | −2.0% |
| W3 | 128 experts, 5 | 76.48 GiB | 37.14 tok/s | 2.43 | **+4.21%** |
| W4 | 128 experts, 6 | 76.48 GiB | 35.66 tok/s | 2.53 | +0.1% |

## What we learned

1. **Read the stack before the plan.** The best idea on paper (a depth of 3, about +10% by the estimate) died at a
   config check. DSpark drafts a block of 5 tokens, and the image refuses a smaller depth because the output would be
   wrong. A check of the source cost minutes; a test window costs hours.
2. **A faster draft pass can still lose.** The FP8 draft head made each draft pass cheaper, but the FP8 logits change
   the sampled draft tokens, and fewer of them pass the check (2.20 → 2.10 per step).
3. **The prefill chunk is not the limit here.** Chunks 8 times larger gave only 2% to 5%. They also kept about 3.4
   GiB more memory after a long prompt.
4. **The KV need is a floor, not a dial.** The boot check needs about 7 GiB for one 256k request. A pin of 7.25 GiB
   keeps a 4% margin and frees 0.75 GiB. That is the only change that went live.
5. **"Nearly the same logits" is not "the same model".** The FP8 target head matched BF16 well on test data (cosine
   above 0.999, the same top token on 256 of 256 rows). On real bot turns, the answer NLL did not move, but the
   reasoning NLL rose 1.4%.
6. **The router's weight share is not the cost of a cut.** Two rules that move experts between layers by their share
   both did worse than a uniform cut.
7. **Reasoning does not repeat; answers do.** Suffix matches predict tool calls and copied text well (an upper bound
   of +17% on the answer tokens alone), but the reasoning tokens are half of the output and give about nothing.
8. **More draft experts raise the acceptance at every position,** and 128 experts were worth it at depth 5. A depth
   of 6 never paid: the 6th token passes only about 10% of the time.
9. **The prefix cache hit only 1.0% of the prompt tokens** in a 6-hour snapshot. Find the cause before a disk tier.

## The deploy and the decisions

- **Live since 2026-11-10 23:04:** the KV pin of 7.25 GiB (one line of `compose-k160t5.yml` in the server repo; the
  backup and the way back are in `PLAN_SERVE_TUNE.md` §6.8). The boot took about 2 min 40 s.
- **Skipped for now:** the 128-expert draft. It gives +4.2% for 2.39 GiB, and the memory may go to a second model.
  §8.2 of the plan has the steps and a 75-minute recheck.

## Open items

- The 24-hour metrics of §5.1: the windows reset the counters. Take a snapshot after a day with no window.
- The KV tier (§9): one week of metrics first. The image has a file-system tier, but one code path accepts only one
  KV cache group, and this model has several.
- A suffix draft for the answer tokens only (§8.1) is the one variant of #6 worth a second study.

## Files

| File | Use |
|---|---|
| `PLAN_SERVE_TUNE.md` | The plan: the ideas, the stack check, each test and gate, and every result |
| `prefill_probe.py` | The prefill probe and the smoke requests (`--cold`, `--warm`, `--smoke`); never send `--smoke` to port 8888 |
| `compose-tune.yml` | Override of `../compose-mask.yml`: the draft depth and graph sizes, the FP8 draft head, the KV pin, the prefill chunk |
| `compose-head.yml` | Override on top of the two: the FP8 head of the target (`FP8_TARGET_HEAD=1`) |
| `image-patch/` | `fp8_target_head.py` and `head-fp8.diff`; `build_head.sh` builds the patched `deepseek_v4_model.py` (not in git) |
| `alloc_k.py` | The K of each layer for a target mean K (rule `global` or `even`), for `../mask_checkpoint.py --keep-per-layer` |
| `suffix_match.py` | The suffix-match study; run it in the server image, which has `tokenizers` |
| `tune_night.sh` | The Step 1 window (T0, T3, T4) |
| `tune_replay.sh` | The Step 2 window (H1, N1–N3, H0) |
| `tune_draft.sh` | The Step 3 window (D0, W1–W4) |
| `test_*.py`, `test_fp8_head.sh` | The tests |

**Run.** Use the stock-research venv (`~/.hermes/stock-research/.venv/bin/python`). Run docker compose in the repo
root, with `../compose-mask.yml` as the first file. CAUTION: each run script stops the live server; start one only with
the owner's go, never from 05:00 to 10:00 on weekdays, and never beside another test:

    tmux new -d -s tune "START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/serve-tune/tune_night.sh"

**Tests.** `python -m unittest` in the repo root runs the tests of this folder (through `../test_serve_tune.py`).
`./test_fp8_head.sh` runs the GPU test of the FP8 head in the server image.

## Data

| File | Content |
|---|---|
| `data/points.jsonl` | One line per test point: the settings, the weights, the KV tokens, the boot time, the bot and code decode rates per run, the draft tokens passed per position, and the replay result |
| `data/probe.json` | The prefill probe of T0 and T4, and the smoke result of T4 |
| `data/suffix-match.json` | The suffix study: the share and the mean per match length and depth, and the gate estimates |
| `data/metrics-8888-20261110.txt` | The prompt-size histogram, the prefix-cache counters and the preemptions of the live server (§5.1) |

The raw result files (the replay rows, the boot logs, the speed runs) stay in `../results/`, which is not in git.
`points.jsonl` is enough for the charts of a post: the speed per point, the acceptance per position, and the NLL
against the memory.
