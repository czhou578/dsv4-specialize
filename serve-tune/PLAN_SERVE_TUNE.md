# Plan: a faster decode and less memory from the serving settings

**Moved** on 2026-11-10 from `plans/` to `serve-tune/`, with its tools and results; `README.md` in this folder is the
write-up. The other plans that this plan names are in `../plans/`.

**Status:** 2026-11-10: done. Two ideas pass a gate. The KV pin of 7.25 GiB (#2) is live since 2026-11-10 23:04 (§6.8). The 128-expert draft (#7, +4.2% for 2.39 GiB) is skipped for now, your decision (§8.2). The other ideas fail or cannot run. ⛔ Blocked: the 24-hour metrics (§5.1) and the KV tier study (§9) · **Author:** colin-spark · **Date:** 2026-11-09
**Source:** a ranked list of 11 ideas for the speed and the memory of the live server. This plan checks each idea
against this stack and puts the ideas in an order with tests and gates.
**Related:** `PLAN_FEWER_EXPERTS.md` (done 2026-11-10, no deploy; it changes the expert side), `PLAN_PHYSICAL_PRUNE.md` §7 (the
bot-rank draft), `PLAN_BETTER_RANK.md` (the capture data), `PRUNE.md` (the live model)
**Scope:** test boots on port 8890; new tools and run scripts in this repo; edits of `compose-k160t5.yml` in the server
repo, each on your go.

| Step | Status |
|---|---|
| §5 Step 0: prepare (no downtime) | ✅ 2026-11-10: the tools pass their tests (§5.2); one metrics snapshot (§5.1). ⛔ The 24-hour snapshot is blocked: the windows restart the server and reset the counters |
| §6 Step 1: the lossless window (about 3.5 h downtime) | ✅ 2026-11-10 14:13–16:03 (§6.7): #2 (KV pin 7.25 GiB) passes, live since 23:04 (§6.8); #1 is closed (the draft's block size is 5, §3); the FP8 draft head and #3 (prefill chunk) fail |
| §7 Step 2: the replay window (about 3.5 h downtime) | ✅ 2026-11-10 16:04–19:46 (§7.4): #4 (FP8 head of the target) saves 0.49 GiB but fails its gate (+0.91% NLL against the live model); #5 (K per layer) fails at mean K 144 and 152. No deploy |
| §8 Step 3: offline studies, then builds | ✅ 2026-11-10: #6 (suffix drafts) fails the study gate (+8.5%, the gate is 10%), so no build (§8.1); #7 (wider draft): 128 experts at depth 5 pass at the edge (+4.21%) for +2.39 GiB; skipped for now, your decision (§8.2) |
| §9 Step 4: the KV tier study | ⛔ Blocked: it needs one week of metrics (§9.1). The connector checks are done: a file-system tier exists; the support for this model needs a test boot |

---

## 1. Short answer

The live server can get faster and smaller in four steps. Each step has a test window, a gate, and a deploy that you
decide.

1. **Step 1: lossless.** A shorter draft (3 or 4 tokens, not 5), an FP8 copy of the head for the draft, a smaller KV
   pin, and larger prefill chunks. No item changes the model's output. One evening.
2. **Step 2: the output changes.** An FP8 head for the target model, and a mask test with a different K in each layer.
   The replay screen decides.
3. **Step 3: offline studies first.** A count of suffix matches in the bot's sessions, and a wider draft. A build
   follows only when its study passes.
4. **Step 4: later.** A KV tier on the NVMe disk, after a study of the cache misses.

Three ideas get no work now (§10): EXL3 for attention, K2 for cold experts, and the research items.

Expected result of Step 1, by estimate: bot decode +5% to +10% from the depth, and a few percent more from the FP8
draft head. The memory falls by about 0.75 GiB. Long cold prompts prefill about 20% faster. The tests measure each
number.

## 2. Terms

| Term | Meaning |
|---|---|
| Draft depth | The number of tokens that the DSpark draft proposes per step (`DSPARK_TOKENS`, now 5) |
| Verify step | One target pass over the last token plus the draft tokens: depth + 1 rows |
| KV pin | The fixed KV cache size, `--kv-cache-memory-bytes` (now 8 GiB) |
| Prefill chunk | The largest number of prompt tokens per step, `--long-prefill-token-threshold` (now 1024) |
| Noise floor | 4.2% of the bot decode rate (about 1.45 tok/s). A smaller change is not a result |
| Replay gate | The NLL, tool-name and decision gates of the replay screen (§7.1) |
| Window | One run of a run script: the live server stops, the test boots run, the script restores the live server |

## 3. What the stack check changed

The source list used the recipe's README. Five of its facts differ on this stack:

| Claim in the list | What this stack shows | Effect on the plan |
|---|---|---|
| The live KV record is the 584-byte padded FP8 record. The 432-byte record corrupts text | The live server runs the 432-byte NVFP4 record (boot log: "Using DeepSeek's 432-byte NVFP4 MLA KV cache format"). The kernel fixes in the server repo make it work. The image default is 584 bytes | The same result: no smaller record exists, so KV quantization stays closed |
| A smaller KV pin saves up to about 4 GiB | Most of the KV need is fixed. The boot check needs 6.54 GiB at 128k (failed boot of 2026-11-07) and about 7.0 GiB at 256k (1.14x at 8 GiB) | #2 saves about 0.75 GiB at 256k, or about 1.1 GiB at 160k. It shares a boot with #3 |
| Larger prefill chunks need one flag | `MAX_NUM_BATCHED_TOKENS` must stay at 8224: the server README documents a locked-workspace crash below it. The flag to change is `--long-prefill-token-threshold`. With `MAX_NUM_SEQS=1`, no other request can starve | Raise the threshold to 8224. Test a deep warm turn, because chunks above 1024 never ran on the live server |
| The memory is about 85 GiB: 74 weights, 8 KV, 3 draft | The 74.09 GiB of the boot log includes the draft (one load). The runtime adds 9.55 GiB | The sum is about 91.6 GiB (§11) |
| The bot may send strict JSON schemas | The bot plugin and the Hermes chat path send no `response_format`. Tool calls use the default `tool_choice`, auto, which applies no grammar mask | The grammar risk is small. The plan keeps a fixed depth, and each new depth gets a JSON-schema smoke test |

**A sixth fact (found on 2026-11-10, before the window).** A shorter draft (#1) cannot run with the live draft. The
image's `vllm/config/speculative.py` refuses `num_speculative_tokens` below the draft's `dspark_block_size`, which is
5 in `dspark-draft-k64-bot/config.json`: "DSpark requires num_speculative_tokens >= dspark_block_size ... Smaller
values produce incorrect output." DSpark drafts a block of 5 tokens, so depths 3 and 4 do not boot. The window of §6
therefore runs T0, T3 and T4 only (about 2 h 15 min), and `tune_night.sh` keeps T1 and T2 for a draft with a smaller
block size. A depth above 5 boots, but each extra position adds less than the 5th (0.14 accepted tokens), so it
cannot be faster by the estimate of §6.1.

One addition: the fork's `dspark.py` can compute the draft logits with a rowwise-FP8 copy of the head
(`DSPARK_FP8_DRAFT_HEAD=1`). The verify pass stays BF16, so the output does not change. The source list does not
have this item. Step 1 tests it.

## 4. Order, gates and downtime

| Step | Ideas | Downtime | Output change | Gate | Deploy |
|---|---|---|---|---|---|
| 0 Prepare | — | None | — | Tools pass their tests; prompt data read | — |
| 1 Lossless window | #1, #2, #3, FP8 draft head | About 3.5 h | No | Speed above the noise; graph, smoke and warm-turn checks | 4 to 8 min, your decision |
| 2 Replay window | #4, #5 (mask only) | About 3.5 h | Yes | Replay gate | #4: 4 min; #5: a new plan for the prune |
| 3 Offline studies | #6, #7 | None, then about 3 h per test window | No | Match rate; acceptance gain | Your decision |
| 4 KV tier | #8 | Not planned yet | No | A cache-miss study | A separate plan |

CAUTION: Each window stops the bot's model server. The bot gives no answers until the run script restores it.

Rules for every window:

- Do not start a window near the `daily-report` or `daily-followups` cron jobs (05:00 to 10:00 on weekdays).
- Do not overlap a window with a night of `PLAN_FEWER_EXPERTS.md`. One such night runs on 2026-11-09.
- Use `run_lib.sh` for every run script. It keeps one model in memory, applies the deadlines and the memory watch, and
  restores the live server.
- Compare speeds only inside one window. Each window boots its own reference point, because the speed moves between
  days.

## 5. Step 0: prepare (no downtime)

### 5.1 Read the bot's prompt sizes

The counters of `/metrics` start at zero when the server starts, and a test client on port 8888 (for example the judge
of a step F run) adds its own requests. The counters only go up, so take two snapshots: one when no test uses the
live server, and one 24 hours later. The difference is the bot's traffic alone. Read these values on port 8888:

- the histogram of prompt tokens per request (`vllm:request_prompt_tokens`);
- the prefix-cache queries and hits (`vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total`);
- the preemptions (`vllm:num_preemptions_total`).

Check the names first: `curl -s localhost:8888/metrics | grep -E 'prompt|prefix|preempt'`.

The data feeds two decisions:

- **#2.** Hermes compresses a session at 50% of its 256k context, so at about 128k tokens. If no prompt goes above
  about 130k tokens, the 160k option of §6.4 is safe.
- **#8.** The missed tokens on long prompts give the value of a disk tier (§9).

**Result (2026-11-10 13:47).** One snapshot on port 8888 (`results/metrics-8888-20261110.txt`). The live server
started at 07:23, so the counters hold 6 h 24 min of traffic:

| Value | Count |
|---|---|
| Requests | 181 |
| Prompt tokens | 2,067,008 |
| Prompts above 20,000 tokens | 6 (all at or under 50,000) |
| Prompts above 50,000 tokens | 0 |
| Prefix-cache hits | 20,480 tokens (1.0% of the queried tokens) |
| Preemptions | 0 |

- The snapshot is not the bot's traffic alone. The judge of the step F run (`PLAN_FEWER_EXPERTS.md` §9.1) used port
  8888 from 07:25 to 12:27.
- ⛔ **The 24-hour snapshot is blocked.** Tonight's windows restart the live server, and a restart sets the counters to
  zero. Take it after a day with no window.
- **#2.** No prompt went above 50,000 tokens. Six hours do not prove the 160k option of §6.4 safe, because a session
  can grow to 128k tokens before Hermes compresses it. T4 keeps the 256k length; the 160k option waits for the
  24-hour data.
- **#8.** The prefix cache hit only 1.0% of the prompt tokens. A disk tier serves only prefixes that the cache can
  match, so find the cause of the low rate before the study of §9.

### 5.2 Write the tools

| File | Use |
|---|---|
| `compose-tune.yml` | Override of `compose-mask.yml`. It passes the variables of §6.1 and gives each point its own served name. The KV is pinned and the util is 0.90, as in `compose-drop.yml` |
| `prefill_probe.py` | Cold prompts of N tokens (a nonce filler, so the prefix cache cannot hit), the deep warm turn, and the smoke requests. It prints JSON: time to first token, prefill tok/s, pass or fail |
| `test_prefill_probe.py` | Tests of the probe with a fake server, as in `test_capture.py` |
| `tune_night.sh` | The run script of §6: the points T0 to T4, then the restore. `BOOT_MIN_GIB=110`, as in `fewer_night.sh` |

The four files must stay under 400 changed lines, so that one pull request holds them. If they grow past that, split
them: the probe and its test first, then the compose file and the run script.

`run_lib.sh` gets one change: its memory watch also stops `prefill_probe.py`. `run_lib.sh` runs `docker compose` in
`~/Projects/dsv4-specialize`, so the window can start only after the files are in that folder.

**Gate.** All unit tests pass (`python -m unittest`). `bash -n tune_night.sh` passes.
`docker compose -f compose-mask.yml -f compose-tune.yml config` resolves.

**Result (2026-11-10).** The gate passes. Two more checks passed:

- The graph check of `tune_night.sh` passes on the boot log of the live settings (`boot-prune5b-k160.log`), and it
  fails when the sizes do not match.
- A dry run with stub boots and speeds chose the right depth and head in five cases: a normal run, a failed T4 boot
  (T4b follows), a T1 graph failure, a head gain inside the noise, and depths slower than T0.

The probe has only fake-server tests. Before the window, run `prefill_probe.py --base-url http://127.0.0.1:8888
--cold 2000` once, when no test uses the live server. It checks the request format with a 2k-token prompt. Do not
send `--smoke` to the live server: a grammar assertion there stops the bot's server.

**Probe check (2026-11-10 13:50).** The live server was idle. `prefill_probe.py --cold 2000` on port 8888: PASS,
2,001 tokens, time to first token 1.47 s, 1,361 tok/s, lowest available memory 22,979 MiB, the code recalled.
`/tokenize` gives 47 template tokens and 1.0 token per filler word.

**Pull requests.** Step 0 has two: the plan and the probe with its test first (0a), then the compose file, the run
script and the `run_lib.sh` change (0b). `test_moe_drop.py` now skips under `python -m unittest` when pytest is
missing, so that `python -m unittest` passes in the venv.

## 6. Step 1: the lossless window (#1, #2, #3, FP8 draft head)

### 6.1 The variables

| Variable | Live value | Test values | Meaning |
|---|---|---|---|
| `DSPARK_TOKENS` | 5 | 3, 4 | The draft depth |
| `CUDAGRAPH_CAPTURE_SIZES` | `6,12,24` | `4,8,16` (depth 3), `5,10,20` (depth 4) | Multiples of depth + 1, so the verify step gets a full graph |
| `MAX_CUDAGRAPH_CAPTURE_SIZE` | 24 | 16, 20 | The largest size of the list |
| `DSPARK_FP8_DRAFT_HEAD` | 0 | 1 | An FP8 copy of the head, for the draft passes only |
| `--kv-cache-memory-bytes` | 8589934592 (8 GiB) | 7784628224 (7.25 GiB) | The KV pin |
| `--long-prefill-token-threshold` | 1024 | 8224 | The largest prefill chunk |

`MAX_NUM_BATCHED_TOKENS` stays at 8224 in every point.

The reasons for the values:

- **Depth.** The live acceptance per position is 0.80, 0.58, 0.39, 0.24 and 0.14 (bot test, 2026-11-07). The 4th and
  5th tokens add 0.38 accepted tokens per step. They cost two draft passes and two verify rows. Each draft pass reads
  the full 0.99 GiB BF16 head. The estimate puts the best depth at 3, about 10% above depth 5.
- **Graph sizes.** The launcher's rule is `MAX_CUDAGRAPH_CAPTURE_SIZE` = sequences × (depth + 1). The live boot
  captures one full graph, for the six-row verify step. If the list does not hold the new row count, the verify step
  falls back to a slower path. The test then shows a false loss.
- **FP8 draft head.** The draft computes logits over the full vocabulary on each pass. The FP8 copy halves the bytes
  of that read. The copy costs about 0.5 GiB. My estimate is +3% to +8% bot decode.
- **KV pin.** 7.25 GiB keeps a margin of about 3% over the 7.0 GiB that the 256k check needs. The pool falls from
  292,952 to about 265,000 tokens.
- **Prefill chunk.** With one sequence, the cap of 1024 protects no other request. Each chunk adds the overhead of
  one step.

### 6.2 The points

All points boot the live folder (`MASK_NAME=prune5b MASK_K=160`, a link to `data/tp1-k160t5`) with the bot draft
(`DRAFT_DIR=.../data/dspark-draft-k64-bot`) on port 8890.

| Point | Depth | FP8 draft head | KV pin | Prefill chunk | Tests | Time |
|---|---|---|---|---|---|---|
| T0 | 5 | Off | 8 GiB | 1024 | The reference: graph check, prefill probe, bot decode × 5, code test | 50 min |
| T1 | 3 | Off | 8 GiB | 1024 | Graph check, smoke, bot decode × 5, code test | 40 min |
| T2 | 4 | Off | 8 GiB | 1024 | The same as T1 | 40 min |
| T3 | The fastest valid depth of T0, T1, T2 | On | 8 GiB | 1024 | Graph check, head check, bot decode × 5, code test | 40 min |
| T4 | As T3; the head only if T3 passed | — | 7.25 GiB | 8224 | Graph check, smoke, prefill probe with the deep warm turn and the memory, bot decode × 3, code test | 40 min |

Then the restore follows (about 5 minutes). The window takes about 3 h 35 min. The code test is the one of `speed()`
in `run_lib.sh` (four short runs). It is for information only.

T0 also runs the graph check. The check must pass on the live settings. If it fails, the check itself is wrong, and the
script logs a warning. A depth counts for T3 only if its graph check passes and its bot decode beats T0.

If T4 fails the KV check, boot T4b: 8 GiB and the 8224 chunk. Record the KV need from the error message.

### 6.3 The checks

- **Graph check (T1, T2).** The boot log lists the new sizes in `cudagraph_capture_sizes`. It shows
  `Capturing CUDA graphs (FULL)` with at least one graph, and `Capturing dspark CUDA graphs (FULL)`. If a line is
  missing, the point is not valid: fix the sizes, and do not use the speed.
- **Head check (T3).** The boot log has the line "draft-time only; verify pass untouched" from
  `maybe_init_fp8_draft_head`. A warning about an unsupported device means that the kernel does not run on GB10. Then
  drop T3, and record the result for #4.
- **Smoke (T1, T2, T4).** `prefill_probe.py --smoke` sends three requests: one with a `response_format` JSON schema,
  one tool call with `tool_choice` auto, and one with `required`. Each must return valid JSON, and the server must stay
  healthy. An assertion in the log fails the point.
- **Prefill probe (T0, T4).** Cold prompts of 8k, 32k and 128k tokens give the time to first token and the prefill
  rate. Then the deep warm turn runs: a cold prompt of 200k tokens, then the same prompt plus 8k new tokens (a
  prefix-cache hit). The warm turn must end with no `Workspace is locked` assertion.
- **Memory under load (T0, T4).** The memory watch logs the available memory every 30 seconds. Record the lowest
  value during the 128k cold prompt.

### 6.4 Gates

| Idea | Deploy when | Else |
|---|---|---|
| #1 depth | The best depth beats T0 by more than the noise floor, and its graph check and smoke pass | A gain inside the noise: your decision, as for the bot draft (`PLAN_PHYSICAL_PRUNE.md` §7.1). A loss: keep depth 5 |
| FP8 draft head | T3 beats its base by more than the noise floor | Keep it off, because it costs 0.5 GiB |
| #2 KV pin | T4 boots, and the boot log gives at least 1.02x for 256,000 tokens per request | Keep 8 GiB |
| #3 prefill chunk | The cold 32k and 128k prompts get at least 10% faster, the warm turn passes, and the lowest available memory stays above 15 GiB | Keep 1024 |

**The 160k option of #2.** Set `MAX_MODEL_LEN` to 160000 and pin about 6.9 GiB. In Hermes, set `context_length` to
160000 and the compression `threshold` to 0.8, so that the compression still starts at 128k. This saves about
0.35 GiB more than the 256k pin. Use it only if the data of §5.1 shows no prompt above about 130k tokens.

### 6.5 Deploy (your decision)

1. Back up `compose-k160t5.yml` as `compose-k160t5.yml.bak.<date>_tune`.
2. Change only the values that passed their gate. Update the comment lines of the file.
3. Run `docker compose -f compose-k160t5.yml up -d` in the server folder. The downtime is 4 to 8 minutes, because
   the first boot compiles the new graph sizes into `cache-k160t5`.
4. Check the boot lines, one reply, and the graph check of §6.3.
5. Watch the next `daily-report` run.

To go back, copy the backup over the file and run `docker compose -f compose-k160t5.yml up -d`.

### 6.6 When to test the depth again

The best depth depends on the acceptance. Run T0, T1 and T2 again (about 2 hours) after a change that moves the
acceptance:

- a deploy from `PLAN_FEWER_EXPERTS.md` (the draft then runs at top-k 6);
- a wider draft (#7);
- suffix drafts (#6).

Since 2026-11-10 this applies only to depths of 5 and more, because the draft cannot go below its block size (§3).

### 6.7 Result (2026-11-10, 14:13 to 16:03)

You moved the start from 20:00 to 14:13. `tune_night.sh` ran T0, T3 and T4 (T1 and T2 cannot boot, §3). The live
server was down from 14:13 to 16:03 (1 h 50 min). The restore was healthy at 16:03 (reply "OK.", gateway active).
Logs: `logs/tune-night.log`, `results/boot-tune-t*.log`, `results/probe-tune-t*.json`, `results/speed-*-tune-t*`.

| Point | Settings | Boot | Bot decode (runs) | Accepted per step | Code decode |
|---|---|---|---|---|---|
| T0 | The live settings | 180 s, 74.09 GiB, 292,952 KV tokens, 1.14x | **35.00** tok/s (33.39–35.30) | 2.20 | 40.90 tok/s |
| T3 | + FP8 draft head | 180 s, **74.58 GiB**, 292,952, 1.14x | 34.27 tok/s (33.39–38.75) | 2.10 | 41.75 tok/s |
| T4 | KV pin 7.25 GiB, prefill chunk 8,224 | 165 s, 74.09 GiB, **265,498**, **1.04x** | 35.36 tok/s (3 runs, 35.02–36.65) | 2.21 | 40.84 tok/s |

The graph check passed on all three points. The smoke requests of T4 passed (the JSON schema and both tool choices),
and no point logged an assertion.

| Prefill probe | T0 (chunk 1,024) | T4 (chunk 8,224) | Change |
|---|---|---|---|
| Cold 8k | 6.0 s, 1,338 tok/s | 5.8 s, 1,378 tok/s | +3.0% |
| Cold 32k | 23.9 s, 1,338 tok/s | 22.8 s, 1,400 tok/s | +4.7% |
| Cold 128k | 103.8 s, 1,233 tok/s | 102.1 s, 1,254 tok/s | +1.7% |
| Warm turn: 8k new tokens on 200k | 196,608 tokens from the cache, 12.2 s, pass | 196,608, 12.5 s, pass | — |
| Available memory: before the probe, lowest | 22.6 GiB, 22.6 GiB | 24.3 GiB, **20.8 GiB** | −3.4 GiB under load |

The gates of §6.4:

- **#1 depth: closed.** Depths 3 and 4 do not boot with the live draft (§3). No deploy.
- **FP8 draft head: fails.** T3 is 2.1% slower than T0 (34.27 against 35.00 tok/s). The FP8 logits change the draft
  choices, and the accepted tokens per step fall from 2.20 to 2.10 at every position. The code decode gains 2.1%,
  inside the noise. The head costs 0.49 GiB (74.58 against 74.09 GiB). Keep it off.
- **#2 KV pin: passes.** T4 boots with 7.25 GiB, and the boot log gives 1.04x for 256,000 tokens per request (the
  gate is 1.02x). The pool falls from 292,952 to 265,498 tokens. The bot decode does not fall (35.36 tok/s).
- **#3 prefill chunk: fails.** The cold 32k and 128k prompts are only 4.7% and 1.7% faster (the gate is 10%). The
  warm turn passes, and the lowest available memory stays above 15 GiB. But after a long prefill the larger chunk
  holds about 3.4 GiB more memory (24.3 GiB available at idle, 20.8 GiB during and after the 128k prompt; T0 did not
  fall). Keep 1024.

**Deploy recommendation (your decision): #2 only.** Change `--kv-cache-memory-bytes 8589934592` to `7784628224` in
`compose-k160t5.yml`, and keep `--long-prefill-token-threshold 1024`, the depth 5 and no FP8 draft head. Follow the
steps of §6.5. The graph sizes do not change, so the boot needs no new compile (about 3 min). The gain is 0.75 GiB of
memory and no output change. The risk: the margin over the 256k need falls from 14% to 4%, and the pool holds 9%
fewer warm tokens. §5.1 saw no prompt above 50,000 tokens. Do not take the 160k option of §6.4 before the 24-hour
data of §5.1 exists.

### 6.8 Deploy of #2 (2026-11-10, 23:01 to 23:04, on your go)

1. The backup is `compose-k160t5.yml.bak.20261110_tune` in the server folder.
2. One line of `compose-k160t5.yml` changed: `--kv-cache-memory-bytes 8589934592` became `7784628224`, and its
   comment names this plan. Nothing else changed: the prefill chunk stays 1024, the depth 5, no FP8 draft head.
3. `docker compose -f compose-k160t5.yml up -d` recreated the container at 23:01:09. It was healthy at 23:03:51
   (downtime about 2 min 40 s).
4. The checks pass. The boot log says "reserved 7.25 GiB memory for KV Cache", 265,498 KV tokens, 1.04x for 256,000
   tokens per request, the graph sizes 6, 12, 24, and both full graph captures. The weights take 74.09 GiB. The
   reply is "OK.", both served names answer, and the Hermes gateway is active. 23.5 GB stays available at idle.
5. Watch the next `daily-report` run (2026-11-11 from 05:00).

To go back, copy `compose-k160t5.yml.bak.20261110_tune` over `compose-k160t5.yml` and run
`docker compose -f compose-k160t5.yml up -d` in the server folder.

## 7. Step 2: the replay window (#4, #5)

These ideas change the output. The replay screen gates them. A deploy of #5 also needs the full runs.

### 7.1 The replay gate

Use the gates of the earlier plans, against `baseline-k216.jsonl`:

- the summed NLL rises by at most 3%;
- the tool names reach at least 98% of the baseline;
- the decisions fall below the baseline by no more than the noise (3).

For #4, apply one more gate against the live replay (`prune5-k160.jsonl`): the summed NLL rises by at most 0.5%. The
run noise is about 0.22%.

### 7.2 #4: an FP8 head for the target

`head.weight` is BF16, 129,280 × 4,096 (0.99 GiB). Each verify step reads all of it. The draft shares it.

Work (no downtime, about one day):

1. Copy the image's `vllm/models/deepseek_v4/nvidia/model.py` to `image-patch/`. Keep the change as
   `image-patch/head-fp8.diff`, as for the router patch.
2. After the weights load, quantize the head with `quantize_draft_head` from the module of the FP8 draft head.
3. Compute the target logits with `fp8_draft_head_logits`, and free the BF16 weight.
4. Let the draft use the same FP8 tensor. The net saving is then about 0.49 GiB, and the draft gets the FP8 speed with
   no second copy.
5. Write the unit tests:
   - the FP8 logits against BF16 on random hidden states (cosine and top-1 agreement);
   - the BF16 weight is freed;
   - the draft and the target use one tensor.
6. Mount the file in a compose override, `compose-head.yml`.

T3 of Step 1 shows whether the FP8 head kernel runs on GB10. If it does not, step 3 needs another kernel, for example
`torch._scaled_mm`.

**Test.** One replay point (40 min) and one speed point (bot decode × 5, 35 min). T0's settings are the reference in
the same window.

**Gate.** The replay gate of §7.1. The speed gain (an estimated +2% to +3%) is inside the noise. The reason to deploy
is the memory.

**The patch (2026-11-10).** It differs from steps 1 to 6 in one point: the draft needs no change.

- `image-patch/fp8_target_head.py`: after the load, `install` quantizes the head with `quantize_draft_head`, puts an
  FP8 method in place of the head's `quant_method`, and frees the BF16 weight. `LogitsProcessor` calls
  `lm_head.quant_method.apply`, so the target logits use `fp8_draft_head_logits`. The DSpark draft has no head of its
  own (`has_own_lm_head = False`), and `load_dspark_model` gives it the head module of the target. So the draft reads
  the same FP8 tensor, and `dspark.py` (a file of the server repo) does not change.
- `image-patch/head-fp8.diff`: 5 lines at the end of `load_weights` in the image's `model.py`. `FP8_TARGET_HEAD=1`
  turns the patch on. `image-patch/build_head.sh` builds the patched file (not in git).
- `compose-head.yml`: an override on top of `compose-mask.yml` and `compose-tune.yml`. It mounts the two files and
  keeps `DSPARK_FP8_DRAFT_HEAD` at 0: the FP8 draft head reads the BF16 weight, and `install` stops with an error if
  that flag is on.
- `test_fp8_head.py` (`./test_fp8_head.sh`, in the server image on the GB10): 4 tests pass with a real
  `ParallelLMHead` and `LogitsProcessor`. The FP8 logits match BF16 (cosine above 0.999; the same top-1 on 256 of 256
  rows). The BF16 weight leaves the memory, and the draft reads the same FP8 tensor. A CUDA graph replay equals an
  eager call. So `torch._scaled_mm` runs on the GB10, and step 3 needs no other kernel.

### 7.3 #5: a different K in each layer (mask test only)

Uniform K 144 fails the NLL gate with every rank (+3.29% to +3.90%, `PLAN_BETTER_RANK.md` §6). One K step on all
43 layers holds about 0.38 GiB (9.0 MiB per expert). A mean K of 152 saves about 3 GiB, and 144 saves about 6 GiB.

Work (no downtime):

1. Add `--keep-per-layer FILE` to `mask_checkpoint.py`: a JSON list of 43 K values. Keep the hash rule of layers 0–2.
2. Write `alloc_k.py`. It reads the routing weight per expert and layer from the capture of `PLAN_BETTER_RANK.md` §4,
   and computes the K of each layer for a target mean. It has two rules:
   - **global:** drop the experts with the smallest routing share, in any layer;
   - **even:** drop the same routing share in each layer.
3. Keep each layer between 128 and 216. Keep layers 0–2 at 160, because the hash remap of a smaller K has no test.
4. Build each folder with `mask_checkpoint.py`, then with `topk_checkpoint.py --top-k 5`. Each build takes about
   2 minutes and 24 GB.
5. Write unit tests on the small synthetic checkpoint, as in `test_mask_checkpoint.py`.

| Point | Mean K | Rule | Saving after a prune |
|---|---|---|---|
| N1 | 144 | Global | About 6 GiB |
| N2 | 144 | Even | About 6 GiB |
| N3 | 152 | The better rule of N1 and N2 | About 3 GiB |

Each point is one replay (40 min). K does not change the speed, so the points need no speed test.

**Gate.** The replay gate of §7.1, against the baseline.

If a point passes:

- The mask saves no memory. The saving needs a physical prune with a different expert count in each layer. The
  runtime reads one count for all layers, so the prune needs a loader patch with its own plan.
- A deploy also needs the full runs, the judge and your blind review, as for the live pick (`PLAN_FINAL_REVIEW.md`).

Run #5 after the decision of `PLAN_FEWER_EXPERTS.md`, because both plans change the expert side.

**The code and the folders (2026-11-10, no downtime).** `PLAN_FEWER_EXPERTS.md` ended on 2026-11-10 with no deploy,
so #5 runs now.

- `mask_checkpoint.py --keep-per-layer FILE` takes a JSON list of 43 K values and needs `--out`. The rule for the
  mandatory tool experts moved into `drop_order`, which `alloc_k.py` also uses. So both tools drop the same experts.
  `MASK.json` records the list and its mean.
- `alloc_k.py --target K --rule global|even` reads the weight share of each expert from `expert_rank_weight.json`
  (the capture of `PLAN_BETTER_RANK.md` §4). The same file gives the drop order inside each layer. Layers 0–2 keep
  160, layers 3–42 stay between 128 and 216, and the 43 values sum to exactly 43 × the target.
- Tests: `test_alloc_k.py` (the sum, the bounds, the mandatory rule, the two rules, the agreement with
  `mask_checkpoint.py`) and one more test in `test_mask_checkpoint.py`.
- Each folder took about 50 s to build and holds 12 GB. `topk_checkpoint.py --check` passes on each, and no link is
  broken. The intermediate mask folders are deleted.

| Point | Folder | Mean K | Rule | K in layers 3–42 | Weight share dropped (sum over layers 3–42) |
|---|---|---|---|---|---|
| N1 | `tp1-klg5-k144` | 144 | global | 128–163 | 2.773 |
| N2 | `tp1-kle5-k144` | 144 | even | 131–155 | 2.790 |
| N3 | `tp1-klg5-k152` | 152 | global | 134–169 | 2.093 |
| N3 | `tp1-kle5-k152` | 152 | even | 140–162 | 2.108 |

The global rule drops 0.6% less weight share than the even rule. `tune_replay.sh` chooses the folder of N3 from the
replay of N1 and N2.

The window holds #4 (75 min), #5 (2 h) and the restore: about 3.5 hours.

### 7.4 Result (2026-11-10, 16:04 to 19:46): every point fails a gate

`tune_replay.sh` ran right after the Step 1 window, on your go. It ran the gate points first (H1, N1, N2, N3), then
the speed reference H0. Every point booted with `compose-tune.yml` at the live settings and the bot draft. Logs:
`logs/tune-replay.log`, `results/tune-{h1,n1,n2,n3}*`.

| Point | What | Boot | Summed NLL against the baseline | Decisions (baseline 41) | Tool names (baseline 77 of 83) | Gates |
|---|---|---|---|---|---|---|
| Live model (2026-11-06) | Prune, K 160 | — | +1.89% | 42 | 78 | pass |
| H1 | #4: the FP8 head of the target | 181 s, **73.60 GiB** | +2.82% (**+0.91%** against the live replay) | 39 | 80 | **fails** the gate of #4 |
| N1 | #5: global rule, mean K 144 | 241 s, 95.39 GiB | **+3.79%** | 39 | 79 | **fails** (NLL) |
| N2 | #5: even rule, mean K 144 | 225 s, 95.39 GiB | **+4.26%** | 38 | 79 | **fails** (NLL) |
| N3 | #5: global rule (the better one), mean K 152 | 285 s, 95.39 GiB | +2.74% | **37** | 77 | **fails** (decisions: −4, the noise is 3) |

**#4, the FP8 head of the target: fails.**

- The patch works: the boot log says "FP8 target head: 129280 rows; 0.99 GiB BF16 freed, 0.49 GiB FP8 kept". The
  weights take 73.60 GiB against 74.09 GiB, so the patch saves the 0.49 GiB of the estimate.
- The gates against the baseline pass, but the summed NLL rises 0.91% against the live replay (20,858 against
  20,669). The gate of #4 allows +0.5%, and the run noise is about 0.22%. The answer NLL does not move (0.2689 to
  0.2691); the loss is in the reasoning tokens (0.7443 to 0.7547, +1.4%).
- Speed, against H0 (the live settings, the same window, 180 s boot, 74.09 GiB):

  | Point | Bot decode (runs) | Accepted per step | Code decode (runs) | Code accepted per step |
  |---|---|---|---|---|
  | H0 | 34.64 tok/s (33.77–35.58) | 2.14 | 40.58 tok/s (40.56–40.70) | 2.63 |
  | H1 | 35.70 tok/s (35.16–36.49) | 2.14 | 43.51 tok/s (42.36–43.59) | 2.66 |

  The bot decode gains 3.1%, inside the noise floor of 4.2%, as the estimate of +2% to +3% said. The code decode
  gains 7.2%, above its noise (1.65 tok/s), with the same acceptance; the code test is for information only. H0
  accepts 2.14 tokens per step with the settings of T0 (2.20 in §6.7), so a part of the drop of T3 (2.10) is
  noise between boots.
- No deploy, by the gate of §7.1. The 0.49 GiB and the code decode gain do not outweigh a loss of 0.91% of the
  summed NLL, which is four times the run noise.

**#5, a K per layer: fails at every point.**

- At mean K 144, both rules are worse than a uniform K 144 (`PLAN_BETTER_RANK.md` §6.1: +3.29% with the floor rank,
  +3.51% with the count rank). The even rule is the worst (+4.26%), although it spreads the dropped share evenly.
- At mean K 152, the NLL passes (+2.74%), but the correct decisions fall from 41 to 37. A uniform K 152 lies, by the
  steps of `PLAN_BETTER_RANK.md` §6.1, at about +2.6% to +2.9%. So the allocation by weight share gains nothing
  over a uniform K.
- So the weight share of the router alone does not predict the cost of a cut in a layer. A better allocation needs
  a measured cost per layer (one replay per layer cut), which is outside this plan.
- No new prune plan follows. K 160 stays the floor (`PLAN_BETTER_RANK.md` §6.3). No point passed, so the prune with
  a different K in each layer, its full runs and your blind review do not apply. The folders `tp1-klg5-k144`,
  `tp1-kle5-k144`, `tp1-klg5-k152` and `tp1-kle5-k152` are deleted.

## 8. Step 3: offline studies, then builds (#6, #7)

### 8.1 #6: suffix drafts in the existing draft slots

A draft with a variable length hits the same limits as a dynamic depth: the fixed graph shapes and the grammar mask.
The version that fits fills the existing draft slots from a suffix match. When no match exists, DSpark fills them.

Study (no downtime, about half a day):

1. Write `suffix_match.py`. Tokenize the recorded bot sessions (`model_eval/data/rank/sessions.jsonl` in the eval
   worktree) with the tokenizer of the checkpoint.
2. At each output position, find the longest suffix match in the session's prompt, its earlier output, and the older
   sessions.
3. Count how many of the next d tokens the match predicts (d is the depth from Step 1).
4. Report, for match lengths m = 4, 8 and 16:
   - the share of output positions with a match of at least m tokens;
   - the mean number of correct tokens at those positions.

**Gate.** Estimate the gain as: share × (suffix mean − DSpark mean) ÷ (1 + DSpark mean). The DSpark mean is 2.14 at
depth 5. Build only if the estimate is at least 10%. The estimate is an upper bound, because DSpark also does well on
copied text.

Build (about two days, only if the gate passes):

1. Patch `dspark.py` (already a mount in the server repo). When a match of at least m tokens exists, put the match
   tokens in the draft slots. Keep the slot count and all shapes fixed.
2. Give each suffix token a one-hot draft probability. The standard rejection sampler then stays exact.
3. Test: the greedy output equals the live model's output on 20 prompts; bot decode × 5 against a reference in the same
   window.

**Gate.** The bot decode gain is above the noise floor.

**Result of the study (2026-11-10): the gate fails, so the build does not start.** `suffix_match.py` (tests:
`test_suffix_match.py`) ran in the server image, with no GPU, in about 11 s. It read the 66 sessions of
`model_eval/data/rank/sessions.jsonl`. An output position is one token of a target turn of the bot: the reasoning,
the content and the tool calls (599 turns, 675,623 positions). The search history holds 3,159,695 tokens, and a match
is at most 32 tokens long. The depth is 5, because the draft cannot go below its block size (§3).

| m | Positions with a match of at least m tokens | Correct tokens of the next 5 | Gate estimate at depth 5 |
|---|---|---|---|
| 4 | 44.0% | 2.36 | +3.1% |
| 8 | 20.2% | 3.45 | **+8.5%** |
| 16 | 10.5% | 3.98 | +6.2% |

- No estimate reaches 10%. The best is +8.5%, and it is an upper bound. At depths 3 and 4 the best is +6.0%.
- The reasoning tokens are 53% of the positions, and their estimate is about 0 at every m. The answer tokens alone
  (the content and the tool calls) reach +17.2% at m = 8. DSpark probably accepts more than 2.14 tokens on those
  tokens too, so that number does not prove a gain. A suffix draft for the answer tokens only is the one variant to
  study again.
- 31% to 35% of the matched positions find their match in an older session. These sessions repeat sandbox prompts,
  so the live bot reuses less text, and the real gain is probably lower.
- No deploy follows.

### 8.2 #7: a wider draft (96 or 128 experts)

The live draft keeps the 64 MTP experts with the highest bot rank of layer 42 (3.0 GB). A wider draft reads more bytes
per pass. So the acceptance must rise more than the draft time.

1. Build `dspark-draft-k96-bot` and `dspark-draft-k128-bot` with `build_dspark_draft.py`, as in
   `PLAN_PHYSICAL_PRUNE.md` §7.1: the bot order of layer 42, `--experts 96` or `128`, and the per-category reservation
   at 0. This needs no downtime.
2. Check that the runtime reads the expert count from the draft's `config.json`. If it does not, set
   `DSPARK_DRAFT_EXPERTS` to match.
3. Test with `tune_night.sh` and `DRAFT_DIR`. Boot the live draft as the reference. Boot each wide draft at the
   Step 1 depth and at that depth + 1, because a higher acceptance can move the best depth up. That is 5 points, about
   3 hours.

**Gate.** The bot decode beats the reference by more than the noise floor. The memory cost (about 1.5 or 3 GiB) works
against the memory goal, so the deploy is your decision.

**Steps 1 and 2 (2026-11-10, no downtime).**

- The plan file of the live draft was not saved. A rebuild found it: a copy of `REAP_K160_PLAN.json` with
  `mtp_ranked` in the weight order of layer 42 (`expert_rank_weight.json`). The 64-expert rebuild gives the live file
  byte for byte (sha256 `729159…cdcd3c`). The count rank (`expert_rank_bot.json`) matches only 57 of the 64 experts.
  The plan file is `results/dspark-draft-bot-plan.json`.
- A correction to `PLAN_PHYSICAL_PRUNE.md` §7: the live draft came from `--source data/tp1`, not from `tp1-k160t5`.
  Its `config.json` equals the one of `data/tp1` with `n_routed_experts: 64`. The MTP tensors are the same in both
  folders.
- The builds used the image's `build_dspark_draft.py` with no GPU, in under 10 s each:

| Draft | Size | More than the live draft | `n_routed_experts` |
|---|---|---|---|
| `dspark-draft-k64-bot` (live) | 2.94 GiB | — | 64 |
| `dspark-draft-k96-bot` | 4.14 GiB | +1.20 GiB | 96 |
| `dspark-draft-k128-bot` | 5.33 GiB | +2.39 GiB | 128 |

- The first 64 experts of each wide draft are the 64 of the live draft, in the same order.
- The runtime reads the expert count from the draft's `config.json`: the draft gets its own `ModelConfig`, and
  `dspark.py` builds the draft layers with that count. `DSPARK_DRAFT_EXPERTS` only sets the build of a missing draft
  in the entrypoint, so a test boot needs no change. No CUDA graph size depends on 64. One kernel choice changes: the
  native router kernel exists for 64 and 128 experts, and 96 experts take the exact PyTorch path, as the 160 experts
  of the target do.
- `tune_draft.sh` is the window of step 3: D0 (the live draft at depth 5), then W1 to W4 (96 and 128 experts at
  depths 5 and 6), each with the graph check and 5 bot decode runs. The depth cannot go below 5 (§3).
  `tune_night.sh` cannot do this test, because it sets `DRAFT_DIR` itself.

**Step 3: the window (2026-11-10, 19:46 to 22:40).** You moved the start of the windows to the afternoon, so this
window fit. `tune_draft.sh` ran right after the Step 2 window. The live server was down from 19:46 to 22:40, and the
restore was healthy at 22:40 (reply "OK."). Logs: `logs/tune-draft.log`, `results/speed-*-tune-{d0,w1,w2,w3,w4}*`.

| Point | Draft, depth | Weights | Bot decode (runs) | Accepted per step | Code decode | Against D0 |
|---|---|---|---|---|---|---|
| D0 | 64 experts (live), 5 | 74.09 GiB | 35.64 tok/s (35.12–36.25) | 2.23 | 41.14 tok/s | — |
| W1 | 96 experts, 5 | 75.29 GiB | 35.82 tok/s (34.91–36.72) | 2.28 | 42.05 tok/s | +0.5% |
| W2 | 96 experts, 6 | 75.29 GiB | 34.93 tok/s (34.13–35.75) | 2.37 | 43.49 tok/s | −2.0% |
| W3 | 128 experts, 5 | 76.48 GiB | **37.14 tok/s** (36.80–38.41) | 2.43 | 43.20 tok/s | **+4.21%** |
| W4 | 128 experts, 6 | 76.48 GiB | 35.66 tok/s (35.29–38.40) | 2.53 | 43.86 tok/s | +0.1% |

- Every point booted in 166 s to 195 s, and every graph check passed, also with the sizes 7, 14, 28 of depth 6.
- **W3 passes the gate, at the edge:** +4.21% against a noise floor of 4.2% (37.14 against 37.137 tok/s). Each of
  its 5 runs is above each run of D0. The acceptance rises at every position (2.23 to 2.43, +9%), and the code decode
  rises 5.0%. So the gain is most likely real, and about as large as the noise floor.
- **Depth 6 does not pay.** The 6th position adds only 0.10 to 0.12 accepted tokens, and W2 and W4 are not faster
  than D0. Depth 5 stays, also with a wider draft (§6.6).
- **96 experts add too little acceptance** (2.23 to 2.28) for the larger reads.
- **The cost:** the 128-expert draft takes 2.39 GiB more (76.48 against 74.09 GiB).
- The two wide drafts are deleted. The plan file `results/dspark-draft-bot-plan.json` rebuilds either one in under
  10 s.

**Deploy recommendation (your decision): W3 only if the memory is free to spend.** It gives +4.2% bot decode and
+5.0% code decode, with no output change, for 2.39 GiB. That works against the memory goal of this plan: the KV pin
of §6.7 saves only 0.75 GiB, and `PLAN_MODEL_ROUTER.md` §3 wants the room for a second model. If you want more
certainty first, run D0 and W3 once more in one window (`POINTS="d0:k64:5 w3:k128:5"`, about 75 min). To deploy:

1. Rebuild the draft into the server repo: the command of step 3 with `--experts 128` and
   `--output .../DeepSeek-v4-Flash-One-DGX-Spark/data/dspark-draft-k128-bot`.
2. Back up `compose-k160t5.yml` as `compose-k160t5.yml.bak.<date>_k128draft`, and point the draft mount (now
   `data/dspark-draft-k64-bot`) at the new folder.
3. Run `docker compose -f compose-k160t5.yml up -d` in the server folder (about 3 min; the graph sizes do not change).
4. Check the boot lines (76.48 GiB with the 8 GiB pin), one reply, and the next `daily-report` run.

To go back, copy the backup over the file and run `docker compose -f compose-k160t5.yml up -d`.

**Your decision (2026-11-10): skip the 128-expert draft for now.** The live draft stays `dspark-draft-k64-bot`. The
steps above and the plan file in `results/` stay ready for a later switch.

## 9. Step 4: a KV tier on the NVMe disk (#8)

The goal: a warm session skips the prefill after the pool evicts it, or after a restart. A prefill of 128k tokens
takes about 2 to 3 minutes. Its KV (about 3.7 GB at 29 KB per token) reads from the disk in about one second.

On this machine, a CPU tier gives nothing: the CPU and the GPU share the same 121 GiB. Only the disk adds capacity.
`KV_OFFLOAD_GB` stays at 0.

1. From the metrics of §5.1, count the prompt tokens per day that miss the prefix cache on prompts above 32k tokens.
2. Check the image for a disk KV connector (LMCache or the offload connector of vLLM).
3. Check that the connector supports the 432-byte NVFP4 record and the hybrid cache groups.

**Gate.** More than about 20 long misses per day, and a connector that supports the record. Then write a separate
plan.

### 9.1 Result (2026-11-10): blocked

⛔ **Step 1 is blocked.** It needs one week of metrics, and §5.1 has one snapshot of 6 hours. Tonight's windows
restart the live server, which sets the counters to zero. Start the week after the last window of this plan.

Steps 2 and 3 need no metrics, so they ran on the image (no downtime):

- **A connector exists.** The image's vLLM has the native offload connector (`offloading_connector.py`) with a
  file-system tier (`vllm/v1/kv_offload/tiering/fs`). The LMCache connector code is there, but the `lmcache` package
  is not installed.
- **The support for this model is not proven.** The connector has code for hybrid models, but one worker path
  asserts a single KV cache group (`distributed/kv_transfer/kv_connector/v1/offloading/worker.py:250`). This model
  has several groups (the compressed attention and the sliding window), and no file of the connector names the
  `nvfp4_ds_mla` record. A test boot with the connector decides.
- **The low hit rate of §5.1 comes first.** The prefix cache hit 1.0% of the prompt tokens. If the bot's prompts do
  not share prefixes, a disk tier has nothing to serve. Find the cause before the week of metrics ends.

## 10. Not now (#9, #10, #11)

| Idea | Why not now | Look again when |
|---|---|---|
| #9 Attention and shared expert to EXL3 | It needs a calibrated EXL3 conversion from the original model, and nobody has run one. The kernel speed on GB10 is unknown | A conversion is published, or Step 2 leaves the memory short |
| #10 K2 for cold experts | It needs kernel work: two bit rates in one layer | #5 shows which experts are cold. It is the largest memory lever (up to 10 GiB) |
| #11 Tree verification, streamed NVFP4 prefill, basis sharing, HOPE | Research projects | — |

## 11. The memory budget

Measured on the live settings (boot log of 2026-11-07 and `logs/kv8-reboot.log`):

| Part | GiB |
|---|---|
| Weights of the target and the draft (one load) | 74.09 |
| KV cache (pinned) | 8.00 |
| Peak activation (profiled) | 5.08 |
| Non-torch memory | 4.21 |
| CUDA graphs | 0.26 |
| Sum | 91.64 |

After the boot, `free` showed 23.7 GB available.

| Idea | Change | Output change |
|---|---|---|
| #2 KV pin | About −0.75 GiB (−1.1 GiB at 160k) | No |
| FP8 draft head | About +0.5 GiB | No |
| #4 FP8 target head | About −0.49 GiB; the draft copy is then not needed | Yes |
| #5 K per layer | −3 to −6 GiB, after a loader patch | Yes |
| #10 K2 for cold experts | Up to −10 GiB | Yes |

The routed experts hold nearly all the memory. After #2, each real saving goes through the replay gate.

## 12. Risks

| Risk | Effect | Guard |
|---|---|---|
| A new depth gets no full graph | The verify step uses a slow path; the test shows a false loss | The graph check of §6.3 |
| A larger prefill chunk needs more MLA workspace than the warmup locked | An assertion stops the server | The deep warm turn of T4 |
| The FP8 head kernel does not run on GB10 | T3 and #4 need another kernel | The head check of T3 |
| A smaller KV pool keeps fewer warm sessions | More cold prefills | #3 makes cold prefills faster; the data of §5.1 |
| A larger prefill chunk raises the memory peak | Less room for a second model | The memory check of T4 |
| A gain stays inside the noise | No reason to deploy | 5 bot runs per point, in one window, with its own reference |
| A deploy of `PLAN_FEWER_EXPERTS.md` moves the acceptance | The best depth changes | §6.6: test the depth again |

## 13. Effort and timeline

| Step | Work | Downtime | When |
|---|---|---|---|
| 0 | One day for the tools and tests; the prompt data after one day of traffic | None | After the fewer-experts night |
| 1 | One evening | About 3.5 h | The first free evening after Step 0 |
| 1 deploy | 15 min | 4 to 8 min | Your decision |
| 2 | Two days for the head patch and the K allocation | About 3.5 h | After the fewer-experts decision |
| 3 | Half a day per study; two days for the suffix build | About 3 h per test window | Beside Step 2 |
| 4 | A study first | — | After one week of metrics |

Step 1 and its deploy fit in the first week. Steps 2 and 3 take the second week.
