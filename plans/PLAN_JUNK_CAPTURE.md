# Plan: the route capture of the junk prompts

Status (2026-11-17 00:05): done. The night chain (`junk_night.sh` in the bisect worktree, at your ask) ran window 1,
the report, the builds and window 2 from 22:35 to 23:57. c3 comes from the MD step: 88 experts put back drop c3 from
18/20 to 3/10, but miss the pass rule. c2 comes from the cut to K 160, not from top-k 5. Results in §10.

## 1. The question

The junk probes (`results/junk-stock`, `results/junk-k160`) put the junk tokens on the prune. Two probe prompts
show a step:

| Prompt | Stock (K 216, top-k 6) | K 160 (top-k 5) | MD (K 149 mean, top-k 5) | Step that adds the junk |
|---|---|---|---|---|
| c2 (a line number after a lexer quote) | 0/10 | 6/10 | 3–6/10 | stock → K 160 |
| c3 (a line number after an LRU quote) | 0/10, sane 0.99 | 0/10, sane 0.98 | 7–10/10 in 4 runs | K 160 → MD |
| c1 (punctuation in reasoning) | 3/10 | 1/10 | 1–5/10 | none: the stock model has it too |

This plan asks one question for each step:

- **c3:** does the stock model send c3 to the 486 experts that the MD step removed? If so, does a small set of
  those experts, put back into MD, remove the c3 junk?
- **c2:** the step from stock to K 160 has two changes: 2,408 experts go, and top-k goes from 6 to 5. Which change
  makes the c2 junk?

## 2. The facts the plan uses

- **The sets nest.** In every layer, MD keeps a subset of K 160, and K 160 keeps a subset of stock (checked from
  the `keep` lists of the two `PRUNE.json` files on 2026-11-16).
  - K 160 removed 2,408 experts: 56 in each of the 43 layers.
  - The MD step removed 486 more, in layers 3–42 only (6 to 20 per layer).
  - Layers 0–2 route by a hash of the token id. K 160 sent the hash entries of a removed expert to the most similar
    kept expert (`MASK.json`, `hash_rule`).
- **Top-k:** stock routes 6 experts per token. K 160 and MD route 5 (`topk_checkpoint.py`, `TOPK.json`).
- **The bot-traffic baseline** (`results/capture-rank.jsonl`, the 63 segments of the 11-06 capture): 5.37% of the
  stock picks go to the K 160 removed set, and 1.43% go to the MD-extra set. In layers 0–2, 13.50% of the picks go to
  the K 160 removed set. The weight shares (from the 26 rows with no NaN) are 4.77% and 1.10%.
- **One expert** holds 9.04 MiB (22,833,271,776 bytes over 2,408 experts). 1 GiB holds about 113 experts.
- **The capture stack** (`compose-capture.yml`) boots stock with Model Runner V1 and `--enable-return-routed-experts`.
  - Both `/v1/chat/completions` and `/v1/completions` return `routed_experts`: an array of (prompt tokens, 43, 6).
    The second route was checked in the image on 2026-11-16, so the raw probe prompts can go in as they are.
  - `routing_sums.npz` gives the count and the router weight per layer and expert. The client reads it before and
    after each request.
  - On 11-06 the boot took 225 s, and the capture ran at 1,000 to 1,400 tokens per second.
  - On 11-06, at `GPU_MEMORY_UTILIZATION` 0.93, only 2.7 GB stayed available, with a KV cache of 473,251 tokens.
    The VS Code and Claude sessions can hold more memory today. This plan therefore pins the KV cache.
- **The coding traffic:** `results/agent-eval/pilot` has 40 DeepSeek transcripts (C, D, L, R and W tasks, seed 0).
  The prompts at their last turns hold 564,119 tokens. The longest holds 81,242 tokens. The probe prompts c1, c2, c3
  and d3 come from four of them (C1, C2, C3 and D3).

## 3. The method

Capture on stock only. Stock shows the experts that the full model wants. A capture on K 160 or MD would need the
loader patch under Model Runner V1, which is not tested.

**What the capture records:**

1. The probe prompts: c1, c2, c3, d3 and the greedy prompt of `results/junk-bisect/prompts.json`. Each prompt goes
   3 times through `/v1/completions` with `max_tokens: 1` and a new `cache_salt`. The 3 rounds show how stable the
   routing is.
2. The 40 pilot transcripts: one request per task, through `/v1/chat/completions`, with the messages up to the last
   turn and the tools of agent-eval. The server's own template renders them, as in the pilot.

For each request, keep the routed ids of every token and the weight sums of the request.

**What the report computes:** it computes these values for each prompt, for each pilot task family, and for the
bot baseline:

- The share of picks and the share of weight on the K 160 removed set. Report layers 0–2 apart from layers 3–42.
- The share of picks and the share of weight on the MD-extra set.
- The same shares at the probe position (the last row) and over the last 256 rows (the quote and the JSON around it).

**The rules, fixed before the data:**

- **G1 (a coverage gap for c3):** on the MD-extra set, c3 has at least twice the bot share of picks (2.86% or more).
  The whole prompt or the last 256 rows can pass it.
- **G2 (a coverage gap for c2):** on the K 160 removed set, in layers 3–42, c2 has at least twice the bot share.
- **The X rule:** the pool is the 486 MD-extra experts.
  1. Score each pool expert by its weight share in its layer, on the pilot tasks without C2 and C3. These two tasks
     stay out, so that the test on c2 and c3 is held out.
  2. Take the experts in score order until X holds 50% of the pool's weight on those tasks.
  3. Stop at 113 experts (1 GiB) in any case.
- **G3 (X generalizes):** X holds at least 50% of c3's MD-extra weight.

## 4. The work before the window (no server stop)

CAUTION: do not change files in the main tree while a window runs. Build the tools in a worktree.

1. **`junk_routes.py capture`**: the client of §3.
   - The raw prompts go through `/v1/completions`. The pilot transcripts go through `/v1/chat/completions`.
   - It builds the pilot messages as the earlier probe did (`agent_eval.SYSTEM`, the task prompt, the turns with
     their tool results, `agent_eval.NUDGE` after a turn with no call). It imports `agent_eval` from
     `~/Projects/dsv4-specialize-agent-eval/agent-eval`, as `capture.py` imports the replay.
   - It writes one JSONL row per request (ids as a compact array, the weight-sum delta, the token count, the
     seconds). A rerun skips the rows that are done.
2. **`junk_routes.py report` and `select`**: the shares of §3, G1 to G3, the X rule, and the new drop file
   (`results/cold-md.json` without X).
3. **`junk_capture.sh`**: the window script of §5.
   - It sources `run_lib.sh`, so it runs under `job run` and shows on machine-dash.
   - It takes the Qwen stop and restart and the memory guard from `bisect_junk.sh`.
4. **`compose-capture-junk.yml`**: an override of `compose-capture.yml` with `MAX_MODEL_LEN: "131072"` and
   `CAPTURE_EXTRA_ARGS: "--kv-cache-memory-bytes 6979321856"` (6.5 GiB). At mtp0, 131,072 tokens need 6.26 GiB.
5. **Tests:**
   - `test_junk_routes.py`: a stub server that returns known `routed_experts` and sums; the share math on small
     sets; the X rule; the drop file.
   - A dry run of `junk_capture.sh` with a stub `run_lib.sh`, as `test_skip_scripts.py` does.

**The pull requests** (each 400 lines or fewer, each with its tests):

1. `junk_routes.py capture` with its tests.
2. `junk_capture.sh` and `compose-capture-junk.yml` with the dry-run test.
3. `junk_routes.py report` and `select` with their tests. This PR can follow the window, because the rules are
   fixed in §3.

The main tree has an uncommitted change to `pair_run.sh` (`KV_BYTES`, the sampling note). The Qwen restart uses it.
Keep it in place, or land it first as its own small PR.

## 5. Window 1: the capture (about 45 min; DeepSeek is down about 30 min)

WARNING: a full memory can hard-reset the Spark. The memory guard kills the test container below 1.5 GiB available.
The 6.5 GiB KV pin holds much less than the 473,251-token KV cache of the 0.93 boot of 11-06.

CAUTION: the script stops DeepSeek and Qwen. Do not start it between 05:00 and 10:00 on a weekday (the Hermes cron
jobs). You start it with a `!` command, because auto mode blocks a script that stops the server.

The steps of `junk_capture.sh`:

1. Wait for `START_AFTER`. Stop Qwen (the process group from `pair_run.sh`'s PID file).
2. `stop_main`: wait for an idle bot (up to 30 min), stop DeepSeek, and check that 100 GiB is available.
3. `boot capture-junk` with `-f compose-capture.yml -f compose-capture-junk.yml`. The boot must be healthy within
   15 min. If it fails, go to step 7.
4. **Smoke:**
   - One short request must move `routing_sums.npz`.
   - One raw prompt must return `routed_experts` with 43 layers and 6 columns.
   - The boot lines must show the 6.5 GiB pin: a "GPU KV cache size" below 150,000 tokens. The stock boot of 11-15
     at the same pin gave 70,514 tokens. The 0.93 boot of 11-06 gave 473,251.
   - If one check fails, go to step 6.
5. **Capture**, with the memory guard on, and a stop at `RESTORE_AT`:
   1. The probe prompts, 3 rounds (about 40k tokens per round).
   2. The 40 pilot transcripts (about 10 min).
6. `down`.
7. `restore`. Wait 5 min, then start Qwen with `pair_run.sh` (the start waits up to 15 min for `NEED_MB`).

The time per step: the stop takes 2 min, the boot 4 to 8 min (a cold compile can add some), the smoke 1 min, the
capture 12 min, and the restore 10 min. Set `BOOT_BY` to the start plus 20 min. Set `RESTORE_AT` to the start plus
50 min. I give you a tested command with deadlines from "now" when PR 2 is in.

## 6. The analysis and the decision (no server stop, about 15 min)

Run `junk_routes.py report`. It writes `results/junk-capture/report.md`. Then apply this table:

| Result | Next step |
|---|---|
| G1 and G3 pass | Build MD + X (§7), and run window 2 with all four points |
| G1 passes, G3 fails | The c3 experts are specific to c3. Report it. Window 2 runs `stock5` only, or waits |
| G1 fails | The MD step does not remove experts that c3 needs. The junk comes from a drift of the hidden state. Report it, and do not build MD + X |
| G2 passes | Note it for c2. The `stock5` point still decides between top-k and the cut |

## 7. The builds (no server stop)

1. **MD + X:**
   1. `mask_checkpoint.py --drop-file results/cold-mdx.json`.
   2. `topk_checkpoint.py --top-k 5`.
   3. `prune_experts.py` through `prune_run.sh`. On 11-11 this took about 11 min with the bot up.
   - The folder holds X × 9.04 MiB more than MD (1 GiB at most).
2. **`stock5`:** `topk_checkpoint.py --source data/tp1 --top-k 5`. It rewrites the hash tables and the config only.
   The expert files of `data/tp1` are root:600, so run it in a root container, as `prune_run.sh` does.
3. Run `--check` on both folders. Check that the links resolve inside the test container.

CAUTION: only about 7.7 GB is available beside both servers. Watch MemAvailable during the prune, and stop it below
3 GiB. If it stops, run the build at the start of window 2, after the script stops Qwen.

## 8. Window 2: the probe (about 1 h 15 min; DeepSeek is down about 1 h)

Use `bisect_junk.sh` in the bisect worktree, as for the stock and K 160 runs. Add the points `mdx` and `stock5`
(mtp0, 64k context, 6.5 GiB KV, `BISECT_MODEL_DIR` set), each with its `expect` strings. Run the points `mdsmall mdx
stock5 mdsmall2` at 10 runs per probe.

| Point | Pass |
|---|---|
| `mdx` | c3 junk at 2/10 or less with sane mass 0.9 or more, while both MD controls give 5/10 or more |
| `stock5` | c2 junk at 4/10 or more: top-k 5 makes the c2 junk. At 1/10 or less: the cut makes it |

## 9. After this plan

- **If `mdx` passes:** write a deploy plan for MD + X. It needs the replay gate against MD, the memory check beside
  Qwen (X × 9.04 MiB less margin), the speed test and a review, as for MD.
- **If `stock5` points to top-k:** weigh top-k 6 (a slower decode) against the c2 junk.
- **c1:** it is in the stock model too, so this plan does not cover it.

## 10. Results (2026-11-16, the night chain)

- **Window 1** (22:35 to 22:49): 55 rows, no errors, 523,734 tokens in 8 min. The first try at 22:07 skipped the
  capture: the smoke prompt was too short to run eagerly (#95).
- **The report** (`results/junk-capture/report.md`): coding traffic uses the removed experts about twice as much as
  the bot traffic.

  | Group | K 160 removed picks | MD-extra picks |
  |---|---|---|
  | Bot baseline | 5.37% | 1.43% |
  | Pilot, 40 tasks | 10.62% | 3.30% |
  | c2 | 11.18% (10.96% in layers 3-42) | 4.13% |
  | c3 | 10.48% | 5.27% |

  - G1 passes (c3: 3.7 times the bot share). G2 passes (c2: 2.3 times the bot share in layers 3-42).
  - X has 88 experts (0.78 GiB). They hold 50.05% of the pool's held-out share.
  - G3 passes by a small margin: X holds 50.77% of c3's MD-extra weight.
  - c1 has the largest removed share (13.65%), but its junk does not change with the prune. A high share alone does
    not prove a cause.
- **The builds:** `tp1-stock5-k216` and `tp1-prunemdx5-k151` (MD + X). Both self-checks pass. The prune took 9 min.
  MD + X loads 67.31 GiB, 0.77 GiB more than MD.
- **Window 2** (23:00 to 23:37, `results/junk-mdx/table.md`), junk with the mean sane mass:

  | Point | c1 | c2 | c3 | All |
  |---|---|---|---|---|
  | MD control | 1/10 (0.83) | 8/10 (0.30) | 9/10 (0.17) | 18/30 |
  | MD + X | 1/10 (0.81) | 10/10 (0.00) | 3/10 (0.67) | 14/30 |
  | Stock, top-k 5 | 5/10 (0.52) | 0/10 (0.99) | 0/10 (0.99) | 5/30 |
  | MD control again | 4/10 (0.57) | 6/10 (0.35) | 9/10 (0.19) | 19/30 |

- **The answers:**
  1. c3 comes from the MD-extra experts. X drops c3 from 18/20 (both controls) to 3/10 (p = 0.0017). It misses the
     pass rule (2/10 or fewer, sane mass 0.9 or more). X holds only half of c3's MD-extra weight, so a partial fix fits.
  2. c2 comes from the cut to K 160, not from top-k 5: stock at top-k 5 gives 0/10 with 0.99. MD + X does not touch the
     K 160 removed set, so c2 stays junk (10/10).
  3. c1 is not the prune.
- **Qwen** did not start after the window: 28,523 MB available, below `NEED_MB` 28,600. A retry at 23:57 gave 28,467 MB.
- The Hermes cron jobs and the watchdog timer stay paused until you resume them.
