# Plan: do the cold experts matter?

**Status:** 2026-11-11: done. The bot does not need the 257 cold tool experts (M1, +0.42%). The repeat window shows
that MD, both cold sets (486 experts, 4.29 GiB after a prune), has no measurable cost: +0.18% against R0 over two
replays. Next: the loader-patch plan (§5.5). No deploy · **Author:** colin-spark ·
**Date:** 2026-11-10
**Source:** the question "of the 74 GiB of weights, how much does the bot not need?" The routed experts hold
60.71 GiB of the 74.09 GiB that the live server loads. The other parts are on the path of every token.
**Related:** `PLAN_SERVE_TUNE.md` §7.3 (a K per layer, points N1 to N3), `PLAN_BETTER_RANK.md` (the capture and the
K 144 failure), `PLAN_FINAL_REVIEW.md` (the deploy chain), `PRUNE.md` (the live model)
**Scope:** four mask folders, one replay window on port 8890, and new tools in this repo. No change to the live server.

| Step | Status |
|---|---|
| §3 The data check (no downtime) | ✅ 2026-11-10: the cold tail is mostly mandatory tool experts |
| §6 The tools and the folders (no downtime, about half a day) | ✅ 2026-11-10 23:20–23:35: the tools pass their tests; the four folders are built and checked (§6.1) |
| §7 The replay window (about 3 h 15 min downtime) | ✅ 2026-11-10 23:33 to 2026-11-11 02:11 (§7.2): the R0 check passes; the memory held; all smoke tests pass. The Hermes cron jobs are paused for the morning of 2026-11-11 (§7.1) |
| §5 The decision | ✅ 2026-11-11 (§5.4): M1 has no measurable cost (+0.42%): the bot does not need the cold tool experts. D1 (+0.70%) and MD (+0.46%) contradict each other: repeat them before the loader-patch plan. No deploy |
| §7.3 The repeat window (R0, D1, MD) | ✅ 2026-11-11 08:30 to 10:29 (§7.3): the R0 check passes (+0.22%); D1 +0.03% and MD −0.10% against R0; the memory held. The Hermes cron jobs stay paused |
| §5.5 The decision after the repeat | ✅ 2026-11-11: the conflict of D1 and MD was replay noise (up to 0.56% for one folder). MD passes in both replays: write the loader-patch plan of §8. No deploy |

---

## 1. Short answer

- **The question:** can the experts that the bot's traffic rarely uses go, with no cost on the replay?
- **The finding of §3:** 268 kept experts get less than 0.1% of the routing weight of their layer. 257 of them are
  "mandatory" tool experts, and `mask_checkpoint.py` never drops a mandatory expert. Every earlier cut kept them, so
  K 144 had to drop warmer experts. It then failed the NLL gate (+3.29% to +3.90%).
- **The test:** four masks in one replay window. R0 is the live set. M1 drops the cold tool experts, D1 drops the cold
  ordinary experts, and MD drops both. Each point compares against R0 from the same window.
- **The outcome:** a measured answer for each kind of cold expert. If MD passes, a prune could free about 4.3 GiB.
- **What it does not give:** the masks free no memory and change no speed. A prune with a different K in each layer
  needs a loader patch with its own plan (§8).

## 2. Terms

| Term | Meaning |
|---|---|
| Weight share | The share of an expert in the summed router weight of its layer, from `capture.py --rank --order weight` |
| Cold expert | A kept expert with a small weight share in the capture of the live set |
| Mandatory expert | One of the top 32 experts of a layer in either tool category of `REAP_K216_PLAN.json` (`agentic_tool_trajectory`, `tool_calling`). `mask_checkpoint.py` never drops it |
| Ordinary expert | A kept expert that is not mandatory |
| The live set | The 160 experts per layer of `tp1-k160t5`, with the 56 live drops per layer |
| Mask | A folder with all 216 experts, in which the router never chooses a dropped expert. It tests quality, not memory |
| Prune | A folder that holds only the kept experts. It frees the memory |

## 3. What the data shows (2026-11-10, no downtime)

**The data.**

- **The capture:** `results/capture-profile-k160.jsonl`, from the K 160 mask at top-k 6. It covers 63 segments of
  55 sessions, with 2.84 M prompt tokens and 2.42 M routed tokens per layer.
- **The rank:** `capture.py --rank results/capture-profile-k160.jsonl ... --order weight` makes the rank file of the
  live set. At K 160, its drop order gives exactly the 56 live drops of each layer.
- **The replay:** 60 targets from 15 held-out sessions. No replay session is in the capture.

**The cold tail of layers 3–42.** Layers 0–2 (hash routing) have no expert below 0.1%.

| Set | Experts | Mandatory | GiB | Weight share lost per layer, mean (max) |
|---|---|---|---|---|
| Weight share below 0.05% | 118 | 118 | 1.04 | 0.06% (0.18%) |
| Weight share below 0.1% | 268 | 257 | 2.36 | 0.36% (0.73%) |
| Weight share below 0.2% | 894 | 512 | 7.89 | 2.88% (5.38%) |
| The coldest ordinary experts, up to 1% of the weight | 229 | 0 | 2.02 | 0.92% (1.00%) |

Each expert holds 9.04 MiB (60.71 GiB / 6,880 experts).

**Why it matters.** About 46 of the 160 kept experts in each layer are mandatory. Together they hold 26% of the routing
weight, but many of them are the coldest experts of the layer. To free 2 GiB, a cut of cold tool experts loses 0.33%
of the routing weight per layer. A cut of ordinary experts loses 0.92% for the same memory.

**A hypothesis, not a fact.** REAP chose the mandatory experts from an outside data set
(`0xSero/deepseek-v4-flash-reap-observations-v2`: 4,096 tool-calling rows and 2,253 agent trajectories). The bot's
tools and its prompt format differ, so the bot can need a different set of tool experts. The test of §4 checks this.

## 4. The points

Each point drops the live drops plus the set in the table.

| Point | Drops in addition to the live drops | Experts | GiB after a prune | Weight share lost per layer, mean (max) | K in layers 3–42 | Question |
|---|---|---|---|---|---|---|
| R0 | None (the live set as a mask) | 0 | 0 | 0 | 160 | The reference |
| M1 | The mandatory experts with a weight share below 0.1% | 257 | 2.27 | 0.33% (0.73%) | 145–159 | Does the bot need the cold tool experts? |
| D1 | The coldest ordinary experts, while the lost weight stays at or below 1% | 229 | 2.02 | 0.92% (1.00%) | 151–156 | Does the bot need the cold ordinary experts? |
| MD | M1 and D1 | 486 | 4.29 | 1.25% (1.72%) | 140–154 | Do the two costs add? |

The rules for all points:

- Layers 0–2 keep the live 160 experts and the live hash tables.
- Each folder is a mask of `data/tp1` (216 experts), then `topk_checkpoint.py --top-k 5`, as for the folders of
  `PLAN_SERVE_TUNE.md` §7.3.
- R0 is a mask, not the live prune folder. A mask and a prune of the same set differ by 0.37% NLL (§5.1), so each
  point must compare against a mask.
- M1 and D1 free about the same memory. The pair thus shows which kind of cold expert costs less.

## 5. Gates and the decision

### 5.1 The noise (measured)

| Pair | NLL | Tool names | Decisions |
|---|---|---|---|
| `baseline-k216` against its repeat | 0.23% | 0 calls | 1 of 60 |
| The K 160 mask against the K 160 prune (`prune5-k160-vs-botmask5-k160.txt`) | 0.37% | 1 call (1.2 points) | 0 |
| R0 against its repeat (§7.3, added 2026-11-11) | 0.12% | 1 call (1 target changes) | 1 (5 targets change) |
| D1 against its repeat (§7.3) | 0.56% | 0 calls (2 targets change) | 4 (4 targets change) |
| MD against its repeat (§7.3) | 0.44% | 1 call (3 targets change) | 2 (6 targets change) |

The repeat window of §7.3 shows that the first two rows were too small for the masks. No target gives the same NLL
twice for the same folder. One target moves by up to 25% of its NLL, and 4 to 6 decisions change in each repeat. The
net counts hide this, because the changes go both ways.

### 5.2 The comparisons

Compare each point in two ways:

1. **Against R0, from the same window.** The point has "no measurable cost" if all of these are true:
   - the summed NLL rises by at most 0.5%;
   - the tool names fall by at most 1 call (1.2 points);
   - the decisions fall by at most 1;
   - the smoke test passes (a JSON schema, a tool call with `auto`, and one with `required`).
2. **Against `baseline-k216`:** the gate of `PLAN_SERVE_TUNE.md` §7.1. The summed NLL rises by at most 3%, the tool
   names reach at least 98% of the baseline, and the decisions fall by no more than 3.

Also report two diagnostics. They are not gates, because some skills have few targets:

- the NLL change against R0 for each skill;
- each target whose tool name changes.

**The R0 check.** R0 must match the earlier K 160 mask (`botmask5-k160`, NLL 0.4541) within 0.5%. If it does not, the
folders or the image changed. Then stop, and do not use the other points.

### 5.3 How to read the result

| Result | Meaning | Next step |
|---|---|---|
| M1 has no measurable cost | The bot does not need the cold tool experts. The mandatory rule protects 2.27 GiB that the bot does not use | Change the rule for this bot: protect a tool expert only if it is warm in the bot's capture. Then plan the K cuts again with that rule |
| M1 costs tool names, or its smoke test fails | The tool experts matter on rare tokens, in spite of the low share | Keep the rule |
| D1 has no measurable cost | The bot does not need the cold ordinary tail | Use D1 as a floor for the K per layer |
| D1 has a cost | Even the last 1% of the weight matters. This agrees with the K 144 result | Stop the ordinary cuts |
| MD has no measurable cost and passes the baseline gate | About 4.3 GiB can go | Write the loader-patch plan (§8) |
| MD costs more than M1 and D1 together | The two costs interact | Prune only the better single set |
| All three points have no measurable cost | The cut can grow | A second window: all experts below 0.2% (894 experts, 7.89 GiB), and then a cut between MD and that |

### 5.4 The decision (2026-11-11)

By the table of §5.3, with the results of §7.2:

- **M1 has no measurable cost** (+0.42% against R0, decisions +2, tool names +1, the smoke test passes). It also
  passes the baseline gate (+2.80%). So the bot does not need the 257 cold tool experts on this traffic, and the
  mandatory rule protects 2.27 GiB that the bot does not use.
- **D1 has a cost** (+0.70% against R0, decisions −4), and it fails the baseline gate (+3.10% against +3%).
- **MD has no measurable cost** (+0.46%, decisions +1), and it passes the baseline gate (+2.84%).

**D1 and MD contradict each other.** MD drops every expert that D1 drops, plus the experts of M1. A larger cut cannot
cost less. Yet MD scores better than D1: 0.24 points less NLL and 5 more correct decisions. So the difference between
the two is the noise of one replay, and both sit at the 0.5% limit: D1 just above it, MD just below it. The noise of
§5.1 (0.23% to 0.37% NLL, 1 decision) is smaller than what this window shows between two nested sets.

The decision:

1. **M1 is the clear result.** Its NLL is the lowest of the three points, its decisions rose, and its smoke test passed.
   For this bot, protect a tool expert only if it is warm in the bot's capture. The next K plan must use that rule.
2. **D1 and MD give no decision from one window.** Do not write the loader-patch plan of §8 yet. First repeat R0, D1
   and MD in one window, so that each point has two replays: `POINTS="r0 d1 md"` with `cold_replay.sh`, about 2 h 15
   min. Write the loader-patch plan only if MD stays at or under +0.5% against R0 in both replays. The four folders
   stay on disk for that window (48 GB).
3. **No deploy.** A mask frees no memory (§8), and nothing changed on the live server, except the KV pin of
   `serve-tune/PLAN_SERVE_TUNE.md` §6.8.

### 5.5 The decision after the repeat (2026-11-11)

With the two windows of §7.2 and §7.3:

| Point | Against R0, first window | Against R0, repeat | Mean of both | Decisions over both, against R0 | Baseline gate |
|---|---|---|---|---|---|
| M1 | +0.42% | — | — | +2 (one window) | pass |
| D1 | +0.70% | +0.03% | +0.36% | −3 | fail, then pass |
| MD | +0.46% | −0.10% | **+0.18%** | +1 | pass, pass |

1. **The conflict of D1 and MD was replay noise.** The same folder moves by up to 0.56% between two windows (§5.1),
   so the 0.24-point gap between D1 and MD in the first window says nothing.
2. **MD passes.** It stays at or under +0.5% against R0 in both replays, which is the condition of §5.4. Its smoke test
   passes twice, and it passes the baseline gate twice. So the bot does not need the 486 cold experts of MD on this
   traffic, and a prune of them frees about 4.29 GiB.
3. **Next: the loader-patch plan of §8** (`PLAN_LOADER_PATCH.md`). A prune with a different K in each layer needs
   it, and without it no cut frees memory. M1 needs no repeat, because MD contains M1 and passes twice.
4. **The gate needs two replays for a small cut.** One replay moves by up to 0.56%, more than the 0.5% limit. For the
   next cut, compare the mean of two replays per point, each against an R0 of the same window.
5. **A larger cut is open.** By §5.3, all three points have no measurable cost (D1 on the mean), so the cut can grow:
   all experts below 0.2% (894 experts, 7.89 GiB). Plan it after the loader patch, with two replays per point.
6. **No deploy.** A deploy also needs the full runs, the judge and the blind review (§8).

## 6. The tools and the folders (no downtime, about half a day)

CAUTION: the Step 2 window of `PLAN_SERVE_TUNE.md` runs `tune_replay.sh` and `run_lib.sh`. Bash reads a script while it
runs. Do not edit those two files until that window ends.

Make a branch from `tune-2b-klayer`, because it holds `mask_checkpoint.py --keep-per-layer`. Then do these steps:

1. Write the rank file of the live set:
   `capture.py --rank results/capture-profile-k160.jsonl results/expert_rank_k160.json --order weight`.
2. Add `--drop-file FILE` to `mask_checkpoint.py`:
   - The file is JSON: `{layer: [current ids to drop]}` for all 43 layers.
   - The tool drops exactly those ids. A mandatory id may drop.
   - The option needs `--out`, and it excludes `--keep` and `--keep-per-layer`.
   - `MASK.json` records the file, its sha256, and the number of mandatory experts that each layer drops.
   - The edits stay as they are: the gate bias for layers 3–42 and the hash remap for layers 0–2.
3. Write `cold_sets.py`:
   - It reads the rank file of step 1, `REAP_K216_PLAN.json` (for the mandatory set) and the live `MASK.json`.
   - It writes four drop files, `results/cold-{r0,m1,d1,md}.json`, with the rules of §4.
   - It prints the table of §4 as a check.
   - `--report R0.jsonl POINT.jsonl` prints the two diagnostics of §5.2.
4. Write the tests:
   - `test_cold_sets.py`, on a synthetic rank: the M1 and D1 rules, the union of MD, the live drops in every file,
     and layers 0–2 untouched.
   - One more test in `test_mask_checkpoint.py`: a drop file can drop a mandatory expert, and the R0 file gives the
     same drop lists as `--keep 160` with the rank of step 1.
5. Build the four folders:
   - Run `mask_checkpoint.py --drop-file results/cold-<point>.json --out tp1-cold<point>-k<K>`.
   - Then run `topk_checkpoint.py --top-k 5` and `topk_checkpoint.py --check` on each folder.
   - `<K>` is the mean K, rounded: 160, 154, 155 and 149. `compose-mask.yml` boots
     `tp1-${MASK_NAME}-k${MASK_K}`, so K is only a label.
   - Each build takes about 50 s and 12 GB, so the four folders need 48 GB. The disk has 2.8 TB free.
6. Write `cold_replay.sh`, with the point flow of `tune_replay.sh` (boot, checks, down, the two-failure rule and the
   restore):
   - It runs the points `r0 m1 d1 md`, in that order.
   - For each point, it boots the folder, runs `prefill_probe.py --smoke`, runs the replay, and stops the server.
   - After each replay, it compares the point against R0 and against `baseline-k216`.

The code is about 300 lines with the tests, so it fits in one pull request.

### 6.1 Result (2026-11-10, 23:20 to 23:35)

The work is on the branch `cold-experts`, made from `main`: `tune-2b-klayer` is merged into `main` since 2026-11-10.
The code went into two pull requests (the tools, then the window script), so that each stays small.

1. **The rank file** `results/expert_rank_k160.json`: 63 segments, 2,838,972 prompt tokens, no NaN slot. At K 160, its
   drop order gives exactly the 56 live drops of each layer (checked against the live `MASK.json`).
2. **`mask_checkpoint.py --drop-file FILE`**, as specified. The rule for the mandatory experts is now one function,
   `mandatory()`, which `drop_order` and `cold_sets.py` both use.
3. **`cold_sets.py`** wrote `results/cold-{r0,m1,d1,md}.json`. Its check table equals the table of §4 in every number:

   | Point | Experts | Mandatory | GiB | Weight share lost per layer, mean (max) | K in layers 3–42 | Mean K |
   |---|---|---|---|---|---|---|
   | R0 | 0 | 0 | 0 | 0 | 160 | 160 |
   | M1 | 257 | 257 | 2.27 | 0.33% (0.73%) | 145–159 | 154 |
   | D1 | 229 | 0 | 2.02 | 0.92% (1.00%) | 151–156 | 155 |
   | MD | 486 | 257 | 4.29 | 1.25% (1.72%) | 140–154 | 149 |

   The R0 file equals the live drops and the drops of `--keep 160` with the rank file of step 1, in all 43 layers. Each
   file holds the live drops, and layers 0–2 hold only the live drops.
4. **The tests:** `test_cold_sets.py` (4 tests: the M1 and D1 rules, the union of MD, the live drops in every file,
   layers 0–2 untouched, the report) and one more test in `test_mask_checkpoint.py` (a drop file drops a mandatory
   expert; a drop file with the drops of `--keep 6` gives the same mask). `python -m unittest`: OK.
5. **The folders** `tp1-coldr05-k160`, `tp1-coldm15-k154`, `tp1-coldd15-k155` and `tp1-coldmd5-k149`: each build took
   45 to 55 s and holds 12 GB. `topk_checkpoint.py --check` passes on each, and no link is broken. The hash tables of
   R0 got the same remap as the live mask (166,409, 170,172 and 166,531 entries in layers 0–2). The intermediate mask
   folders are deleted.
6. **`cold_replay.sh`**, with the point flow of `serve-tune/tune_replay.sh`. A dry run with a stub `run_lib.sh`
   checked the point order, the smoke and report lines, and the stop after a failed R0 check. One change against the
   plan: the test boots pin the KV cache at 7.25 GiB, as the live server does since 2026-11-10 23:04. A mask of 216
   experts left only 1.2 to 4 GiB free on 2026-11-10 (`serve-tune/PLAN_SERVE_TUNE.md` §7.4), and the smaller pin
   adds 0.75 GiB. The KV size does not change the NLL.

## 7. The replay window (about 3 h 15 min downtime)

CAUTION: this window stops the bot's model server for about 3 h 15 min. Start it only with your go. Do not start it
between 05:00 and 10:00 on a weekday, or during another test.

| Order | Point | Minutes |
|---|---|---|
| 1 | R0 | 44 |
| 2 | M1 | 44 |
| 3 | D1 | 44 |
| 4 | MD | 44 |
| 5 | The restore of the live server | 10 |

Each point takes about 4 min to boot, 2 min for the smoke test, 35 min for the replay, and 3 min to stop. The script
skips a point that cannot end before `RESTORE_AT`.

```
tmux new -d -s cold "START_AFTER=... BOOT_BY=... RESTORE_AT=... ~/Projects/dsv4-specialize/cold_replay.sh"
```

A mask of `data/tp1` loads 95.4 GiB. The N1 point of `PLAN_SERVE_TUNE.md` booted such a mask on 2026-11-10 and had
4 GiB free after the boot. The memory watch of `run_lib.sh` stops a point that runs short.

### 7.1 Before the window (2026-11-10, 23:24 to 23:33)

- **The cron jobs, on your go.** Both Hermes cron jobs use DeepSeek through port 8888: `daily-followups` (every 20 min,
  05:00 to 09:40 on weekdays) and `daily-report` (05:30 on weekdays). Both are paused with `hermes cron pause` since
  23:24. The tmux session `cron-resume` runs `logs/cron-resume-20261111.sh`: at 10:05 on 2026-11-11 it resumes both
  jobs (`hermes cron resume`), so the next runs are on Thursday morning. The backup of `~/.hermes/cron/jobs.json` is
  `logs/hermes-jobs.json.bak.20261110_cold`. `daily-report` still needs a pin to the renamed model (`PRUNE.md` §5).
- **The memory.** Before each boot, `run_lib.sh` waits for 110 GiB free (vLLM's boot check at util 0.90 needs 109.5).
  The live server alone left 25 GiB free; after its stop, 115 GiB. The 7.25 GiB pin gives each mask 0.75 GiB more
  room. The memory watch stops a client below 1 GiB. The overnight guard (tmux `guard`) restores the live server at
  04:15 if it is down.
- **The start:** `tmux new -d -s cold "START_AFTER=<now> BOOT_BY='2026-11-11 02:40' SPEED_BY='2026-11-11 02:50'
  RESTORE_AT='2026-11-11 03:30' ~/Projects/dsv4-specialize/cold_replay.sh"` at 23:33. The bot was idle, and the live
  server stopped at 23:33:01.

### 7.2 Result (2026-11-10 23:33 to 2026-11-11 02:11)

The live server was down for 2 h 38 min. The restore was healthy at 02:11 (reply "OK.", gateway active), with the
7.25 GiB pin. Logs: `logs/cold-w1/` (`cold-replay.log` and the memory logs); result files: `results/cold-w1/`, not in
git. They moved there before the repeat of §7.3, because the replay skips the targets that its output file holds.

| Point | Boot | Free memory after the boot, lowest in the replay | Smoke | Summed NLL against K 216 | Against R0 | Decisions (K 216: 41) | Tool names (K 216: 77 of 83) |
|---|---|---|---|---|---|---|---|
| R0 | 225 s | 3 GiB, 1.79 GiB | pass | +2.38% | — | 41 | 77 |
| M1 | 210 s | 3 GiB, 1.53 GiB | pass | +2.80% | **+0.42%** | 43 | 78 |
| D1 | 211 s | 3 GiB, 1.33 GiB | pass | **+3.10%** | **+0.70%** | **37** | 79 |
| MD | 225 s | 3 GiB, 1.57 GiB | pass | +2.84% | **+0.46%** | 42 | 79 |

- **The R0 check passes:** R0 is +0.09% against `botmask5-k160` (the limit is 0.5%). The folders and the image did not
  change, so the other points can be read against R0.
- **The memory held.** Every boot found at least 114 GiB free, and no replay came near the 1 GiB stop.
- **The smoke requests passed on all four points** (the JSON schema and both tool choices, no assertion). So even
  without the 257 cold tool experts, the model still writes valid tool calls and valid JSON.
- **No tool name got worse at any point.** Three targets improved (one subagent target in all three points, one
  earnings target in D1).

The NLL change against R0 for each skill (a diagnostic, not a gate; 7 or 8 targets per skill):

| Skill | M1 | D1 | MD |
|---|---|---|---|
| earnings | −0.00% | +0.70% | +0.62% |
| filings | +0.48% | +0.45% | −0.24% |
| other | +0.16% | +1.88% | +0.64% |
| quant | +1.01% | +1.62% | +1.90% |
| research | −0.53% | −0.18% | −0.46% |
| stock-update | +0.86% | +0.20% | +0.43% |
| subagent | +0.31% | +0.28% | −0.50% |
| watchlist | +1.41% | +1.64% | +1.15% |

Quant and watchlist pay most at every point.

### 7.3 The repeat window (2026-11-11, 08:24 to 10:29)

**The start, on your go:** the decision of §5.4 asked for a repeat of R0, D1 and MD.

1. The Hermes cron jobs were still paused from §7.1. The tmux session `cron-resume` stopped at 08:23, so the jobs
   stay paused after this window too. Resume them with `hermes cron resume 0983231e9695` and
   `hermes cron resume 89988a054346`.
2. The first window's files moved to `results/cold-w1/` and `logs/cold-w1/` (§7.2).
3. The window started with:
   `tmux new -d -s cold2 "POINTS='r0 d1 md' LOG=.../logs/cold-replay-2.log START_AFTER='2026-11-11 08:30'
   BOOT_BY='2026-11-11 10:45' SPEED_BY='2026-11-11 10:45' RESTORE_AT='2026-11-11 11:30' .../cold_replay.sh"`.
4. A guard (`GUARD_AT=11:50 RECHECK_AT=12:10 overnight_guard.sh`, tmux `guard`) restarts the live server if it is
   down after the window.

The live server stopped at 08:30:08 and was down for 2 h 0 min. The restore was healthy at 10:29 (reply "OK.",
gateway active), with the 7.25 GiB pin. Logs: `logs/cold-replay-2.log` and `logs/memory-cold-*.log`; result files:
`results/cold-*`, not in git.

| Point | Boot | Free memory after the boot, lowest in the replay | Smoke | Summed NLL against K 216 | Against R0 | Decisions (K 216: 41) | Tool names (K 216: 77 of 83) |
|---|---|---|---|---|---|---|---|
| R0 | 226 s | 2 GiB, 1.29 GiB | pass | +2.50% | — | 40 | 78 |
| D1 | 226 s | 2 GiB, 1.38 GiB | pass | +2.52% | **+0.03%** | 41 | 79 |
| MD | 211 s | 2 GiB, 1.46 GiB | pass | +2.39% | **−0.10%** | 40 | 78 |

- **The R0 check passes:** R0 is +0.22% against `botmask5-k160`, and +0.12% against the R0 of §7.2.
- **The memory held.** Every boot found 115 GiB free, and no replay came near the 1 GiB stop.
- **All three points pass the baseline gate.** D1 failed it in §7.2 (+3.10%), and passes now (+2.52%).
- **The tool-name change of §7.2 was noise.** The subagent target `20260809_172245_54cf37` that improved at every point
  of §7.2 also improves at R0 here, with no drop.
- **The NLL per skill is noise at this size.** For example, filings at MD went from −0.24% to +2.17%. Do not read a
  skill from one replay of 7 or 8 targets.

## 8. What a pass does not give

- **Memory:** a mask frees no memory. A prune with a different K in each layer needs a loader patch, because the
  runtime reads one expert count for all layers (`PLAN_SERVE_TUNE.md` §7.3). Write that plan only if MD, or N3 of
  `PLAN_SERVE_TUNE.md`, passes.
- **Speed:** none. The decode reads only the routed experts, so a cold expert costs memory, not time.
- **A deploy:** it also needs the full runs, the judge and your blind review, as for the live pick
  (`PLAN_FINAL_REVIEW.md`).

## 9. Risks

| Risk | Effect | Answer |
|---|---|---|
| The capture ran at top-k 6, and the live model uses top-k 5 | The shares differ a little from the live routing | The masks use top-k 5, so the replay measures the live routing |
| The capture counts prompt tokens, and the replay scores answer tokens | An expert that is cold in prompts can be warm in answers | The replay measures exactly this |
| The replay has 60 targets and about 83 tool calls | A tool that the replay never calls can need a cold expert | The smoke test covers the JSON schema and the tool calls. The full runs cover the rest before a deploy |
| A mask of 216 experts leaves about 4 GiB free | A point can run short of memory | The memory watch stops the point, and the script goes to the next one |
| R0 does not match `botmask5-k160` | The window gives no answer | The R0 check of §5.2 stops the reading before a wrong decision |

## 10. The relation to `PLAN_SERVE_TUNE.md` §7.3

The points N1 to N3 of that plan run in the Step 2 window of 2026-11-10. They differ from this plan in three ways:

- They rank the experts by the capture of the 216-expert model. This plan uses the capture of the live set.
- They keep the mandatory rule. This plan tests it (M1).
- They can keep an expert that the live set dropped. This plan cuts only inside the live set.

Read N1 to N3 before this window. If N3 passes, D1 will probably pass too. M1 asks a new question either way.

## 11. Timeline

| Work | Time | Downtime |
|---|---|---|
| §6: the tools, the tests and the folders | About half a day | None |
| §7: the window | About 3 h 15 min | About 3 h 15 min |
| §5: the report and the decision | About 30 min | None |
| §7.3: the repeat window (R0, D1, MD) | About 2 h 10 min | About 2 h 10 min |
