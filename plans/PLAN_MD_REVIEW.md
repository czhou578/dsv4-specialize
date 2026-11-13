# Plan: The deploy review of the MD prune

**Status:** 2026-11-12: done. The MD prune passes rules 1, 2 and 3, and you accept the hard faults of rule 4
(§8.1). A deploy is possible. No deploy: the deploy and the use of the 4.18 GiB wait for your go · **Author:**
colin-spark · **Date:** 2026-11-11
**Related:** `PLAN_LOADER_PATCH.md` §7.1 (the decision that asks for this review), `PLAN_FINAL_REVIEW.md` (the
method: the full runs, the LLM judge, the blind review), `PLAN_FEWER_EXPERTS.md` §9 (the last review of this kind),
`PLAN_OVERNIGHT_MASK_GRID.md` §2 (the safety rules of a test boot)
**Scope:** two run scripts, a report script and its test in this repo; four new runs in `~/.hermes-e2e/`; one night
of downtime on the live server. No change to the live bot, the server repo or the server config. No deploy.

| Step | Status |
|---|---|
| §3 The same code and the same date | ✅ 2026-11-11 16:06: the bot code is `a0fd27b` for all runs; the two bot sessions confirmed the freeze |
| §4 The base runs on the live model (no downtime) | ✅ 2026-11-11 16:06 to 22:14: `live-1` (188 min) and `live-2` (180 min), 19 of 19 prompts each; both checks have 0 FAIL lines |
| §5 The test window: the runs on the MD prune | ✅ 2026-11-11 22:15 to 2026-11-12 05:28 (7 h 13 min of downtime): the boot in 165 s, the smoke passes, `md-1` (210 min) and `md-2` (217 min), 19 of 19 prompts each, the restore at 05:27 (§6.2) |
| §6 The checks and the judge (no downtime) | ✅ 2026-11-12 05:28 to 15:25: the checks, the blind page and 114 judge pairs with 0 errors. Rule 1 and rule 2 pass. At 15:26 `review_finish.sh` found no step left (§6.2) |
| §7 Your blind review of 20 pairs | ✅ 2026-11-12: 9 W / 7 L / 4 T for the MD prune; rule 3 passes (§7.1) |
| §8 The decision | ✅ 2026-11-12: you accept the hard faults. A deploy is possible; no deploy yet (§8.1) |

---

## 1. Goal

The MD prune (`tp1-prunemd5-k149`) removes 486 cold experts from the live model. It loads at 69.79 GiB, against
74.09 GiB for the live model, and it frees 4.18 GiB (`PLAN_LOADER_PATCH.md` §6.2). It passes the replay screen and
the bot speed test. The replay screen has only 60 targets and about 83 tool calls, so a deploy also needs the full
runs, the judge and your blind review, as for the live pick.

This plan asks one question: **is the MD prune worse than the live model on real bot prompts, end to end?**

The test window gave one caution: the code decode test is 4.9% slower, with 7.5% fewer draft tokens accepted. So this
review also looks at the prompts that write code (the `quant` skill) and at the run time of each skill.

## 2. Design

### 2.1 The prompts

All 19 held-out bot prompts, as in `PLAN_FINAL_REVIEW.md` §2.1 (`--research 8`). Do not use the calibration
prompts: the experts were ranked on them.

### 2.2 The runs and the pairs

The base is the live model, not the stock model. The question is the cost of the cut against what the bot has now.

| Runs | Labels | Server |
|---|---|---|
| Base: the live model (K 160, top-k 5) | `live-1`, `live-2` | the live server, port 8888, no downtime |
| New: the MD prune (K 140 to 160, top-k 5) | `md-1`, `md-2` | the patched image, port 8890, the live settings |

- **Test pairs:** 2 new runs × 2 base runs × 19 prompts = **76 pairs**.
- **Control pairs:** `live-1` against `live-2`, and `md-1` against `md-2`: 2 × 19 = **38 pairs**. One model is on
  both sides, so a control shows the noise and the bias of the judge at the same scale.
- **The same date.** All four runs fall within about 16 hours. The research prompts read live web data, and the
  judge must not see a date gap as a quality gap. The old baselines of `PLAN_FINAL_REVIEW.md` are 6 to 8 days old,
  so they are not used.
- **The same settings.** The MD boot uses the live KV pin (7.25 GiB) and the live draft
  (`dspark-draft-k64-bot`). The 4.18 GiB that MD frees stays unused here.

### 2.3 The pass rules

The rules of `PLAN_FINAL_REVIEW.md` §2.3, with a one-sided sign test at p < 0.05 on the pairs with a winner:

| Pairs with a winner | 15 | 20 | 30 | 40 | 50 | 60 | 70 | 76 |
|---|---|---|---|---|---|---|---|---|
| Fail at this many losses or more | 12 | 15 | 20 | 26 | 32 | 37 | 43 | 46 |

1. **The judge test:** the MD prune fails if its losses in the 76 test pairs reach the line.
2. **The judge controls:** neither control may reach the line in either direction. If one does, the judge has a bias
   that the test cannot separate, and rule 3 decides alone.
3. **Your blind review (§7):** the MD prune fails if you prefer the live answer in as many pairs as the line for the
   number of pairs with a winner (15 of 20, 12 of 15).
4. **Hard faults:** a wrong number, a broken format, a missing save, a literal `</s>`, or a call to an unknown tool.
   You decide each case. Compare each one with the live runs: a fault that both models make does not separate them.

The MD prune passes if rules 1 and 3 pass and you accept the hard faults of rule 4.

### 2.4 Readings with no gate

- **The five gates of step F** (`model_eval.e2e.check --compare`): tool-call validity, call order, the report
  marker, the takes JSON and the number fidelity, for each MD run against each live run.
- **The skills:** the wins, losses and ties of each skill, with the `quant` skill first.
- **The weak prompts:** a prompt that loses against both live runs.
- **The run time of each skill** (`done.json`), the mean of the MD runs against the mean of the live runs. The live
  runs share the server with the bot, and an agent run can take a different path, so this is a rough reading. A
  large change in the `quant` skill is the place to look for the code result.

### 2.5 The judge

The judge is the live model on port 8888, the same judge for every pair. It can prefer its own output, and that bias
works against the MD prune, so it makes rule 1 stricter, not weaker. The controls do not measure it.

## 3. The same code and the same date

- The runs use the bot code in `~/.hermes/stock-research` (`main` at `a0fd27b`) and the eval worktree (detached at
  the same commit). A change between the base runs and the MD runs would mix a code change into the result.
- On 2026-11-11 at 16:06 I told the other sessions on this machine (`stock-research-ec`, `stock-research-b1`,
  `dsv4-specialize-f1`) the server times, and asked the two bot sessions for a code freeze until 08:30 on
  2026-11-12. `stock-research-b1` confirmed.
- The Hermes cron jobs `daily-report` and `daily-followups` stay paused until the review ends. The
  `hermes-daily-watchdog` timer still runs at 06:20. On 2026-11-10 it wrote a fallback daily report during a run,
  and the runner's guard marked one prompt `CHANGED`. The sandbox works on copies, so the result does not change.

## 4. The base runs on the live model (no downtime)

`review_live.sh` runs `live-1`, then `live-2`, on port 8888, then the check of each run. The runs are clients of the
live server: a bot message waits behind the current request. A memory watch stops a run below 1 GiB available, and a
rerun skips the done prompts. The last log line is "live runs done". Log: `logs/review-live.log`.

Time (estimate): about 3.5 hours for each run (the runs of the live pick took 212 and 208 minutes), so from 16:06 to
about 23:15.

## 5. The test window: the runs on the MD prune

**Caution:** the live server is down for about 7.5 hours. The bot does not answer in this time.

`review_window.sh` waits for "live runs done" in the base-run log and for 22:30. Then:

1. It stops the live server when the bot is idle, and checks that no other server runs and 110 GiB are available.
2. It boots `tp1-prunemd5-k149` on the patched image, as P1 of `PLAN_LOADER_PATCH.md` §6, and logs the pool line.
3. It runs the smoke requests (the JSON schema and the two tool-call modes). If they fail, it restores the live
   server with no runs.
4. It runs `md-1`, then `md-2`, with the memory watch of §4.
5. It stops the test server and restores the live server ("OK." and the gateway).

The deadlines:

| Deadline | Time | Action |
|---|---|---|
| `START_AFTER` | 2026-11-11 22:30 | No stop of the live server before this time |
| `BOOT_BY` | 2026-11-12 01:00 | If the base runs have not ended, no window. Start the window on the next night |
| `RESTORE_AT` | 2026-11-12 08:30 | The runs stop and the restore starts. A rerun on the next night skips the done prompts |
| The overnight guard | 2026-11-12 09:00 and 09:20 | If port 8888 is down, the guard (tmux `guard`) restores the live server |

## 6. The checks and the judge (no downtime)

After the restore, `review_window.sh` goes on with the live server up:

1. The check of each MD run, and `check --compare` against each live run (§2.4).
2. The blind page: `judge live-1 live-2 md-1+md-2 --page results/review-md.html --sample 20 --seed 13`. It needs no
   model, so it is ready a few minutes after the restore. You can start §7 while the judge runs.
3. The judge: `judge live-1 live-2 md-1`, then `… md-2` (38 pairs each), then the controls `judge live-1 live-2` and
   `judge md-1 md-2` (19 pairs each). About 5.6 minutes per pair (`PLAN_FEWER_EXPERTS.md` §9.1), so about 10.5
   hours for 114 pairs, until about 19:00 on 2026-11-12.
4. The report (`review_report.py`) in the window log: rules 1 and 2, the weak prompts, the skills, the `</s>` count
   of each run, and the run time of each skill.

### 6.1 The safety net (`review_finish.sh`)

The window script has two gaps after the window. A failed request (for example, a lost connection) ends a judge call,
and the script does not try again. A failed restore ends the script (exit 3) before the checks and the judge. The
script was already running, and a change to a running bash script can break it, so a second script closes the gaps:

1. It waits until `review_window.sh` has ended, for any reason, and until port 8888 is healthy (up to 4 hours, so
   that the guard can restore the live server at 09:00 or 09:20).
2. It runs the checks and the compares again (seconds), and it makes the blind page if it is missing.
3. It runs each judge call again, up to 3 attempts. The judge skips the rows that are already in its file, so a
   complete call ends in seconds.
4. It writes the report to `results/review-report.txt` and to `logs/review-finish.log`.

It only sends requests to the live server. A missing judge file shows "NO DATA" in the report, not "pass". The test on
2026-11-11 used a stub judge that failed once for each call: both calls passed on the second attempt.

Files: `model_eval/results/e2e/judge-live-1+live-2-vs-md-{1,2}.jsonl`, `judge-live-1-vs-live-2.jsonl` and
`judge-md-1-vs-md-2.jsonl` in the eval repo; `results/review-*` in this repo.

### 6.2 The result of the window and the judge (2026-11-12)

**The window.** The base runs ended at 22:14. On your go, the window started at 22:15, not at 22:30.

- The boot: 165 s, 69.79 GiB loaded, a KV cache of 265,498 tokens, and 26.8 GB available. The 15 expert counts
  (140 to 160) share one scratch pool (decode 10.9 MiB). The smoke passes: the JSON schema and both tool-call modes.
- The runs: `md-1` in 210 min and `md-2` in 217 min, 19 of 19 prompts each, all with exit 0.
- The restore: the live server was healthy at 05:27, with the reply "OK." and the gateway active. The downtime was
  7 h 13 min.
- The judge: from 05:28 to 15:25 (about 10 hours for 114 pairs), 0 errors, and no call failed.
  `review_finish.sh` found the window script ended at 15:26, and every step was complete.

**Rules 1 and 2 (the LLM judge).** A win is a win for the MD prune. In a control, a win is a win for the second run.

| Test | W / L / T | Pairs with a winner | Fail line | Result |
|---|---|---|---|---|
| `md-1` against `live-1` and `live-2` | 11 / 10 / 17 | 21 | 15 | pass |
| `md-2` against `live-1` and `live-2` | 14 / 6 / 18 | 20 | 15 | pass |
| **Rule 1: the 76 test pairs** | **25 / 16 / 35** | 41 | 27 | **pass** |
| Rule 2: control `live-1` against `live-2` | 5 / 7 / 7 | 12 | 10 (both directions) | pass |
| Rule 2: control `md-1` against `md-2` | 7 / 4 / 8 | 11 | 9 (both directions) | pass |

**The five gates of step F.** All gates pass, with two flags. Neither flag separates the models:

- `md-1` fails the number fidelity on the "Hi!" prompt (`8eb30d36`), against both live runs. The prompt has no data
  task. The number share of `md-1` is 3.1%, against 3.4% for `live-1`.
- `md-2` against `live-1` fails the report marker on `54cc02`. `live-2` has no marker on this prompt either, so the
  flag is not a result of the MD prune.

**The readings with no gate (§2.4).**

- The hard faults that a script can find: all four runs have 19 of 19 prompts with exit 0 and a report. No run has a
  literal `</s>` or a call to an unknown tool.
- The weak prompts: three prompts lose against both live runs. These are the "Hi!" prompt and the subagent prompt
  `54cf37` in `md-1`, and the research prompt `68c0317c` in `md-2`. Each prompt is weak in one MD run only.
- The skills (W / L / T for the MD prune):
  - quant 2/0/2, filings 4/0/0, earnings 3/1/0, daily-report 3/0/1, watchlist 3/0/1, stock-update 1/0/3.
  - research 6/6/20.
  - subagent 3/5/4 and other 0/4/4. These are the losses, and both counts are too small for the sign test.
- The run time: the mean of all prompts is 12,806 s for the MD runs against 11,022 s for the live runs (+16.2%).
  - Per model call, the MD runs take 80.9 s and 78.3 s, and the live runs 71.3 s and 71.4 s (about +11%).
  - This is more than the decode test of `PLAN_LOADER_PATCH.md` §6.2 (−1.3% for the bot, −4.9% for code). A model
    call also holds the tool time and long prefills, and the MD runs ran at night on the test compose. So this is
    a rough reading.
  - The `quant` time of +105% comes from one run: `md-2` took 311 s on `quant-drift`, against 100 s in the other
    three runs, with the same 3 calls.

## 7. Your blind review of 20 pairs

1. I put the 20 pairs of `results/review-md.html` on a private review page: one pair per prompt and one more, each
   with a random live run and MD run. The order and the A/B side are random, and no page shows a run name.
2. For each pair, choose A, B or tie, and mark any hard fault (rule 4). Do not read the judge rows first.
3. I map your answers with `results/review-md.key.json` and apply rule 3.

This takes about 2 hours: the answers are long research reports.

### 7.1 The result of the blind review (2026-11-12)

**Who reviewed.** A Claude session filled the review page: each note starts with "[Claude]". You read the answers
and count them as your review. This reviewer is not the live model, so the bias of §2.5 does not apply to it.

**Rule 3.** The MD prune has 9 wins, 7 losses and 4 ties. 16 pairs have a winner, so the fail line is 12 losses.
The MD prune has 7 losses, so rule 3 passes (p of a loss = 0.77).

| Pair | Runs | Prompt | Skill | Result for MD | Faults of MD | Faults of live |
|---|---|---|---|---|---|---|
| 1 | live-2 / md-2 | `3c708852` | stock-update | win | – | number |
| 2 | live-1 / md-2 | `quant-drift` | quant | win | – | tool |
| 3 | live-2 / md-2 | `38c7eb58` | filings | win | – | other |
| 4 | live-2 / md-2 | `fd10d6b7` | research | loss | garbled, number | garbled |
| 5 | live-1 / md-1 | `8eb30d36` | other | loss | – | – |
| 6 | live-2 / md-2 | `3d277250` | research | loss | – | garbled |
| 7 | live-2 / md-2 | `495a6ca2` | research | win | number, garbled | number |
| 8 | live-2 / md-2 | `54cc02` | subagent | win | garbled | garbled |
| 9 | live-1 / md-1 | `68c0317c` | research | loss | number, garbled | number, garbled |
| 10 | live-2 / md-1 | `ccd5f1fb` | other | win | garbled, number | garbled |
| 11 | live-1 / md-2 | `54cf37` | subagent | loss | garbled | garbled |
| 12 | live-2 / md-1 | `5a7a4ea3` | watchlist | tie | – | – |
| 13 | live-2 / md-1 | `669cff0b` | research | loss | garbled, number, other | garbled |
| 14 | live-1 / md-2 | `6186758a` | research | tie | garbled, number | garbled |
| 15 | live-2 / md-1 | `daily-report` | daily-report | tie | – | – |
| 16 | live-1 / md-2 | `17bd3e34` | research | win | garbled, number | garbled, number |
| 17 | live-2 / md-1 | `c7f2cb` | subagent | loss | garbled, number, other | garbled, number |
| 18 | live-1 / md-2 | `5a7a4ea3` | watchlist | tie | – | – |
| 19 | live-2 / md-1 | `0d31caa5` | earnings | win | garbled, number | garbled |
| 20 | live-2 / md-1 | `3e2f0342` | research | win | garbled, number | garbled |

By skill, all 7 losses are in research (3/4/1), subagent (1/2/0) and other (1/1/0; the loss is the "Hi!" prompt).
The short tool tasks have 4 wins (quant, filings, earnings, stock-update) and 3 ties (watchlist, daily-report).

**Rule 4 (the hard faults that the review marked).**

| Fault | Pairs of MD | Pairs of live | Note |
|---|---|---|---|
| A wrong number | 10 | 5 | 6 pairs have it on the MD side only, 1 pair on the live side only (sign test p = 0.06). The faults are of one kind on both sides: a wrong percentage, multiple or range in a long report |
| Stray tokens (garbled) | 12 | 12 | Both models have them at the same rate, so they do not separate the models |
| A tool fault | 0 | 1 | Pair 2: Nasdaq HTTP 403 warnings in the live run |
| Other | 2 | 1 | MD: pair 13 does not name a short report, and pair 17 gives an acquisition to the wrong company. Live: pair 3 has a duplicated link |

The answers with the notes and the key are in `results/review-md-answers.json` (not in git).

## 8. The decision (yours)

| Result | Next |
|---|---|
| Rules 1 and 3 pass, and you accept the hard faults | A deploy of the MD prune is possible. Decide the use of the 4.18 GiB (KV room, or the second model of the router plan). The deploy mounts the two built files of `image-patch/perlayer/` in the server repo's compose file and serves `tp1-prunemd5-k149`. The rollback is the current compose file |
| Rule 1 or rule 3 fails | No deploy. Look at the skills and the weak prompts: if the losses are in one skill, the cold set of that skill is the next question |
| A control fails | Rule 3 decides alone. Look at the control rows for the bias |
| The `quant` skill loses, or its run time grows a lot, and the rest is even | You decide: the bot's traffic has few code prompts, but the cut costs them something |

### 8.1 The decision (2026-11-12)

The first row of the table applies:

- Rule 1 passes (25 W / 16 L / 35 T), and both controls of rule 2 pass (§6.2).
- Rule 3 passes (9 W / 7 L / 4 T, §7.1).
- You accept the hard faults of rule 4. The wrong numbers are of the same kind as in the live runs, and their excess
  is not significant (p = 0.06). The stray tokens are equal (12 against 12). No run has a literal `</s>`, a call to
  an unknown tool or a missing save.
- The `quant` row does not apply: `quant` has 2/0/2 by the judge and a win in the blind review. Its +105% of run time
  comes from one run.

**Result:** a deploy of `tp1-prunemd5-k149` is possible. No deploy is done, and the server repo has no change.

Your decisions on the open items (2026-11-12):

1. The 4.18 GiB is for the second model of the router plan. The KV pin stays at 7.25 GiB.
2. The deploy plan is `PLAN_MD_DEPLOY.md`. It includes the check of the time per model call before and after the
   deploy (§6.2 shows about +11% in the runs, against −1.3% in the decode test).
3. The Hermes cron jobs `daily-report` and `daily-followups` run again since 2026-11-12 17:16. Both jobs are now
   pinned to the provider `custom` and the model `deepseek-v4-flash-0731-k160t5`. Before, `daily-report` failed
   with "global inference config drifted" (last run 2026-11-09). The backup of the job file is
   `logs/hermes-jobs.json.bak.20261112_resume`.

## 9. Risks

| Risk | Effect | Answer |
|---|---|---|
| The base runs take longer than 9 hours | No window tonight | `BOOT_BY` stops the window. Start it on the next night; the base runs are then one day older than the MD runs |
| A bot session changes the code during the runs | The pairs mix a code change into the result | The freeze of §3; check `git -C ~/.hermes/stock-research log -1` before the window and after it |
| The MD runs do not end by 08:30 | Fewer pairs | The judge uses the prompts that both runs have. A rerun on the next night finishes the run |
| The watchdog writes the daily state at 06:20 | One prompt is marked `CHANGED` | The sandbox works on copies (§3) |
| A restore fails | The bot stays down | The overnight guard at 09:00 and 09:20 |

## 10. Effort and order

| Step | Time | Downtime |
|---|---|---|
| §4 The base runs | About 7 hours, from 16:06 | None |
| §5 The window | About 7.5 hours, from 22:30 or later | About 7.5 hours |
| §6 The checks and the judge | About 10.5 hours | None |
| §7 Your review | About 2 hours, during §6 | None |

The total is about 1.5 days, mostly unattended.
