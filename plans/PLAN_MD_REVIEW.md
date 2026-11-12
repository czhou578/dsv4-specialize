# Plan: The deploy review of the MD prune

**Status:** 2026-11-11: in progress. The base runs on the live model run (§4). The test window is set for tonight
(§5) · **Author:** colin-spark · **Date:** 2026-11-11
**Related:** `PLAN_LOADER_PATCH.md` §7.1 (the decision that asks for this review), `PLAN_FINAL_REVIEW.md` (the
method: the full runs, the LLM judge, the blind review), `PLAN_FEWER_EXPERTS.md` §9 (the last review of this kind),
`PLAN_OVERNIGHT_MASK_GRID.md` §2 (the safety rules of a test boot)
**Scope:** two run scripts, a report script and its test in this repo; four new runs in `~/.hermes-e2e/`; one night
of downtime on the live server. No change to the live bot, the server repo or the server config. No deploy.

| Step | Status |
|---|---|
| §3 The same code and the same date | ✅ 2026-11-11 16:06: the bot code is `a0fd27b` for all runs; the two bot sessions confirmed the freeze |
| §4 The base runs on the live model (no downtime) | 🔨 2026-11-11 16:06: `live-1`, then `live-2` (`review_live.sh`, tmux `live`) |
| §5 The test window: the runs on the MD prune | ⏳ tonight, after the base runs: `md-1` and `md-2` (`review_window.sh`, tmux `window`) |
| §6 The checks and the judge (no downtime) | ⏳ after the window, in the same script |
| §7 Your blind review of 20 pairs | ⏳ the page is ready after the window |
| §8 The decision | ⏳ |

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

Files: `model_eval/results/e2e/judge-live-1+live-2-vs-md-{1,2}.jsonl`, `judge-live-1-vs-live-2.jsonl` and
`judge-md-1-vs-md-2.jsonl` in the eval repo; `results/review-*` in this repo.

## 7. Your blind review of 20 pairs

1. I put the 20 pairs of `results/review-md.html` on a private review page: one pair per prompt and one more, each
   with a random live run and MD run. The order and the A/B side are random, and no page shows a run name.
2. For each pair, choose A, B or tie, and mark any hard fault (rule 4). Do not read the judge rows first.
3. I map your answers with `results/review-md.key.json` and apply rule 3.

This takes about 2 hours: the answers are long research reports.

## 8. The decision (yours)

| Result | Next |
|---|---|
| Rules 1 and 3 pass, and you accept the hard faults | A deploy of the MD prune is possible. Decide the use of the 4.18 GiB (KV room, or the second model of the router plan). The deploy mounts the two built files of `image-patch/perlayer/` in the server repo's compose file and serves `tp1-prunemd5-k149`. The rollback is the current compose file |
| Rule 1 or rule 3 fails | No deploy. Look at the skills and the weak prompts: if the losses are in one skill, the cold set of that skill is the next question |
| A control fails | Rule 3 decides alone. Look at the control rows for the bias |
| The `quant` skill loses, or its run time grows a lot, and the rest is even | You decide: the bot's traffic has few code prompts, but the cut costs them something |

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
