# Plan: The final review of bot K 160 at top-k 5

**Status:** 2026-11-06: done. Rules 1 to 4 pass (you accept the hard faults). The pick passes. Next is Phase 3
· **Author:** colin-spark · **Date:** 2026-11-05
**Related:** `PLAN_MODEL_SPECIALIZE.md` §4.5 rule 5 (this plan carries it out), `PLAN_E2E_RUNNER.md` §15.9 (step F
on the pick), `PLAN_OVERNIGHT_MASK_GRID.md` §2 (the safety rules for a test boot)
**Scope:** a small change to `model_eval/e2e/judge.py` in `stock-research-model-eval`; new runs in
`~/.hermes-e2e/`; one night of downtime on the main server. No change to the live bot or the server config.

| Step | Status |
|---|---|
| §3 Judge changes (one pull request) | 🔨 2026-11-05: done on `final-review-judge` (366858b), 155 lines, 621 tests pass; pull request #59 open, not merged (the runs used the branch) |
| §4 More baseline runs (main server, no downtime) | ✅ 2026-11-05 16:14–22:05: `baseline-1` and `baseline-2` +4 research prompts, `baseline-3` 19 of 19; all exit 0, 0 tool failures, guard ok, attempt 1 each; lowest free memory 1,047 MB (VS Code Server start) |
| §5 More top-k 5 runs (one night of downtime) | ✅ 2026-11-05 22:30 to 2026-11-06 03:16: `botmask5-k160` +4 research prompts and `botmask5-k160-r2` 19 of 19, all exit 0, guard ok; main server healthy at 03:16. Compare against each base: `botmask5-k160` 4 of 5 gates (marker `54cc02`, the known prompt problem); `botmask5-k160-r2` 3 of 5 (marker `54cc02`, and tool-call validity on `17bd3e34`: the mask called an unknown tool `terminate_placeholder`, the agent errored that call and went on). 02:26: two VS Code extension hosts grew to 3.5 GB each. The memory watch stopped the run twice (lowest 436 MB). Attempt 2 went on |
| §6 The judge on 114 pairs and 57 control pairs | ✅ 2026-11-06 03:17–16:25: 114 test pairs 44 W / 19 L / 51 T (p_loss 0.9996), rule 1 passes; the 3 controls pass both directions, rule 2 passes (§6.1) |
| §7 The blind review of 20 pairs (you) | ✅ 2026-11-06: you prefer the pick in 9 pairs and the base in 6, 5 ties (p_loss 0.849), rule 3 passes; 4 hard faults on the pick and 3 on the base wait for your decision (rule 4, §7.1) |
| §8 The decision | ✅ 2026-11-06: you accept the hard faults of §7.1 (rule 4). The pick passes: Phase 3 on bot K 160 at top-k 5 (`PLAN_MODEL_SPECIALIZE.md` §7) |

---

## 1. Goal

`PLAN_MODEL_SPECIALIZE.md` §4.5 rule 5 asks for a final review of the pick before Phase 3: 100 judge pairs, and 20
pairs that you review yourself, blind. The pick is bot K 160 at top-k 5 (`tp1-botmask5-k160`). Step F gave 30 judge
pairs (13 wins, 4 losses, 13 ties). This plan adds the other pairs, sets a pass rule, and gives you the 20 pairs.

Phase 3 (the physical prune and a new draft) takes 2 to 3 days. The review costs one night of downtime and about
one day of judge time. It finds a quality loss before that work, not after it.

## 2. Design

### 2.1 The prompts

Use all 19 held-out bot prompts: the 15 of step F and the 4 research prompts that step F left out (`--research 8`).

- Do not use the calibration prompts. Phase 2 ranked the experts on them, so they favor the mask.
- Do not write new prompts by hand. They are not real bot traffic. The live bot had only 3 new Discord sessions
  since 2026-11-01, too few to add a new split.
- Gap, as in step F: no `technical-analysis` or `peer-comps` prompt.

### 2.2 The pairs

A pair is one prompt, one base run and one new run. The judge compares the two reports twice, with A and B swapped.

| Runs | Labels | Prompts |
|---|---|---|
| Base (K 216, top-k 6, main server) | `baseline-1`, `baseline-2` (2026-11-03), `baseline-3` (new) | 19 each |
| New (bot K 160, top-k 5) | `botmask5-k160` (2026-11-05), `botmask5-k160-r2` (new) | 19 each |

- **Test pairs:** 2 new runs × 3 base runs × 19 prompts = **114 pairs**. Step F already judged 30 of them.
- **Control pairs:** each base run against each other base run: 3 × 19 = **57 pairs**. The same model is on both
  sides, so the control shows the noise and the bias of the judge at the same scale.
- The pairs share runs, so they are not independent. Two new runs and three base runs spread the run-to-run
  variation, which is large for agent runs (`PLAN_E2E_RUNNER.md` §15.6: the same prompt gave `one_message` and
  `split`).
- Run `baseline-3` and `botmask5-k160-r2` within about one day of each other. Research prompts read live web data,
  and the judge must not see a date gap as a quality gap.

### 2.3 The pass rule

The judge does not gate in step F (§15.3), because 15 pairs are too few. With 114 pairs, a sign test can tell a loss
from the noise. Use a one-sided sign test on the pairs with a winner (ties out), at p < 0.05:

| Pairs with a winner | Fail at this many losses or more |
|---|---|
| 40 | 26 |
| 50 | 32 |
| 60 | 37 |
| 70 | 43 |
| 80 | 48 |

1. **Judge test:** the pick fails if its losses reach the limit. Step F had 17 pairs with a winner and 4 losses.
2. **Judge control:** the 57 control pairs must not fail the same test in either direction (base 1 against base 2,
   and so on). If a control fails, the judge has a bias that the test cannot separate. Then rule 3 decides alone.
3. **Your blind review (§7):** the pick fails if you prefer the base answer in 15 or more of 20 pairs with a winner
   (one-sided sign test, p < 0.05; with ties, use the row for the number of pairs with a winner).
4. **Hard faults:** a wrong number, a broken format, or a missing save that you mark in §7. You decide each case.
   Step F and the checks already cover these, so expect none.

The pick passes if rules 1 and 3 pass and you accept the hard faults of rule 4.

### 2.4 Two judge biases

- **Self-preference.** The judge is the main model (K 216). It can prefer its own output. This bias works against
  the pick, so it makes the test stricter, not weaker. The control pairs do not measure it.
- **The data date** (§15.9). Each run fetches the live data on its own day. The judge reads one request, so it marks
  an answer from another day as wrong. §3 fixes this for the daily report.

## 3. Judge changes (one pull request)

Branch `final-review-judge`, base `e2e-check-fixes`. Estimate: 150 lines with tests, below the 400-line limit.

1. **Each answer with its own request.** When the two requests differ (the daily report has its own digest in each
   run), the judge shows both requests and tells it to grade each answer against its own request. Same requests: no
   change.
2. **Resume.** Write each row to the output file when the row is done. Skip the rows that are in the file. The
   judge runs for hours, and a stop now loses all rows.
3. **The sign test.** `summary()` adds `decided` (wins + losses) and `p_loss` (the one-sided sign-test p of the
   losses). It still does not gate.
4. **A sample for the page.** `--page FILE --sample 20 --seed N` writes 20 pairs: one pair per prompt for 19
   prompts, plus one more, each with a random base run and new run. The order and the A/B side are random.
   `FILE.key.json` holds the key.
5. **Hide the run name.** A skill prompt holds its run folder (`[Skill directory: ~/.hermes-e2e/botmask5-k160/…]`).
   `reports()` replaces it with `~/.hermes-e2e/run`. Before this change, the judge and the blind page saw the folder
   name of the base run in skill prompts (never the name of the new run). Also, change 1 saw two different requests
   in every skill prompt.

Then judge the 2 daily-report pairs of step F again with change 1. All other rows of step F stay. Test on the real
runs (page mode, no model calls). The expected result: 20 pairs from 15 prompts; only the daily-report pair shows
two requests; no run name on the page.

## 4. More baseline runs (main server, no downtime)

The runs are clients of the main server, as the control run of §15.6 was. The server runs `MAX_NUM_SEQS=1`, so a
bot message waits behind the current request.

| Run | Command | Time (estimate) |
|---|---|---|
| The 4 extra research prompts in `baseline-1` | `run --base-url http://127.0.0.1:8888/v1 --label baseline-1 --research 8` | 1 h |
| The same in `baseline-2` | `… --label baseline-2 --research 8` | 1 h |
| A new full run | `… --label baseline-3 --research 8` | 3.5 h |

- A rerun in a folder skips the prompts that have a `done.json`, so only the 4 new prompts run in `baseline-1` and
  `baseline-2`.
- Run in the afternoon and evening of the day of §5, for example 16:00 to 21:30. Do not run from 05:00 to 10:00 on
  weekdays: the live `daily-report` and `daily-followups` cron jobs run then.
- Check each run with `model_eval.e2e.check`. A base run that fails a gate stays in, because it is the base.

## 5. More top-k 5 runs (one night of downtime)

**Caution:** the main server is down for about 4 h 40 min. Start after 22:30. Let the run stop at 03:40, so that the
main server is back before the guard (04:15) and before the 05:30 daily report.

1. Copy `~/.hermes-e2e/stepf5.sh` to `final5.sh`. Change the run step to two calls:
   - `run --label botmask5-k160 --research 8` (the 4 extra research prompts, about 1 h);
   - `run --label botmask5-k160-r2 --research 8` (a new full run, about 3.5 h).
2. Keep the safety rules of `PLAN_OVERNIGHT_MASK_GRID.md` §2: one model in memory, 100 GiB available before a boot,
   the memory watch, `restart: on-failure:1`, the guard in its own tmux session.
3. Tell the other Claude session (`stock-research-b1`) the server times before the start.
4. If the run stops at 03:40 before the end, start the same script the next night. A rerun skips the done prompts.
5. After the restore, run `check` on both labels and `check --compare` against `baseline-1` and `baseline-2`.

Close VS Code Server before the start if you can. In step F it took about 1.8 GB, and the memory watch stopped one
prompt.

## 6. The judge on 114 pairs and 57 control pairs

The judge runs on the main server with no downtime. It takes about 5 minutes per pair (step F: 30 pairs in 151
minutes).

| Call | New pairs | Time (estimate) |
|---|---|---|
| `judge baseline-1 baseline-2 baseline-3 botmask5-k160` | 27 (4 research × 2 old bases, 19 on `baseline-3`) + 2 again (daily report) | 2.5 h |
| `judge baseline-1 baseline-2 baseline-3 botmask5-k160-r2` | 57 | 4.8 h |
| `judge baseline-1 baseline-2` | 19 | 1.6 h |
| `judge baseline-1 baseline-3` | 19 | 1.6 h |
| `judge baseline-2 baseline-3` | 19 | 1.6 h |

Before the first call, seed its output file (`judge-baseline-1+baseline-2+baseline-3-vs-botmask5-k160.jsonl`) with
the 28 rows of step F that are not the daily report. Then the resume skips them. Rename the old
`judge-baseline-1-vs-baseline-2.jsonl` (the noise check of §14.5, 15 pairs) first. Then the control judges all 19
pairs with the same judge.

The judge takes about 12 h in total. Run it over the day after §5 and the next night. With the resume change of §3,
a stop costs only the current pair.

Report: the summary per call, the sign test of all 114 test pairs and of each control, the pairs that lose against
all three base runs, and the verdicts per skill.

### 6.1 Result (2026-11-06)

The judge ran from 03:17 to 16:25 on the main server, with no errors and no restart. The new code
(`final-review-judge`, 855a1b5) judged all pairs. The 28 seeded step F rows stay. The judge judged the 2 daily-report
pairs of step F again.

| Pairs | Wins | Losses | Ties | Decided | p_loss | p_win |
|---|---|---|---|---|---|---|
| `botmask5-k160` against the 3 bases | 24 | 6 | 27 | 30 | 0.9998 | 0.0007 |
| `botmask5-k160-r2` against the 3 bases | 20 | 13 | 24 | 33 | 0.9186 | 0.1481 |
| **All 114 test pairs** | **44** | **19** | **51** | **63** | **0.9996** | 0.0011 |
| Control `baseline-1` vs `baseline-2` (win = `baseline-2`) | 3 | 6 | 10 | 9 | 0.2539 | 0.9102 |
| Control `baseline-1` vs `baseline-3` (win = `baseline-3`) | 5 | 8 | 6 | 13 | 0.2905 | 0.8666 |
| Control `baseline-2` vs `baseline-3` (win = `baseline-3`) | 5 | 5 | 9 | 10 | 0.623 | 0.623 |

- **Rule 1 passes.** The pick has 19 losses in 63 decided pairs; the fail line is about 38. No prompt loses against
  all three bases in either new run.
- **Rule 2 passes.** No control fails the sign test in either direction (all p ≥ 0.25).
- **The data date.** Against `baseline-3` (run on 2026-11-05, the day of the new runs) the pick is even: 10 W / 9 L /
  19 T. Against `baseline-1` and `baseline-2` (2026-11-03 for 15 of 19 prompts) it has 34 W / 10 L. So most of the
  wins come from newer data, not from a better model. Read the result as "no loss", not as "a gain".
- **Weak prompts.** `17bd3e34` (research) has 0 W / 3 L / 3 T over the 6 pairs, and it is the prompt of the one
  tool-call fault of §5 (`terminate_placeholder`). `3c708852` (stock-update) has 2 W / 3 L / 1 T. Look at both in §7.
- **Per skill.** No skill has more losses than wins, except stock-update (2 W / 3 L, 5 decided pairs).
- **The judge.** In about half the calls, the thought reached the token limit. Then the verdict came from the second
  call with the notes (`after_notes`), as in step F.

Files: `model_eval/results/e2e/judge-baseline-1+baseline-2+baseline-3-vs-botmask5-k160{,-r2}.jsonl` and
`judge-baseline-{1-vs-baseline-2,1-vs-baseline-3,2-vs-baseline-3}.jsonl`. The old 15-pair control is
`judge-baseline-1-vs-baseline-2-noise15.jsonl`.

## 7. The blind review of 20 pairs (you)

1. Run `judge baseline-1 baseline-2 baseline-3 botmask5-k160 botmask5-k160-r2 --page review.html --sample 20
   --seed 7`. The page shows the request and the two answers of each pair as A and B, with no run names.
2. For each pair, write A, B or tie, and mark any hard fault (rule 4 of §2.3). Do not read the judge rows first.
3. Give me the 20 answers. I map them with `review.key.json` and apply rule 3.

This takes about 2 hours: the answers are long research reports.

### 7.1 Result (2026-11-06)

You gave the 20 answers on the review page (the private artifact, the same pairs and sides as `review.html`). I mapped
them with `review.key.json`.

| Pair | Your verdict | Run | Base | Skill | Hard fault | Judge |
|---|---|---|---|---|---|---|
| 1 | tie | `botmask5-k160` | `baseline-3` | earnings | | tie |
| 2 | base | `botmask5-k160` | `baseline-3` | subagent (`54cc02`) | pick: FY26 adjusted FCF "+182%" is wrong | base |
| 3 | pick | `botmask5-k160` | `baseline-2` | research | base: scenario math | pick |
| 4 | pick | `botmask5-k160` | `baseline-3` | research | | pick |
| 5 | tie | `botmask5-k160` | `baseline-1` | daily report | | pick |
| 6 | base | `botmask5-k160` | `baseline-1` | other (`ccd5f1fb`) | pick: a `</s>` token in a broken sentence | tie |
| 7 | tie | `botmask5-k160` | `baseline-3` | quant | | tie |
| 8 | pick | `botmask5-k160` | `baseline-1` | filings | base: no accession numbers | pick |
| 9 | tie | `botmask5-k160` | `baseline-3` | other | | pick |
| 10 | pick | `botmask5-k160-r2` | `baseline-1` | subagent | | pick |
| 11 | pick | `botmask5-k160-r2` | `baseline-2` | filings | base: no accession numbers | tie |
| 12 | pick | `botmask5-k160` | `baseline-1` | research (`669cff0b`) | pick: broken format (you still prefer it) | pick |
| 13 | base | `botmask5-k160` | `baseline-3` | subagent | | tie |
| 14 | base | `botmask5-k160-r2` | `baseline-2` | research | | pick |
| 15 | pick | `botmask5-k160-r2` | `baseline-3` | watchlist | | tie |
| 16 | tie | `botmask5-k160-r2` | `baseline-1` | research (`17bd3e34`) | | base |
| 17 | base | `botmask5-k160-r2` | `baseline-1` | research (`fd10d6b7`) | pick: Maverick revenue is wrong | tie |
| 18 | pick | `botmask5-k160-r2` | `baseline-3` | research | | pick |
| 19 | base | `botmask5-k160` | `baseline-1` | research | | tie |
| 20 | pick | `botmask5-k160-r2` | `baseline-2` | stock update (`3c708852`) | | base |

- **Rule 3 passes.** You prefer the base in 6 of 15 decided pairs (p_loss 0.849). With 15 decided pairs, the fail
  line is 12. You prefer the pick in 9 (p_win 0.304), so the review also shows "no loss", not "a gain".
- **The judge and you.** You agree on 9 of 20 pairs. You disagree in the opposite direction on 2 pairs (14 and 20).
  On the other 9, one of you gives a tie. The judge is a fair proxy for "no loss", but not for a single pair.
- **The weak prompts of §6.1.** On `17bd3e34` (pair 16) you give a tie, so the `terminate_placeholder` fault did not
  harm the answer. On `3c708852` (pair 20) you prefer the pick.
- **Rule 4: 4 hard faults on the pick, 3 on the base.** The base faults are a scenario math error (pair 3) and no
  accession numbers (pairs 8 and 11). The pick faults are:
  1. Pair 2: a wrong number (FY26 adjusted FCF growth "+182%"). The base runs also make wrong numbers (pair 3).
  2. Pair 6: the text breaks down in one sentence ("...not base-case winners)Skip probability is heavily in the
     tails </s>. All three are..."). A literal `</s>` occurs in 1 of the 57 mask answers (40 at top-k 5), and in 0
     of the 57 baseline answers. This is the only fault that looks like an effect of the mask.
  3. Pair 12: a broken format, but you still prefer the answer.
  4. Pair 17: a wrong number (Maverick revenue).
  5. Also from §5: the unknown tool call `terminate_placeholder` in `botmask5-k160-r2` on `17bd3e34`. The agent
     errored the call and went on, and you give that pair a tie.

The wrong numbers occur on both sides, so they do not separate the runs. The `</s>` leak (1 answer in the 40 top-k 5
answers) and the unknown tool call (1 in 38 new runs) are the two signs of the mask. You decide if you accept them.
If you accept them, the pick passes (§8: Phase 3 on bot K 160 at top-k 5).

## 8. The decision

| Result | Next |
|---|---|
| Rules 1 and 3 pass | Phase 3 on bot K 160 at top-k 5 (`PLAN_MODEL_SPECIALIZE.md` §7) |
| Rule 1 or rule 3 fails | By §4.5, the next point is bot K 160 at top-k 6, which is slower than the baseline (§15.9 of the runner plan). Then the speed goal needs the new draft of Phase 3 first. You decide |
| A control fails (rule 2) | Rule 3 decides alone. Look at the control rows for the bias |

**Decision, 2026-11-06:** rules 1, 2 and 3 pass, and you accept the hard faults of §7.1 (rule 4). The pick passes.
Next: Phase 3 on bot K 160 at top-k 5.

## 9. Effort and order

| Step | Time | Downtime |
|---|---|---|
| §3 Judge changes, review and merge | Half a day | None |
| §4 Baseline runs | 5.5 h | None |
| §5 Top-k 5 runs | 4 h 40 min | 4 h 40 min, one night |
| §6 Judge | About 12 h | None |
| §7 Your review | About 2 h | None |

Order: §3, then §4 and §5 on the same day and night, then §6, then §7. The total is about 3 days, mostly unattended.
