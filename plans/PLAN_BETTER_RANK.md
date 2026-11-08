# Plan: A better expert rank for the next prune

**Status:** 2026-11-07 03:04: done. K 144 fails the NLL gate with every rank (+3.29% to +3.90%). The new rank equals
the live pick at K 160. Decision: stop; K 160 with the count rank stays (§6.3). The plan does not reach §7.
**Author:** colin-spark · **Date:** 2026-11-06
**Related:** `PLAN_MODEL_SPECIALIZE.md` §6 (Phase 2: the count rank, this plan replaces it), §4.5 (the grid and the
pick rule); `PLAN_PHYSICAL_PRUNE.md` (the prune tool and the deploy, reused as they are); `MODEL_PRUNE.md` (what runs)
**Scope:** `model_eval/capture.py` and a new capture patch in `~/Projects/dsv4-specialize/`; new recorded traffic;
new mask folders. The live server (`compose-k160t5.yml`) changes only if §7 reaches a deploy.

| Step | Status |
|---|---|
| §3 Data: new traffic, no leak into the held-out set | ✅ 2026-11-06 22:23–23:22: the 11 new prompts ran on the live server (`rank-1`, all exit 0). `rank_data.py` merged 66 sessions, 63 segments, 2.84 M prompt tokens (§3.3) |
| §4 Capture: the router weights, with the ids | ✅ 2026-11-06 23:24 to 2026-11-07 00:14 (two boots; the first failed under CUDA graph capture, §4.3): 63 segments, 0 errors, 34 min of capture. 198 of 9,288 expert slots went NaN at row 26; `capture.py` estimates them from their finite rows (§4.3) |
| §5 The scores and the new rank file | ✅ 2026-11-07 00:15: four rank files; the weight order nearly equals the count order (overlap 0.92); the floor keeps 5 to 6 experts per layer (§5.1). The new rank for §6 is `skill_floor` |
| §6 Test: masks at K 160 and K 144, the replay screen, the decision | ✅ 2026-11-07 00:17–03:04: four masks (§6.1, §6.2): floor K 160 +1.91% (pass, equal to the live pick), floor K 144 +3.29%, count K 144 +3.51%, 40 k floor K 144 +3.90% (all fail the NLL gate). Decision: stop (§6.3) |
| §7 If a point passes: the full runs, the review, the prune, the deploy | ⏭ Not reached: no point passes |

---

## 1. Goal

The current rank made K 160 possible. Below it, the cost rises fast (`PLAN_MODEL_SPECIALIZE.md` §6, results):

| Step | Routing share removed | NLL cost at top-k 6 | Cost per point of share |
|---|---|---|---|
| K 216 to 176 | 2.47% | +0.29% | 0.12 |
| K 176 to 160 | 2.13% | +1.08% | 0.51 |
| K 160 to 144 | 2.86% | not measured; +1.5% at best, +3% or more if the trend holds | — |

A better rank orders the experts by what they contribute, not by how often the router picked them. The goal is a
lower cost per dropped expert, so that either:

- **K 144 passes** the replay screen at top-k 5 (NLL at most +3%, tool names at least 98% of the baseline) with a cost
  near the cost of K 160 today (+2.27%). That frees about 6 GiB more for a second model (`PLAN_MODEL_ROUTER.md` §3).
- **Or K 160 gets better** at the same size. A lower NLL at K 160 is a free quality gain for the live model. It also
  lowers the risk of the faults that the review found (`PLAN_FINAL_REVIEW.md` §7.1).

If neither happens, K 160 is the floor for this model without a fine-tune, and the plan stops.

## 2. What the current rank lacks

The current rank (`expert_rank_bot.json`) counts how often each expert was in the top 6 of a token, on 34 calibration
segments (2.0 M prompt tokens). It has three gaps:

1. **It counts use, not importance.** The router weight and the size of the expert output do not enter the count. A
   rare expert that writes tool names counts the same as a rare expert that adds little.
2. **It saw too little of the bot.** The 34 segments have 1 quant session, no subagent session, no daily report and no
   technical analysis (`PLAN_MODEL_SPECIALIZE.md` §6.1). The two weak prompts of the review are a research prompt and
   the stock-update prompt, and the one unknown tool call came from a subagent-style run.
3. **It knows nothing about substitutes.** If two experts have near-equal outputs, it is safe to drop one of them. The
   first step was cheap because the router found substitutes by chance.

## 3. Data

### 3.1 The leak rule

The replay screen scores 60 held-out targets. No held-out session and no transcript of a held-out prompt may enter
the capture. **All 17 recorded e2e prompts are held-out sessions**, so the run folders of those prompts in
`~/.hermes-e2e/*/runs/` stay out. The two synthetic prompts, `daily-report` and `quant-drift`, are not held out, so
their runs go in. The capture tool checks each source against `split.json` and refuses a held-out id.

### 3.2 Sources

| Source | What it gives | Size (estimate) |
|---|---|---|
| The 34 calibration segments of the first capture | The base: research, stock-update, earnings, filings, watchlist, peer-comps, 1 quant | 2.0 M tokens |
| The cron sessions in `~/.hermes/state.db` since 2026-11-02 | 4 daily reports, 2 daily-followups runs (`earnings`) | about 0.2 M tokens |
| The e2e runs of `daily-report` and `quant-drift` (about 12 each, across the labels) | The daily report and the quant skill, as the bot runs them | about 0.4 M tokens |
| **New runs on new tickers** through the e2e runner, on the live server, no downtime: 4 subagent research prompts, 3 technical-analysis prompts, 2 quant prompts, 2 peer-comps prompts | The skills that have no or one calibration session | about 0.5 M tokens |

The sources give about 3 M tokens in all. The capture runs at about 1,370 tokens per second (the first capture:
2.0 M tokens in 1,474 s), so it takes about 40 minutes plus the boot.

Write the new prompts into `model_eval/e2e/prompts.py` under a `rank` set, with tickers that no recorded session
uses. Export the cron sessions with `export_sessions.py`, with the cron source allowed and the split fixed to
`calibration`. Convert an e2e run folder to a capture request from its `events.jsonl` (the system prompt, the tools,
the messages up to the last model turn).

The general data of `PLAN_MODEL_SPECIALIZE.md` §6.1 stays out. The bot's own traffic has the tool protocol in it,
and the public observation set has no text and no license.

### 3.3 Result (2026-11-06)

| Source | Sessions | Skills |
|---|---|---|
| The calibration split of the first capture | 37 | research 17, other 11, stock-update 4, earnings, filings, peer-comps, quant, watchlist 1 each |
| The sandbox databases of 7 labels (`baseline-1/2/3`, `botmask-k160`, `botmask5-k160`, `-r2`, `smoke-daily`) | 14 | daily-report 8, quant 6 |
| The live cron jobs since 2026-11-02 | 5 | daily-report 3, earnings (followups) 2 |
| `rank-1`: the 11 new prompts, run on the live (pruned) model, 22:23 to 23:22 | 11 | research (the 4 subagent texts) 4, technical-analysis 3, quant 2, peer-comps 2 |

66 sessions, 63 segments with a target, 2.84 M prompt tokens. Computed tokens per skill in the capture: research
1.55 M, earnings 175 k, daily-report 142 k, stock-update 69 k, quant 48 k, other 22 k, technical-analysis 21 k,
watchlist 18 k, peer-comps 12 k. Two notes:

- The exporter labels the four subagent texts `research`. They are research prompts that ran as a top-level turn.
- The `rank-1` transcripts come from the pruned model. The final review found that model equal to the stock model.

The work found a bug on the way: the e2e sandbox doubled the model suffix (`deepseek-v4-flash-0731-k160t5-k160t5`).
The cause: the live name is a prefix of the served name, and the replacement used a word boundary. The first 8 runs
failed in 10 s each with exit 0. PR #63 fixed the bug in `sandbox.py`.

## 4. Capture: the router weights, with the ids

The built-in capture (`--enable-return-routed-experts`) returns the expert ids per token and layer. It does not
return the weights. The router computes both at the same place: `fused_topk_bias(...)` in
`image-patch/vllm/models/deepseek_v4/nvidia/model.py` (`DeepseekV4MoE.forward`) returns `topk_weights, topk_ids`
and passes them to `self.experts(...)`.

1. Find the class in the image's vLLM that stores `topk_ids` for the reply (`grep -rn enable_return_routed_experts
   /opt/vllm`). Extend it to store `topk_weights` as a float16 array of the same shape (tokens, layers, top-k).
   Return the weights in the reply as `routed_weights` next to `routed_experts`.
2. Mount the patched file into the capture container only (`compose-capture.yml`, like the other patches). The live
   server does not change.
3. `capture.py`: read `routed_weights` when present. Per segment, write three per-layer arrays: the counts, the sum of
   weights, and the sum of squared weights. A rerun still skips finished segments.
4. On the first request, confirm that the weights sum to about 1 per token after the `routed_scaling_factor` (1.5).
   Confirm also that the ids match the built-in capture.

**§4.1 How the patch works (2026-11-06).** The model runner binds `_capture_fn(topk_ids, _layer_id=…, _capturer=…)`
to each router. The patched `base_router.py` reads the layer and the capturer from that function's defaults and
calls `capturer.capture_weights(layer, topk_ids, topk_weights)`. The patched `routed_experts_capturer.py` keeps
`sum_count`, `sum_weight` (layers × experts, float64) and `sum_tokens` on the GPU. It adds the choices of each layer
with `index_add_`. After the last layer of a step, it writes `routing_sums.npz` (an atomic rename) to `/capture`.
`compose-capture.yml` mounts `/capture` from `dsv4-specialize/capture-out/`. The model runner does not change. The
ids still come in the reply, so the same run reproduces the count rank of the first capture as a check.

**Optional, §4.2: the output sizes.** The full REAP score is the mean of weight × ‖expert output‖. The fused kernel
does not expose the output of one expert. `PLAN_MODEL_SPECIALIZE.md` §6.2 step 3 describes the offline way: save the
MoE input for 50 k tokens, dequantize the experts near the cut with ExLlamaV3, apply them. It costs about 17 GB of
saved inputs and a day of work. Do it only if the weighted rank of §5 does not reach the goal of §1. The weight is
the larger part of the score; the output norms of the experts of one layer are close to each other.

### 4.3 Result (2026-11-06 to 2026-11-07)

- **Boot 1 (23:24) failed:** `ids = ids[valid]`, a boolean index, inside the forward. vLLM captures CUDA graphs at
  boot, and a boolean index needs a sync: "operation not permitted when stream is capturing". The script restored the
  live server at 23:33. Fix: static shapes. An id outside 0 to 215 goes to a spare column with a zero weight. The
  flush skips while a stream capture runs, and `clear_buffer` (the step start) flushes instead.
- **Boot 2 (23:33) worked:** healthy after 225 s; the smoke request moved the sums file, 1.5 of weight per token
  (the `routed_scaling_factor`). The capture ran 23:38 to 00:12: 63 segments, 0 errors, about 1,400 tokens per second.
  The restore was healthy at 00:14. Downtime: 50 min in all for the two boots.
- **NaN weights.** 198 of 9,288 expert slots turned NaN at row 26, all at once, and a NaN sum stays NaN. The source is
  most likely a padded row of a graph-replayed step. `capture.py --rank` estimates such a slot from its finite rows
  (37% to 62% of its counts) scaled by the count ratio, and reports `nan_slots`. The capturer now zeroes a NaN weight.
- Memory: about 2.7 GB available during the capture (the stock model at 0.93). Claude Code's own memory check stopped
  a background wait; that stop did not affect the run itself.

## 5. The scores and the new rank file

`expert_rank_bot.json` keeps its format (per layer, every id with its fields, most important first) and gets new
fields per expert. `mask_checkpoint.py --rank` reads the order only, so it needs no change.

| Score | Definition | Use |
|---|---|---|
| `count` | Times in the top 6 | The current rank, kept for comparison |
| `weight` | Sum of the router weights | **The new rank order** |
| `weight_share` | `weight` over the layer total | The share tables of §1 |
| `per_skill` | `weight` per skill | The floor below |

Two rules on top of the order:

- **A floor per skill.** An expert drops only if it is below the cut for every skill. The share of a skill comes from
  the tokens of that skill alone. This protects the experts of a rare skill (quant, subagent) that the global
  sum hides. Record for each layer how many experts the floor kept.
- **The mandatory tool experts** (32 per REAP tool category) stay, as today. The tool-name metric of the replay
  screen tests this rule; if the weighted rank keeps those experts on its own, the guard changes nothing.

**Optional, substitutes.** The cosine of two `gate.weight` rows says how alike the tokens are that the router sends to
two experts (`mask_checkpoint.py` already uses it to remap the hash layers). A variant rank multiplies the weight by
(1 − the largest cosine to any kept expert), so that an expert with a close kept twin drops first. Test it as a
third mask only if the two ranks above leave K 144 short by less than a point.

Report, as in `PLAN_MODEL_SPECIALIZE.md` §6.3: the share that each step removes under the new rank, the overlap of
the lowest 56 experts between the count rank and the weighted rank, and the experts that the floor kept.

### 5.1 Result (2026-11-07 00:15)

Four orders from the same capture (`model_eval/results/expert_rank_{count,weight,skill_max,skill_floor}.json`;
`model_eval/rank_report.py`):

| Step | Weight share removed: count order | weight order | skill_floor order |
|---|---|---|---|
| K 216 to 176 | 2.23% | 2.12% | 2.27% |
| K 176 to 160 | 1.89% | 1.83% | 1.90% |
| K 160 to 144 | 2.56% | 2.51% | 2.56% |
| K 144 to 128 | 3.33% | 3.22% | 3.30% |

- **The weight order nearly equals the count order:** the dropped sets overlap 0.92 at K 160 and 0.93 at K 144, and
  the removed shares differ by about 0.1 points. On this traffic, how often the router picks an expert and how much
  weight it gives it rank the experts almost the same way. The weighting alone does not move the cost curve.
- **`skill_max` is wrong as a global order** (first version of §5). The share of an expert within one small skill is
  always above its global share, so skill specialists outrank general experts. The first step removed 15.6% of the
  routing. **`skill_floor`** replaces it: the best rank of the expert by weight over the whole data and over each skill
  with at least 50 k computed tokens (daily-report, earnings, research, stock-update), ties by weight. An expert then
  drops only if it is below the cut in every one of those skills.
- **The floor keeps 5.2 experts per layer at K 160 and 5.8 at K 144** (max 13) that the weight order would drop.
  That is under the "about 8" line of §10. Quant (48 k tokens) sits just under the token floor, so it sets no floor.
- The new rank for §6 is `skill_floor`; the control is the count order of the same capture, so the test isolates the
  broader data and the floor from the scoring.

## 6. Test: masks, the replay screen, the decision

Build the masks from the stock folder `data/tp1` (216 experts) with `mask_checkpoint.py --rank <new file>`, then
`topk_checkpoint.py --top-k 5`:

| Mask | Question |
|---|---|
| New rank, K 160, top-k 5 | Does the new rank beat +2.27% at the same size? |
| New rank, K 144, top-k 5 | Does K 144 pass, and at what cost? |
| Count rank, K 144, top-k 5 | The control: what the old rank costs at K 144 |

Run one test night, as in `PLAN_OVERNIGHT_TOPK.md`: each mask boots on port 8890 (`compose-mask.yml`, `MASK_NAME`),
runs the replay screen (35 min), and stops. Three masks take about 2 h 15 min of downtime; the run script restores
the live server. Then compare each result with `baseline-k216.jsonl` and `botmask5-k160.jsonl`.

| Result | Decision |
|---|---|
| New K 144 passes both gates with NLL at most +2.5% | Go to §7 with K 144 |
| New K 144 passes, but costs more than +2.5% | Your call: the 6 GiB against the quality. The full runs and the review of §7 decide |
| New K 144 fails, new K 160 beats +2.27% | Go to §7 with K 160: a better live model at the same size |
| Neither | Stop. K 160 with the count rank stays. The memory for a second model comes from the other side (`PLAN_MODEL_ROUTER.md` §3) |

### 6.1 Result (2026-11-07, 00:17 to 02:15)

`rank_night.sh`: each mask booted in 225 s and ran the 60 replay targets in about 35 minutes. The live server was
restored at 02:15 ("OK.", gateway active). Log: `~/.hermes-e2e/rank-night.log`.

| Mask, top-k 5 | NLL | Against the baseline (0.4440) | Tool names | Gates |
|---|---|---|---|---|
| Live pick: count rank of the first capture, K 160 (mask / prune) | 0.4541 / 0.4524 | +2.27% / +1.89% | 79 of 83 | pass |
| **Floor rank, K 160** (`wmask5-k160`) | 0.4525 | +1.91% | 79 of 83 | pass |
| **Floor rank, K 144** (`wmask5-k144`) | 0.4586 | +3.29% | 80 of 83 | **NLL fails** (+3%) |
| Count rank of the same capture, K 144 (`botmask5-k144`, the control) | 0.4596 | +3.51% | 80 of 83 | **NLL fails** |

- **At K 160 the new rank equals the live pick.** The sum of the NLL over the 60 targets differs by +0.03% from the
  pruned pick (and −0.35% from its mask run), inside the run noise. Per skill, it moves the cost between skills:
  filings +5.1% → +3.0%, quant +4.5% → +3.0%, watchlist +4.0% → +2.7%, while earnings and other get a little worse.
  The gain is not worth a new prune at the same size.
- **K 144 fails the gate with either rank.** The step from K 160 to K 144 costs 1.35 points for 2.56% of the routing,
  about 0.53 per share point, the same rate as the step before. The curve did not steepen, and it did not flatten.
- **The data and the floor are worth 0.22 points at K 144** (control +3.51% against +3.29%). The tool names improve
  to 80 of 83 (the best of any point). But the floor did not reach the skill that pays most: quant (+6.1% at K 144)
  sits at 48 k tokens, under the 50 k floor. Stock-update pays +5.3%.
- **Decision by the table of §6:** neither outcome. K 160 with the count rank stays the live model, and the memory
  for a second model comes from the other side (`PLAN_MODEL_ROUTER.md` §3). One cheap retest ran before that
  decision became final: the floor at 40 k tokens, which brings quant in (§6.2).

### 6.2 The fourth mask: the floor at 40 k tokens (2026-11-07, 02:15 to 03:04)

Quant pays most at K 144 and sat just under the 50 k-token floor, so one more mask tested the floor at 40 k tokens,
which brings quant in (`expert_rank_skill_floor40.json`, `tp1-wmask5q-k144`, `rank_night2.sh`). The first replay
stopped at 12 of 60 targets: the memory watch saw 476 MB free after a spike in the test server (the stock model at
0.93 leaves about 2.2 GB; two earlier replays stopped the same way). The rerun resumed the file and finished.

| Mask, top-k 5 | NLL | Against the baseline | Tool names | Gates |
|---|---|---|---|---|
| 40 k floor, K 144 (`wmask5q-k144`) | 0.4613 | +3.90% | 79 of 83 | **NLL fails** |

| Skill | Floor 50 k, K 144 | Count, K 144 | Floor 40 k, K 144 |
|---|---|---|---|
| quant | +6.1% | +6.4% | +5.7% |
| stock-update | +5.3% | +6.6% | +6.7% |
| watchlist | +4.2% | +6.4% | +8.9% |
| other | +3.7% | +2.8% | +4.1% |
| research | +3.4% | +3.4% | +3.5% |
| filings | +3.5% | +3.3% | +3.3% |
| earnings | +1.1% | +1.0% | +0.8% |
| subagent | +1.2% | +0.3% | +1.7% |

At a fixed K the floor is a zero-sum trade: the experts it keeps for quant push out experts that watchlist and
stock-update need. The 50 k floor is the best of the three K 144 masks, and it still fails.

### 6.3 Decision (2026-11-07)

**Neither outcome of the table.** K 144 fails the NLL gate with every rank (+3.29% at best against a +3% gate), and
at K 160 the new rank equals the live pick within the run noise. So:

- **K 160 with the count rank stays the live model.** No new prune, no new deploy.
- **K 160 is the floor for this model without a fine-tune.** The step to K 144 costs about 0.53 points of NLL per
  point of routing share with either rank, and a better rank moves that by 0.2 points at most.
- **The memory for a second model comes from the other side** (`PLAN_MODEL_ROUTER.md` §3): a smaller or more
  compressed second model, not a smaller DeepSeek. The room is about 12 GiB with a 10 GiB free margin, or 18 GiB
  with a 4 GiB margin.
- What the work leaves behind: a capture that records the router weights (`dsv4-specialize/image-patch/`), a rank
  data set with every skill (`model_eval/data/rank/`), four rank files, and the finding that on this traffic the
  router weight ranks the experts like the count does.

Downtime tonight: the two capture boots (23:24 to 00:14, 50 min), the first test night (00:17 to 02:15), the fourth
mask and its rerun (02:15 to 03:04); about 3 h 40 min in all, with the live server restored after each window.

## 7. If a point passes

Reuse the steps that K 160 went through, with the same tools and gates:

1. Step F: the full runs on the mask (`PLAN_E2E_RUNNER.md` §15), 19 prompts, 5 gates. About 2 h 40 min of downtime.
2. The final review (`PLAN_FINAL_REVIEW.md`): two runs of the mask, three baseline runs (the three of 2026-11-05 can
   serve again if the live data has not moved too far; otherwise one new baseline run), the judge, and your blind
   review of 20 pairs. The judge runs on the live server with no downtime.
3. `prune_experts.py --source <mask> --out tp1-prune5-k<K>` (`PLAN_PHYSICAL_PRUNE.md` §3 to §5), the test boot, and
   the deploy with a new compose file and a new served name, for example `deepseek-v4-flash-0731-k144t5`.

## 8. Safety

The rules of `PLAN_OVERNIGHT_MASK_GRID.md` §2 apply to every boot: one model in memory, at least 100 GiB available
before a boot, `restart: on-failure:1`, the memory watch, deadlines, and the restore as the safe action. The run
scripts and the guard already use `compose-k160t5.yml` for the live server. Run nothing that stops the server on
weekdays between 05:00 and 10:00, when the cron jobs run. Close VS Code Server before a test night: its extension
hosts took 3.5 GB each on 2026-11-06.

The capture boot runs the stock model on Model Runner V1 without the draft, from `compose-capture.yml`. If `data/tp1`
moves, the capture boot and the pruned folder (which links into it) break. Keep `data/tp1` in place.

## 9. Effort and downtime

| Step | Work | Downtime |
|---|---|---|
| §3 New runs through the e2e runner (11 prompts) | Half a day, mostly waiting | None |
| §3 Export and convert the other sources | Half a day | None |
| §4 The capture patch and `capture.py` | 1 day | None |
| §4 The capture run | 1 h | About 50 min, one window |
| §5 Scores, rank file, report | Half a day | None |
| §6 Three masks and the replay screen | One night | About 2 h 15 min |
| §7 Full runs, review, prune, deploy | As for K 160: about 4 days | About 4 h 40 min in two nights, plus 4 min for the deploy |

The §6 decision takes about a week, and a deploy takes about two weeks.

## 10. Open questions

1. **Where does the built-in capture keep the ids?** The patch of §4 depends on that class. One grep in the image
   answers it; if the ids come out of a C++ or Triton path with no Python hook, the fallback is a patch in
   `DeepseekV4MoE.forward` that writes `topk_weights` to a shared buffer per request.
2. **How much new Discord traffic will there be?** Real use is better than synthetic prompts. If you use the subagent
   and technical-analysis skills in the coming week, the capture takes those sessions in place of some new runs.
3. **Does the per-skill floor keep too much?** If it keeps more than about 8 experts per layer at K 144, the cut is
   not really K 144. The report of §5 shows the number before any boot.
