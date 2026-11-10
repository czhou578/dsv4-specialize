# How the tests chose the pruned model

Since 2026-11-06 20:35, the stock-research bot runs on a pruned copy of DeepSeek V4 Flash: 160 of 216 routed experts
per layer, 5 experts per token in place of 6. This page records why, how the tests chose that point, and what is
open. The work ran from 2026-11-02 to 2026-11-07. The plans in `plans/` hold the details. §6 records an extension
of 2026-11-09 and 2026-11-10: fewer experts per token on the pruned model.

This page moved here from the bot's `MODEL_PRUNE.md` on 2026-11-08. That page keeps what runs now, how to go back
to the stock model, and how the harness measures a model.

## 1. Why

`PLAN_MODEL_ROUTER.md` (bot repo) wants two models in memory at once: DeepSeek for the deep work and a fast model for
the rest. The DGX Spark has 121 GiB, and the stock DeepSeek loaded 95.39 GiB. The plan set a size cap of about
72 GiB for DeepSeek. `PLAN_MODEL_SPECIALIZE.md` (bot repo) is the plan that made the model smaller, with the rule "no
loss that you notice" and, if possible, a faster decode.

## 2. The grid

Two settings make a point: K, the experts per layer (size), and top-k, the experts per token (speed). A mask
checkpoint (`mask_checkpoint.py`, `topk_checkpoint.py`) tests a point without a rewrite of the weights. Each point ran
the replay screen first; only the points that passed ran the speed tests and the full runs.

| Rank | K | Top-k | Replay NLL change | Tool names | Bot / code decode | Result |
|---|---|---|---|---|---|---|
| — | 216 | 6 | 0 (0.4440) | 77 of 83 | 33.64 / 38.86 tok/s | The baseline |
| REAP (the public rank) | 192 | 6 | +3.31% | 78 of 83 | — | Fails the NLL gate (+3%) |
| REAP | 176 | 6 | +6.31% | — | — | Fails |
| Bot | 176 | 6 | +0.29% | 77 of 83 | — | Passes |
| Bot | 160 | 6 | +1.37% | 78 of 83 | 31.50 / 35.67 (−6.4% / −8.2%) | Passes, but slower: the draft model loses acceptance (2.25 → 2.09) |
| **Bot** | **160** | **5** | **+2.27%** | **79 of 83** | **33.98 / 42.07 (+1.0% / +8.3%)** | **The pick.** Full runs: 15 of 15 prompts, 4 of 5 gates (see below) |
| Bot | 160 | 4 | +6.05% | 79 of 83 | 36.05 / 42.78 | Fails the NLL gate |
| — | 216 | 5 | +1.29% (35 targets) | 49 of 54 | — | Top-k 5 costs the same on K 216 as on K 160 |

Findings:

- The public REAP rank drops experts that the bot uses. The bot rank at K 176 costs a tenth of the REAP rank at K 192.
- A smaller K saves memory and gives no speed. The DSpark draft model was built for K 216, so a mask alone is slower.
- Top-k 5 wins the speed back and gains 8% on code. Top-k 4 costs too much NLL.
- One gate fails on one prompt (`54cc02`, the report marker): the model writes the report in the reply and saves no
  file. The stock model does the same on 1 of 3 runs. It is a prompt problem, not a model problem.

The expert rank came from `capture.py`: the routed experts of 34 calibration segments (1.75 M tokens) of the bot's own
traffic, captured on a Model Runner V1 boot. The rank file is `results/expert_rank_bot.json`.

## 3. The final review (`plans/PLAN_FINAL_REVIEW.md`)

The pick ran twice on 19 prompts, against three baseline runs of the stock model.

| Test | Result | Rule |
|---|---|---|
| LLM judge, 114 pairs | 44 wins, 19 losses, 51 ties for the pick (p_loss 0.9996) | Passes: the pick does not lose |
| 3 baseline-against-baseline controls | 3 W / 6 L, 5 W / 8 L, 5 W / 5 L; none significant | Passes: the judge has no position bias |
| Blind review by the user, 20 pairs | 9 for the pick, 6 for the base, 5 ties | Passes |
| Hard faults | 2 wrong numbers on each side; on the pick also 1 broken format, 1 literal `</s>` in a reply, 1 call to an unknown tool | The user accepted them |

Read the result as "no loss", not as a gain: against the baseline run of the same day, the judge gives 10 wins and
9 losses. Most of the wins against the older baselines come from newer market data.

## 4. The physical prune (`plans/PLAN_PHYSICAL_PRUNE.md`)

A mask hides the dropped experts from the router but still loads them. `prune_experts.py` removes them:

- Each expert is 12 separate EXL3 tensors in one file per layer. The tool copies the bytes of the 160 kept experts
  and renumbers them. The tool does no new quantization.
- The router rows, the router bias and the hash tables of layers 0 to 2 follow the same new order.
- `config.json`, the weight index, the tier bitmap, the manifest and a new `REAP_K160_PLAN.json` get the new count.
- The MTP layer and the DSpark draft do not change: the target model skips the MTP tensors, and the draft is its own
  folder.

The build ran in a root container (the expert files are `root:600`), 11 minutes, with the bot up. It removed
28,896 tensors (21.27 GiB). The self-check compared every kept tensor with its source byte for byte. The test boot
matched the mask: replay NLL 0.4524 against 0.4541 (the difference has the size of the run noise), speed inside the
noise, 74.09 GiB loaded.

## 5. Open items

- **The size cap.** 74.09 GiB is 2 GiB over the 72 GiB of the router plan. The user accepted it on 2026-11-06.
  The second model gets about 2 GiB less. For two models, DeepSeek must boot at about `GPU_MEMORY_UTILIZATION` 0.75
  (about 91 GiB in total). A test boot at 0.73 and 128k failed the KV check on 2026-11-06: the KV need is a fixed
  6.5 to 7 GiB, so a smaller context saves nothing. Since 21:10 the live server pins the KV at 8 GiB with
  `--kv-cache-memory-bytes`; it then uses about 94 GiB in all, and 23.7 GB stays available (`PLAN_MODEL_ROUTER.md`
  §3, bot repo).
- **The bot-rank draft model** is live since 2026-11-07 (`plans/PLAN_PHYSICAL_PRUNE.md` §7.1): acceptance +4.7% in
  the bot test.
- **K 144** (about 68 GiB) fails the NLL gate with every rank tried on 2026-11-07 (+3.29% to +3.90% against a +3%
  gate; `plans/PLAN_BETTER_RANK.md` §6). K 160 is the floor for this model without a fine-tune.
- **The daily report job** has not run since the rename of the served model. On 2026-11-09 at 05:30, Hermes skipped
  `daily-report` (job `89988a054346`): "global inference config drifted since this job was created (model
  'deepseek-v4-flash-0731' -> 'deepseek-v4-flash-0731-k160t5'), and this job is unpinned". Pin the job to the new
  model name. `daily-followups` is not affected.

## 6. Extension: fewer experts per token (`plans/PLAN_FEWER_EXPERTS.md`)

The prune made the model smaller, not faster: K 160 at top-k 5 decodes at about the speed of the stock model. This
extension asked if the model can use fewer experts per token. A verify step gets about 9% faster for each expert it
drops, but a plain cut from 5 to 4 experts cost +3.69% NLL on top of top-k 5 (`plans/PLAN_OVERNIGHT_TOPK.md`).

**The tool.** A router patch (`image-patch/moe_drop.py` and `router-drop.diff`) boots the model at top-k 6 and drops
slots by a rule. A dropped slot gets the expert id -1, and the EXL3 route pack reads no weights for it. The patch can
also divide the kept weights by the mass of the top 6 experts in place of their own sum: gain anchoring (Chen and
Yao 2026, https://arxiv.org/abs/2609.04575). `compose-drop.yml` mounts it on a test boot, and `fewer_night.sh` and
`fewer_stepf.sh` run the tests.

**The tests** ran from 2026-11-09 20:20 to 2026-11-10 07:25, with two short restores of the live server:

| Phase | Result |
|---|---|
| A: the routing profile (§5.1 of the plan) | The weights inside the top 6 are flat: the 5th and 6th experts hold 12.4% and 11.3% of a token's mixture |
| C: the replay (§7.1) | Every cut below 5 experts fails: +6.4% to +21.8% NLL against a +3% gate. At top-k 4, the anchor makes the loss larger. The anchored top-5 (C2) is the best point: +1.17% against the baseline, 0.71 points better than the live folder (95% interval −1.37 to +0.02) |
| D: the speed (§8.1) | No point is faster than the live server. C2 runs within the noise of the live speed (−4.0%; the noise is 4.2%). The anchor on the live folder alone (C6) gains nothing clear |
| E: the full runs (§9.1) | C2 passes all 5 gates against both baselines. The judge finds no gain: 18 wins, 20 losses and 16 ties in 54 pairs (the live pick in the same setup: 44 wins, 19 losses) |

**Result.** This model cannot use fewer than 5 experts per token without training: more speed must come from
elsewhere (a longer draft, or the second, fast model of `PLAN_MODEL_ROUTER.md` in the bot repo). The anchored top-5
is a little better on the replay (0.71 points of NLL), but the judge does not see a gain in the full runs. A deploy
would need a top-k 6 prune of the mask and the patch on the live server.
On 2026-11-10 you decided: no deploy (`plans/PLAN_FEWER_EXPERTS.md` §9.2). The live model did not change, and the
test folders were deleted.
