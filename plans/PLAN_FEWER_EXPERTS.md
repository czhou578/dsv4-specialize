# Plan: Fewer experts per token, with an anchored gain

**Status:** 2026-11-09: Phase B done (the router patch and its tests); the test night runs · **Author:** colin-spark · **Date:** 2026-11-09
**Related:** `PLAN_OVERNIGHT_TOPK.md` (the top-k night of 2026-11-04: top-k 4 fails), `PLAN_BETTER_RANK.md` (the
capture patch and the run scripts), `PLAN_PHYSICAL_PRUNE.md` (the live model and its deploy), `PLAN_MODEL_SPECIALIZE.md`
§8 in the bot repo (Phase 4, the speed axis), `PRUNE.md`
**Scope:** a patched router file and its tests in this repo; an extension of the capture patch; new mask and prune
folders; test boots on port 8890 (about 6 hours of downtime in all, in three windows). The deploy (§9) is your decision.

| Step | Status |
|---|---|
| §5 Phase A: the routing profile (capture, about 50 min downtime) | ⬜ |
| §6 Phase B: the router patch and its tests (no downtime) | ✅ 2026-11-09 (§6.1) |
| §7 Phase C: the replay night (about 4 h downtime) | ⬜ |
| §8 Phase D: the speed tests (about 1 h downtime) | ⬜ |
| §9 Phase E and F: the full runs, the review and the deploy (your decision) | ⬜ |

---

## 1. Short answer

A plain cut from 5 to 4 experts per token fails, and the data shows why. A cheaper fix exists: keep the gain of the
expert branch at its trained value. The router now divides the kept weights by their own sum, so each cut makes the
kept experts stronger. A 2026 paper fixes this with one integer and no training: it divides by the mass of the top
6 experts (the trained top-k), not by the mass of the kept experts. The paper reports that this recovers most of the
loss of a top-k cut on fine-grained MoE models.

The plan tests that idea on the bot's replay set, then a confidence threshold, then per-layer cuts. The runtime
already skips a slot whose expert id is -1, so one patched Python file does all of it, with no kernel change.

The speed ceiling is modest. A verify step gets about 9% faster per expert removed, but the draft accepts fewer
tokens when the output changes. The best case for 4 experts is about +9% bot decode (34.5 → 37.6 tok/s); the likely
case is about +5% to +6%. An anchored gain can also lower the NLL cost of the live top-k 5 at the same speed.

## 2. What we know

### 2.1 The data (results of 2026-11-04 to 2026-11-07)

| Step on K 160 | Summed NLL | Reasoning | Answer | Decisions (of 60) | Tool names (of 83) |
|---|---|---|---|---|---|
| Run noise (baseline against its repeat) | −0.22% | −0.34% | +0.00% | 41 → 42 | 77 → 77 |
| Top-k 6 → 5 | +0.87% (95% CI +0.25 to +1.61) | +1.35% | +0.04% | 43 → 42 | 78 → 79 |
| Top-k 5 → 4 | +3.69% (95% CI +2.69 to +4.64) | +3.64% | +3.78% | 42 → 39 | 79 → 79 |

- The step from 5 to 4 costs about four times the step from 6 to 5. At 6 → 5, almost all of the cost is in the
  reasoning. At 5 → 4, the answers start to lose too.
- The cost of top-k 4 is broad: 42 of 60 targets get worse, and the median target loses 3.1%. The cost per token
  does not depend on the length. Tool-call turns lose about twice as much per token as other turns.
- The skills that lose most at 5 → 4: stock-update (+5.9%), watchlist (+6.8%), quant (+4.9%), research (+4.7%).
- The per-target losses of the two steps do not correlate (r = 0.03).

| K 160 | Bot decode | Accepted per draft (bot) | Code decode | Accepted per draft (code) |
|---|---|---|---|---|
| Top-k 6 | 31.50 tok/s | 2.09 | 35.67 tok/s | 2.45 |
| Top-k 5 | 33.98 tok/s | 2.10 | 42.07 tok/s | 2.76 |
| Top-k 4 | 36.05 tok/s | 2.00 | 42.78 tok/s | 2.51 |

- Without the acceptance effect, each top-k step makes a verify step 8% to 9% faster, in both tests.
- At 5 → 4, the acceptance fell (bot −4.6%, code −9%), and that ate most of the gain. The code test also locks into
  one fixed text for some languages, so its numbers depend on the text more than on the model.
- In the router weights of the 2026-11-07 capture, the weights inside the top 6 are flat: the 6th expert probably
  holds 10% to 14% of a token's mixture, not a few percent. Each removed slot drops about the same weight.

### 2.2 The runtime (read from the image and the live container)

- **One global top-k.** `model.py` reads `num_experts_per_tok` once. The hash tables of layers 0–2 have one column
  per expert (`tid2eid`, vocab × top-k). The EXL3 runtime plans one prefill arena per top-k (1,052 MiB at 4,
  1,277 MiB at 5, 1,502 MiB at 6) before the CUDA graph capture. A change of top-k needs a restart.
- **A slot with id -1 costs nothing.** The route pack drops ids outside 0 … E−1 before it sorts the routes, and the
  EXL3 GEMM runs a tile only for a valid expert. So a -1 slot reads no weights and adds nothing to the output. This
  works under CUDA graphs, because the block count lives on the device. A slot with weight 0 and a valid id still
  costs its full compute.
- **The router.** With 160 experts, the target uses the PyTorch fallback `_topk_softplus_sqrt_torch` in
  `fused_topk_bias_router.py` (160 is not in the native kernel's list). The fallback computes the scores of all 160
  experts, picks the top-k by score plus bias, takes the weights from the unbiased scores, divides them by their sum
  (`norm_topk_prob`), and multiplies by 1.5. The hash layers take their weights from the same scores. So one patch
  in this file reaches all 43 layers.
- **The draft.** The DSpark draft builds its layers from the target's config, so it runs the target's top-k. Its own
  config value (6) is ignored. The draft uses the native kernel (64 experts), so a router patch does not reach it.
- **Not verified:** -1 ids end to end through the FusedMoE glue code, and the real number of distinct experts per
  verify step.

### 2.3 The research (sources in §13)

- **Gain anchoring** (Chen and Yao, 2026): activate the top k1 experts, but divide by the mass of the top k2. With
  k2 = k1 (what our router does), the gain rises with each cut. On Qwen3.6-35B-A3B, 8 → 4 experts cost 4.65 MMLU
  points with plain renormalization and 0.35 points with k2 = 16; at 8 → 6, the perplexity change went from +4.08%
  to −1.61% with k2 = 8. It needs no training. Perplexity and task scores favor different k2.
- **Static beats dynamic:** a controlled study of dynamic methods (thresholds, Ada-K and others) found at most about
  2.6 points over a static top-k with the same mean number of experts. Calibrated thresholds also drift on new data.
- **Per-layer top-k** (LExI) keeps the sensitive layers (often the first and the last) at a higher top-k.
- **Speculative decoding:** the verify step of an MTP draft loads almost the union of all tokens' experts, and its
  time grows with that union. So a smaller mean top-k shrinks the verify cost, if the acceptance holds.
- **Risks:** strict-format outputs (math answers, and probably tool calls and JSON) fail first when the experts per
  token fall.
- No source tests a reduced or dynamic top-k on DeepSeek V3, R1 or V4 with benchmarks.

## 3. The design: one top-k 6 runtime, with a drop rule and an anchor

Boot the K 160 model at top-k 6, and let the patched router decide per token which slots to drop (id -1) and how to
normalize the kept weights. Reasons:

- **The hash layers get a real drop rule.** The live top-k 5 cut the hash tables with an estimate from the token
  embedding (`topk_checkpoint.py`). With 6 columns, the router sees the real scores from the FFN input and drops the
  weakest of the six.
- **The anchor needs the 6th score.** For k2 = 6, the denominator is the sum of the six selected weights. For k2 > 6
  in layers 3–42, the router takes the top k2 of the full score vector; in the hash layers, k2 stays 6.
- **One arena, every variant.** Static cuts, thresholds and per-layer cuts are all -1 masks inside one top-k 6
  runtime. The arena costs 225 MiB more than at top-k 5, and the memory margin is about 22 GiB.
- **The draft returns to its trained top-k 6.** Phase D measures what that costs.

The rules, set by environment variables at boot (the router runs inside the CUDA graphs, so a rule cannot change
per request):

| Variable | Values | Meaning |
|---|---|---|
| `MOE_DROP` | `none`, `lowest:N`, `threshold:T:MIN` | Drop no slot; the N lowest-weight slots; or every slot whose share of the six is below T, but keep at least MIN |
| `MOE_NORM` | `kept`, `anchor:K2` | Divide by the sum of the kept weights (the current rule), or by the mass of the top K2 |
| `MOE_LAYERS` | `all`, a list such as `3-15,30-42` | The layers where the drop rule applies; other layers keep all six |

## 4. Order and gates

Each phase has a gate. A phase that fails its gate stops the plan, and §10 says what remains.

| Phase | Downtime | Gate |
|---|---|---|
| A: routing profile | About 50 min | A threshold rule can reach a mean of 4.5 experts or less, and the window union shrinks |
| B: patch and tests | None | All unit tests pass; `MOE_DROP=none` gives the stock output bit for bit |
| C: replay night | About 4 h | A top-4 point passes the NLL gate (+3%), or an anchored top-5 beats the live top-5 by more than the noise |
| D: speed | About 1 h | The bot decode beats the live model by more than the noise floor (4.2%), same day |
| E: full runs, review | About 2 h 40 min | The same bar as the live pick: 5 gates, the judge, your blind review |
| F: deploy | About 4 min | Your decision |

## 5. Phase A: the routing profile

Goal: numbers that pick the thresholds and predict the speed, before any replay.

1. Rebuild the K 160 mask at top-k 6 (no downtime, about a minute):
   `python mask_checkpoint.py --keep 160 --rank results/expert_rank_bot.json --out tp1-botmask-k160`.
2. Extend the capture patch (`image-patch/routed_experts_capturer.py`). Per layer, over every computed token, add:
   - the summed weight at each rank position 1 to 6, in weight order;
   - a histogram of the share of the 5th and of the 6th weight (bins of 2.5%);
   - the share of tokens whose 5th or 6th share is below 5%, 10% and 15%;
   - for windows of 6 consecutive tokens (a stand-in for one verify step of 1 + 5 draft tokens), the mean number of
     distinct experts under each rule: all six, `lowest:1`, `lowest:2`, and `threshold:T:MIN` for T in 0.05, 0.08,
     0.10, 0.12, 0.15 and MIN in 3, 4.

   Keep every shape static and do no boolean indexing: the first capture boot failed on one (`PLAN_BETTER_RANK.md`
   §4.3).
3. Run `capture_run.sh` with `CAPTURE_DIR` on the K 160 mask, over the rank data (63 segments, about 35 minutes).
4. Write the report: per layer and per skill, the share of the 5th and 6th experts, and for each rule the mean
   experts per token and the window union against top-k 5.

**Gate.** Skip the threshold points of Phase C if no threshold reaches a mean of 4.5 or less while it keeps the
union smaller than at top-k 5. A flat profile (the 5th share above 10% for most tokens) predicts that.

## 6. Phase B: the router patch and its tests

1. Copy `fused_topk_bias_router.py` from the image to `image-patch/`, and keep the original's diff as
   `image-patch/router-drop.diff`, as for the capture patch.
2. Change `_topk_softplus_sqrt_torch` after the gather of the weights:
   1. Compute the denominator: the sum of the six weights, or the top-K2 mass (`torch.topk` of the biased scores,
      then a gather of the unbiased scores) for `anchor:K2` in layers 3–42.
   2. Mark the slots to drop: the N lowest weights (`torch.sort`), or the shares below T with at least MIN kept.
   3. Set the dropped slots' ids to -1 and their weights to 0.
   4. Divide the kept weights by the denominator (`kept` uses the kept sum), then multiply by 1.5.
3. Give each router its layer index at bind time, so `MOE_LAYERS` can apply. The capture patch read the layer from
   the capture function's defaults; a field on the router is cleaner.
4. Unit tests (CPU, in the server image's Python):
   - `MOE_DROP=none` and `MOE_NORM=kept` return the stock function's output bit for bit;
   - `lowest:1` with `kept` equals a top-5 cut with renormalization;
   - `anchor:6` keeps the dropped slots' share out of the sum; `anchor:8` uses the top-8 mass;
   - a threshold keeps at least MIN slots, and every shape stays [M, 6].
5. A CUDA graph test: capture the function in `torch.cuda.graph` on small tensors, replay it, and compare. It needs
   a few MB beside the live server.
6. Mount the file in `compose-mask.yml` with the three variables (default `none`, `kept`, `all`).

### 6.1 Result (2026-11-09)

The patch is `image-patch/moe_drop.py` (the rule) and `image-patch/router-drop.diff` (17 lines in the router file).
`test_moe_drop.py` has 8 tests, and all pass in the server image (`./test_moe_drop.sh`). Changes against the steps
above:

- **The place of the patch.** The rule runs in `FusedTopKBiasRouter._compute_routing`, after the stock top-k, not
  inside `_topk_softplus_sqrt_torch`. The effect is the same. The router class knows its expert count, so the rule
  reaches only the target layers (160 or 216 experts) and never the draft (64 experts).
- **The layer index.** The n-th target router that the model builds is layer n. Each router writes one
  `[moe_drop] layer N … hash=…` line to the boot log, and the run script checks that there are 43 lines and that
  only layers 0–2 have hash tables.
- **The weakest slot.** `lowest:N` drops by the selection score (score + bias) in layers 3–42, as the stock router
  picks, and by the weight in the hash layers. So `lowest:1` in layers 3–42 is exactly a top-5 router, and C1
  changes only the hash layers against the live rule. A test checks this against a stock top-5.
- **No full copy of the router file in git.** `image-patch/build_router.sh` applies the diff to the image's file.
  A full copy would add about 475 lines to the pull request.
- **An override, not an edit.** `compose-drop.yml` mounts the two files on top of `compose-mask.yml`, so the other
  run scripts do not change. It also pins the KV cache at 8 GiB, as on the live server, and sets
  `GPU_MEMORY_UTILIZATION` to 0.90 (with a pinned KV cache, that value only sets vLLM's boot check).
- **Bit for bit.** With `MOE_DROP=none` and `MOE_NORM=kept`, the patched router returns the stock tensors
  (`torch.equal`), because the rule does not run.
- **CUDA graphs.** A test captures the rule in a CUDA graph, replays it on new inputs and compares it with an eager
  call. It failed once with a CUDA error while the live server ran, then passed in four runs.

## 7. Phase C: the replay night

`rank_night.sh` with one boot per point (`POINTS` and a variable for the rule), each on `tp1-botmask-k160`
(top-k 6) with the patch. Each point takes about 40 minutes: 4 for the boot, 35 for the 60 targets.

| Point | `MOE_DROP` | `MOE_NORM` | Question |
|---|---|---|---|
| C0 | `none` | `kept` | Does the patch change nothing? Must equal `botmask-k160.jsonl` (+1.37%) within the noise |
| C1 | `lowest:1` | `kept` | Today's rule, but with the real hash scores. Against the live mask (+2.27%): how much did the estimate cost? |
| C2 | `lowest:1` | `anchor:6` | The anchored top-5. Is it better than C1? |
| C3 | `lowest:2` | `anchor:6` | The anchored top-4 |
| C4 | `lowest:2` | `anchor:8` | A lower gain for top-4 (hash layers stay at 6) |
| C5 | a threshold from Phase A | the best anchor of C2–C4 | A mean of about 4.3 to 4.5 experts |

The first request of C1 also tests the -1 ids through the FusedMoE glue. If it fails, the run script restores the
live server and the night stops.

**Gates** (against the baseline, `baseline-k216.jsonl`): the summed NLL at most +3%; tool names at least 98% of the
baseline; decisions not below the baseline by more than the noise (3).

| Result | Next |
|---|---|
| C3, C4 or C5 passes | Phase D on the best one, with C2 as the reference |
| Only C2 passes and beats C1 by more than 0.23 points | Phase D on C2: the same speed as today, a better model |
| C3 or C4 misses the gate by less than 0.5 points | A second night: `lowest:2` on one layer group at a time (3–15, 16–29, 30–42), then on the two most robust groups |
| Nothing passes, and C2 does not beat C1 | Stop (§10) |

## 8. Phase D: the speed tests

- Run the bot decode test 5 times on the best point, and 5 times on the live model on the same day. The two days of
  the earlier tests differ, and the gain may be close to the noise floor of 4.2%.
- Run the code test once for information only. Its prompts lock into one fixed text for some languages.
- Read the acceptance per position. An anchored gain keeps the output closer to the stock model, so the acceptance
  should fall less than at a plain cut.
- The draft runs at top-k 6 in this runtime. Compare its acceptance with the live draft at top-k 5.

**Prediction.** Top-4 points: +5% to +9% bot decode. Threshold points: less. The anchored top-5: the same speed as
today, plus or minus the draft effect.

## 9. Phase E and F: the full runs, the review and the deploy

- **Phase E.** Step F (`PLAN_E2E_RUNNER.md` §15) on the chosen point, then the judge against the three baseline runs,
  then your blind review of 20 pairs (`PLAN_FINAL_REVIEW.md`). Watch the tool calls and the takes JSON in particular:
  strict formats fail first.
- **Phase F (your decision).** Build the top-k 6 prune: `SRC=tp1-botmask-k160 OUT=tp1-prune-k160 ./prune_run.sh`
  (about 11 minutes, 73 GB). Copy `compose-k160t5.yml` to a new compose file that serves that folder at top-k 6 and
  mounts the patched router with the chosen variables. Keep the served name pattern
  (`deepseek-v4-flash-0731-k160…`). The rollback is the current compose file.
- **A smaller variant if only the anchored top-5 wins.** Keep the live folder (top-k 5) and mount the patch with
  `MOE_NORM=anchor:6` only. Layers 3–42 then get the anchor from the full score vector; the hash layers keep the
  current rule, because their tables have 5 columns. Test it as one more replay point before a deploy.

## 10. If Phase C stops the plan

- The live model stays as it is: K 160 at top-k 5.
- The speed then comes from elsewhere: the draft (a longer draft, `DSPARK_TOKENS`), or a second, fast model for the
  light requests (`PLAN_MODEL_ROUTER.md` in the bot repo).
- A cheap heal remains possible: train only the 160 selection biases per layer (or the router rows) against the stock
  model's outputs on the rank data. The research reports router-only fine-tuning at minutes on a GPU, but no study
  tested it after a top-k cut. It needs its own plan.

## 11. What not to do

- **A true per-layer top-k** in the runtime: each distinct top-k plans its own arena (1.0 to 1.5 GiB each), and the
  KV margin cannot pay that. The -1 masks give the same effect inside one runtime.
- **A slot with weight 0 and a valid id:** it saves nothing.
- **A separate top-k for the draft:** the draft reads 5% to 7% of the target's expert bytes, so the gain is at most
  about 1%, and the acceptance could fall.
- **A threshold calibrated on the replay targets:** calibrate on the rank data (Phase A), test on the replay set.

## 12. Risks

| Risk | Effect | Guard |
|---|---|---|
| -1 ids fail in the FusedMoE glue code | The boot or the first request fails | C1's first request tests it; the run script restores the live server |
| The patch is not safe under CUDA graphs | The boot fails at graph capture | The graph test of §6 step 5; static shapes, no boolean index |
| The anchor helps the NLL but hurts tool calls | Format faults in real use | Phase E watches tool calls and the takes JSON; your blind review |
| The speed gain stays inside the noise | No reason to deploy | Phase D runs 5 + 5 bot tests on one day |
| The router is slower with the extra sort and topk | Less gain | Phase D measures it; the fallback already sorts 160 scores per token |

## 13. Effort, downtime and sources

| Phase | Work | Downtime |
|---|---|---|
| A | Half a day for the capture extension, 1 h for the run | About 50 min |
| B | Half a day for the patch and the tests | None |
| C | One night | About 4 h (6 points) |
| D | 1 h | About 1 h |
| E | As for the live pick | About 2 h 40 min for step F |
| F | 15 min for the build, 4 min for the switch | About 4 min |

About a week to the Phase C decision, and two weeks to a deploy.

Sources. The first was checked on 2026-11-09; the others come from the research report of the same day:

- Chen and Yao, "Training-Free Halving of Activated Experts in Fine-Grained Mixture-of-Experts Models",
  https://arxiv.org/abs/2609.04575
- "Don't Skip the Baseline" (static against dynamic expert reduction), https://arxiv.org/abs/2610.28369
- LExI (per-layer active experts), https://arxiv.org/abs/2509.02753
- EcoSpec (expert union in MTP verification), https://arxiv.org/abs/2607.12696
- MoE-Spec (expert budget at verify time), https://arxiv.org/abs/2602.16052
- Router-only fine-tuning, cost-normalized, https://arxiv.org/abs/2609.06076
