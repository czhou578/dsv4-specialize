# Plan: a smaller vocabulary for the head (the vocabulary trim)

**Status:** 2026-11-11: plan written; nothing built · **Author:** colin-spark · **Date:** 2026-11-11
**Source:** the research list of 2026-11-11: a vocabulary trim of the drafter (VocabTrim) and a vocabulary prune of
the target head. On this stack the two ideas change the same tensor, so one plan covers both.
**Related:** `../serve-tune/PLAN_SERVE_TUNE.md` (§6.7 the FP8 draft head, §7.2 and §7.4 the FP8 target head: the speed
data of this plan), `PLAN_MD_REVIEW.md` (the deploy review method, and the folder that can be live at Phase C),
`PLAN_LOADER_PATCH.md` (the patched model file of the MD prune), `PLAN_FINAL_REVIEW.md` (the deploy chain),
`PLAN_OVERNIGHT_MASK_GRID.md` §2 (the safety rules of a test boot)
**Scope:** a token-set tool and its test; a patch of two image files (the model file and the draft file), built from
diffs in this repo and mounted on a test boot; one test window on port 8890. The full runs and the deploy are your
decision. No change to the live server or the server repo. Each phase is one pull request of at most 400 lines.

| Step | Status |
|---|---|
| §5 Phase A: the token set and its coverage (no downtime, half a day) | ⬜ |
| §6 Phase B: the patch, its build script and its tests (no downtime, one day) | ⬜ |
| §7 Phase C: the test window (about 3.5 h downtime) | ⬜ |
| §8 Phase D: the full runs, the judge and your blind review (T only, one night) | ⬜ |
| §9 The decision and the deploy | ⬜ |

---

## 1. Short answer

- **One tensor, two readers.** The head (129,280 × 4,096 BF16, 0.99 GiB) gives the logits of the target. The DSpark
  draft has no head of its own and reads the same tensor. So a trim of the head can reach the draft, the target, or
  both.
- **The vocabulary cost of a step.** A decode step reads the head twice: once for the draft (5 rows in one call) and
  once for the verify pass (6 rows). The draft's Markov head adds a 129,280 × 256 matrix, which the draft reads once
  for each of the 5 positions. In all, about 2.28 GiB of each step scales with the vocabulary size.
- **The measured slope.** The FP8 head of the target (H1 of serve-tune) removed about 0.99 GiB of these reads. It gave
  +3.1% bot decode and +7.2% code decode, with the same acceptance.
- **Two variants:**
  - **D, the draft only.** The draft proposes from a small set of tokens, for example 16,000. The verify pass keeps the
    full head, so the output does not change. Estimate: about +3% to +4% bot decode.
  - **T, the target and the draft.** The head keeps a larger set, for example 48,000. The target cannot emit a token
    outside the set. The logits of the kept tokens do not change, so the cause of H1's failure (FP8 rounding) does
    not apply. Estimate with D on top: about +5% to +6% bot decode, and 0.62 GiB less memory.
- **The risk of T.** The bot copies names, tickers and URLs from tool results. A token outside the set forces the model
  to spell the text with other tokens. Phase A measures this offline, before any downtime.
- **The order.** An offline study first: it sets the two sizes, or it stops T. Then one test window. A deploy of D
  needs only your go, because its output does not change. T needs the full deploy chain.

## 2. What we know

### 2.1 The head and the draft (read from the image and the server repo on 2026-11-11)

- **The tokenizer** has 128,000 BPE tokens and 1,283 added tokens (ids 0 to 129,279). Of the BPE tokens, 72,278 are
  ASCII text, 36,064 hold CJK characters, 18,185 hold other text, and 1,473 are partial UTF-8 sequences. The
  pre-tokenizer splits numbers into groups of 1 to 3 digits.
- **The draft shares the head.** The draft class sets `has_own_lm_head = False`, and `load_dspark_model` gives it the
  target's head module (server repo, `image-patch/vllm/models/deepseek_v4/nvidia/dspark.py:318-340`).
- **One head read per draft step.** The speculator calls `compute_draft_logits` once for all 5 positions (image,
  `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:209`). Then, for each position, it adds the Markov bias of the
  previous token: `markov_w2`, 129,280 × 256 (`dspark_markov_rank` in the draft config). The depth estimate of
  serve-tune §6.1 assumed one head read per draft position. The code shows one read per step.
- **The speculator already supports a reduced draft vocabulary.** When the draft model has `draft_id_to_target_id`,
  `load_draft_model` builds a scatter index and a −inf buffer of the target width (`speculator.py:175-191`). A
  probabilistic draft scatters its logits into the target columns before it samples. A greedy draft calls
  `map_draft_to_target` (`speculator.py:246-268`). The DeepSeek V4 draft class sets both hooks to the full vocabulary
  today (`dspark.py:324` and `:426`). The Qwen3 DSpark class uses the same hooks with a smaller `draft_vocab_size`.
- **D keeps the output.** The live server uses `draft_sample_method=probabilistic` and
  `rejection_sample_method=standard` (the defaults of `serve-ds4-flash.sh:100-101` in the server repo). Standard
  rejection sampling keeps the target distribution for any draft distribution, also for a draft with zero
  probability outside its set.
- **The hooks exist.** `serve-tune/image-patch/fp8_target_head.py` puts a new `quant_method` on the head after the
  load (`head-fp8.diff`: five lines at the end of `load_weights`). `maybe_init_fp8_draft_head` runs at draft load,
  before the CUDA graph capture.
- **A token outside the set shows in the replay.** The OpenAI server writes a −inf logprob as −9999.0, also in
  `prompt_logprobs` (`clamp_prompt_logprobs`). So one replay target token outside the set of T adds 9999 to the summed
  NLL.

### 2.2 The speed data

| Test (against its base of the same day) | Vocabulary reads removed per step | Bot decode | Code decode | Accepted per step |
|---|---|---|---|---|
| T3: FP8 draft head (serve-tune §6.7) | about 0.49 GiB | −2.1% | +2.1% | 2.20 → 2.10 |
| H1: FP8 target head, shared by the draft (serve-tune §7.4) | about 0.99 GiB | +3.1% | +7.2% | 2.14 → 2.14 |

- A step takes about 91 ms (35 tok/s, with 2.2 accepted tokens plus one). At the 273 GB/s of the GB10, 0.99 GiB takes
  about 3.9 ms, about 4% of a step. The two measured results agree with this, inside their noise.
- A part of the T3 acceptance drop is noise between boots: H0 accepted 2.14 with the settings of T0 (2.20).
- The noise floor of the bot decode test is 4.2%. The noise of the code test is about 1.65 tok/s (4%).

### 2.3 The estimates

V = 129,280. N is the size of a set. The table scales H1's bot gain by the bytes that each point removes. The values
are estimates, not measurements.

| Point | Vocabulary reads removed per step | Estimated bot decode | Estimated code decode | Memory |
|---|---|---|---|---|
| D at 16,000 | 1.13 GiB | +3% to +4% | about +8% | +0.13 GiB (a copy of 16,000 rows) |
| D at 32,000 | 0.97 GiB | about +3% | about +7% | +0.26 GiB |
| T at 48,000 (the draft also at 48,000) | 1.43 GiB | about +4.5% | about +10% | −0.62 GiB |
| T at 48,000 and D at 16,000 | 1.75 GiB | +5% to +6% | about +13% | −0.62 GiB (the draft rows are a view) |

D alone probably stays under the 4.2% noise floor of the bot test. The code test can show it.

### 2.4 The research (from the list of 2026-11-11; not checked again here)

- VocabTrim rebuilds only the drafter's head from the most frequent tokens. The target verifies every token, so the
  output does not change, and only the acceptance can fall. A follow-up cut a 128K vocabulary to 13K, with up to 16%
  lower latency on domain tasks.
- The list called a trim of the target head "exact for your use case". It is exact only when the target's choice is
  inside the set. At a temperature above 0, the set also takes the probability mass of the tokens outside it.

## 3. Terms

| Term | Meaning |
|---|---|
| V | The full vocabulary: 129,280 ids |
| Set | The kept token ids, in one fixed order: the mandatory tokens first, then the tokens by their count (§5.1) |
| N_T | The size of the target set (variant T) |
| N_D | The size of the draft set. It is the first N_D ids of the same order, so it is a part of the target set |
| Miss | A token of a held-out output that is not in the set |
| H0 | The live settings, booted in the same window |

## 4. Order and gates

| Phase | Downtime | Gate |
|---|---|---|
| A: the token set | None | T: some N_T up to 64,000 has zero misses on the held-out outputs. D: some N_D up to 32,000 covers at least 99.5% of the held-out output tokens |
| B: the patch and its tests | None | All tests pass; the kept logits equal the BF16 logits bit for bit; the graph capture works |
| C: the test window | About 3.5 h | The bot decode beats H0 by more than 4.2%; for T, also the replay check |
| D: the full runs (T only) | About 7.5 h | As for the MD prune: the five gates, the judge, your blind review |
| Deploy | About 4 min | Your decision |

If T fails gate A, the plan continues with D only. If both fail, the plan stops.

## 5. Phase A: the token set and its coverage (no downtime)

### 5.1 The order of the set

1. The mandatory tokens, in id order:
   - all 1,283 added tokens (the special tokens, `<think>`, `</think>` and the DSML markup);
   - the 256 single-byte tokens, so that any text can still be spelled, and a grammar mask always keeps one allowed
     token;
   - every token that holds only digits (the groups of 1 to 3 digits);
   - every token of the tool schemas (`tools.json`), as the encoder writes them in DSML: the tool names and the
     parameter names.
2. The tokens of the calibration outputs, by count, the most frequent first.
3. The tokens of the calibration prompts (the system prompt, the user messages, the tool results), by count. These
   hold the text that the bot copies.
4. All other ids, in id order. The sizes of this plan never reach them.

### 5.2 The data

- **Calibration (builds the set):** `model_eval/data/rank/sessions.jsonl` in the eval worktree, the rank data of
  `PLAN_BETTER_RANK.md` §3. Its leak rule keeps the held-out sessions out.
- **Held-out (tests the set):**
  - the replay targets: `model_eval/data/sessions.jsonl`, split `held_out`, as `replay.pick` selects them;
  - the full-run sessions of the held-out prompts: the sessions that the leak rule of `rank_data.py` drops. Export
    them with `export_sessions.py`, as `rank_data.py` does, into a folder out of git.
- **Tokenizer:** as `serve-tune/suffix_match.py` does it: the tokenizer of the live folder, and the output pieces as the
  DeepSeek V4 encoder writes them. Import `pieces` and `dsml` from that file. Do not copy them.

### 5.3 The tool: `vocab_set.py`

- `--build` writes `results/vocab-set.json`: the ordered ids, the mandatory count, and the size of each part of §5.1.
- `--coverage` writes `results/vocab-coverage.json`. It prints one row for each N of 8,000, 16,000, 24,000, 32,000,
  48,000 and 64,000:
  - the share of the held-out output tokens in the set, for the reasoning and for the answer;
  - the misses: the count, and each miss with its text and its context;
  - the misses that also occur in the prompt of the same session (the copy misses);
  - the misses inside a tool call (a DSML parameter);
  - the vocabulary reads that the size removes per step, and the memory change.
- `test_vocab_set.py` tests:
  - the order: the mandatory tokens first, no duplicates, all ids valid;
  - each smaller set is the start of each larger set;
  - the coverage counts of a small session made by hand;
  - no held-out session in the calibration data.

### 5.4 The gate and the choice

- **N_T:** the smallest size with zero misses on all held-out outputs. If no size up to 64,000 has zero misses, drop T.
  Zero misses on a few hundred thousand tokens still allows a rare miss in real use. Phase D tests that end to end.
- **N_D:** the smallest size that covers at least 99.5% of the held-out output tokens. A miss costs at most the
  acceptance of one draft position.
- Record the table and the choice in §5.5.

### 5.5 Result

(Not run yet.)

## 6. Phase B: the patch, its build script and its tests (no downtime)

### 6.1 The variables

| Variable | Values | Meaning |
|---|---|---|
| `VOCAB_TRIM_FILE` | A path in the container | The ordered ids of §5.3 (`results/vocab-set.json`, mounted read-only) |
| `VOCAB_TARGET_N` | 0, or N_T | 0 keeps the full head for the target. N_T keeps the first N_T ids |
| `VOCAB_DRAFT_N` | 0, or N_D | 0: the draft uses the target's set (the full vocabulary when `VOCAB_TARGET_N` is 0). N_D: the draft uses the first N_D ids |

### 6.2 The files

| File | What it does |
|---|---|
| `image-patch/vocab_trim.py` | The module: it reads the ids, installs T on the head, and installs D on the draft |
| `image-patch/vocab-trim-model.diff` | A hook at the end of `load_weights` in the model file, as in `head-fp8.diff` |
| `image-patch/vocab-trim-dspark.diff` | The hooks in the draft file: the install at draft load, `compute_draft_logits`, `map_draft_to_target` and `markov_bias` |
| `image-patch/build_vocab.sh` | Builds `image-patch/vocab/` (not in git). The model file comes from the image (`BASE=stock`) or from `image-patch/perlayer/` (`BASE=perlayer`, for the MD prune). The draft file comes from the server repo's patch. The script stops if a diff does not apply |
| `compose-vocab.yml` | An override on top of `compose-mask.yml` and `serve-tune/compose-tune.yml` (and `compose-perlayer.yml` for MD). It mounts the module, the two built files and the id file, and sets the variables |
| `test_vocab_trim.py`, `test_vocab_trim.sh` | The tests of §6.4, in the server image on the GB10, as `test_fp8_head.sh` |

### 6.3 How the patch works

- **T, at the end of `load_weights`:**
  1. Copy the first N_T rows of the head, in set order, into one contiguous BF16 weight.
  2. Put a new `quant_method` on the head. Its `apply` computes the N_T logits. Then it writes them with `index_copy_`
     into a tensor of the full head width that holds −inf. The speculator's reduced-vocabulary buffer uses the same
     fill.
  3. Free the full weight, as `fp8_target_head.install` does.
- **D, at draft load (in `maybe_init_fp8_draft_head`, before the graph capture):**
  1. Set `draft_id_to_target_id` to `ids[:N_D] − arange(N_D)`, the offset form that `load_draft_model` expects.
  2. Make the draft head weight. When T is on, it is a view of the first N_D rows of the T weight (no new memory).
     When T is off, it is a copy of N_D rows of the full head.
  3. Copy the N_D rows of `markov_w2`.
  4. Then `compute_draft_logits` returns N_D logits, `markov_bias` returns N_D values, and `map_draft_to_target` maps
     an index to its id.
- **Rules for the CUDA graphs:** all tensors exist before the capture, and the index tensors do not change. Use no
  boolean index (`PLAN_BETTER_RANK.md` §4.3 shows why).
- **Guards:** the install stops with an error in these cases:
  - `FP8_TARGET_HEAD` or `DSPARK_FP8_DRAFT_HEAD` is on;
  - the id file has fewer ids than N;
  - a mandatory id is not in the set.
- **Log:** each variant writes one boot-log line with N and the bytes. The window script checks for it.

### 6.4 The tests

1. T: on the kept columns, the logits equal the BF16 logits bit for bit. All other columns are −inf.
2. T: when the full argmax is in the set, the argmax does not change.
3. D: the scatter index of the speculator equals the kept ids, and `map_draft_to_target` gives the same ids back.
4. D: on the kept columns, the base logits plus the Markov bias equal the values of the full path.
5. T and D together: the draft weight shares the storage of the T weight.
6. The capture and the replay of a CUDA graph give the eager output, for both paths.
7. The memory that T frees equals (V − N_T) × 4,096 × 2 bytes.

Run the GPU tests beside the live server. Check `MemAvailable` first.

### 6.5 Result

(Not run yet.)

## 7. Phase C: the test window (about 3.5 h downtime)

`vocab_window.sh` runs the window on `run_lib.sh`, as `serve-tune/tune_replay.sh` does. Each point boots on port 8890
with the live settings: the KV pin of 7.25 GiB and the live bot-rank draft. The folder is the one that is live at that
time: the K 160 prune (`tp1-prune5b-k160`, as in serve-tune §7), or the MD prune (`tp1-prunemd5-k149`) if
`PLAN_MD_REVIEW.md` deploys it.

**CAUTION: the window stops the bot's model server for about 3.5 hours.** Start it only with your go. Pause the Hermes
cron jobs first. Start the overnight guard with a time after the restore (`PLAN_OVERNIGHT_MASK_GRID.md` §2).

| Point | Settings | Tests | Time |
|---|---|---|---|
| V0 | H0: no trim | Smoke, bot decode × 5, code test | 35 min |
| V1 | D: `VOCAB_DRAFT_N` = N_D | Smoke, the D log line, bot decode × 5, code test | 35 min |
| V2 | T: `VOCAB_TARGET_N` = N_T | Smoke, the T log line, replay (60 targets), bot decode × 5, code test | 75 min |
| V3 | T and D (only if V2 passes the replay check) | Smoke, bot decode × 5, code test | 35 min |
| V0′ | H0 again: the drift check | Bot decode × 3 | 20 min |

The smoke requests are those of `serve-tune/prefill_probe.py`: the JSON schema and both tool choices. They test the
grammar mask with a smaller set.

**The replay of V2 checks the patch, not the quality.** On the kept columns the logits do not change. A smaller softmax
can only lower the NLL and raise the rank of a target token. So, against the live replay (`prune5-k160.jsonl`, as
`tune_replay.sh` uses it, or the MD replay of `PLAN_LOADER_PATCH.md` §6.2):

- no target token at −9999 (Phase A predicts zero);
- the summed NLL is not higher by more than 0.56% (the noise of one replay);
- the decisions are not lower by more than 6, and the tool names stay within the replay gate (at least 98% of the
  baseline).

A failure here means a fault in the patch. The quality risk of T is the tokens outside the set. Phase A measures it
offline, and Phase D measures it end to end.

### 7.1 Gates

| Point | Gate |
|---|---|
| V1 (D) | The bot decode beats V0 by more than 4.2%, and V0′ is within 4.2% of V0 |
| V2 and V3 (T) | The replay check passes, and the bot decode beats V0 by more than 4.2% |

The acceptance per step and the code test are for information. If V1's acceptance falls by more than about 3% (the
noise between boots), N_D is probably too small.

| Result | Next |
|---|---|
| V3 or V2 passes | Phase D on the faster point |
| Only V1 passes | §9: deploy D (no output change), your decision |
| V1 misses the bot gate, but its code test gains more than the code noise | Your decision. D is then probably a real gain below the resolution of the bot test. A combined test with the 128-expert draft (serve-tune §8.2) can show it |
| Nothing passes | Stop, and record the data. The head is not a lever at this step time |
| V2 fails the replay check | Find the fault in the patch and test again, or drop T |

### 7.2 Result

(Not run yet.)

## 8. Phase D: the full runs, the judge and your blind review (T only)

T changes the output, so it needs the same bar as the MD prune (`PLAN_MD_REVIEW.md`):

- Run the 19 held-out prompts twice on the T point (`vocab-1`, `vocab-2`). Use `review_window.sh` with
  `TEST_COMPOSE` set to include `compose-vocab.yml`.
- For the base runs, use the live runs of the MD review if the bot code is still `a0fd27b`. Else run new base runs with
  `review_live.sh` (no downtime).
- Then the five gates, the judge and your blind review of 20 pairs, with `review_finish.sh` and `review_report.py`.

One more check: `vocab_set.py --spelled` tokenizes each assistant turn of the new runs again, and counts the tokens
outside the set. Each one is a place where the model spelled a word with other tokens. The report lists them with
their text. This count is information for the judge and for your review, not a gate.

### 8.1 Result

(Not run yet.)

## 9. The decision and the deploy

- **D:** change the mounts in the server repo's compose file (`compose-k160t5.yml`, or the MD compose file): the built
  draft file, the module, the id file and `VOCAB_DRAFT_N`. Then restart (about 4 min). Rollback: the previous compose
  file. D needs no full runs, as for the KV pin (serve-tune §6.8).
- **T, or T and D:** the same, plus the built model file and `VOCAB_TARGET_N`. Do this only after Phase D passes and
  you decide.
- Each deploy keeps the served names, so Hermes needs no change.

## 10. What not to do

- **Do not trim the embedding table.** A step reads one row of it for each token, so a trim saves memory only. Also,
  every prompt token needs its row.
- **Do not build the set from held-out sessions.** Do not choose N from the replay result.
- **Do not use a set for each request.** The head weight is static inside the CUDA graphs.
- **Do not combine the trim with an FP8 head.** Both FP8 variants failed their gates, and the patch refuses them.

## 11. Risks

| Risk | Effect | Guard |
|---|---|---|
| T: the bot needs a token outside the set | The model spells the word with other tokens, or writes another word | Phase A: zero misses on the held-out outputs; the byte and digit tokens are mandatory; Phase D and the spelled count |
| A grammar mask allows only tokens outside the set | All logits are −inf, and the sampler fails | The 256 byte tokens are mandatory, so the next byte is always allowed; the smoke requests test the JSON schema and the tool choices |
| D: the draft misses more tokens than Phase A predicts | A lower acceptance and less gain | The coverage gate of Phase A; the window measures the acceptance |
| The patch breaks the graph capture | The boot fails | All tensors exist at load; test 6 of §6.4; the window script restores the live server |
| A diff does not apply after a change of the image or the server repo | No build | `build_vocab.sh` stops |
| The gain stays inside the noise | No reason to deploy D alone | V0 and V0′ in the same window; T and D together is the larger point |

## 12. Effort and timeline

| Phase | Work | Downtime |
|---|---|---|
| A | Half a day | None |
| B | One day, with about 30 min of GPU tests beside the live server | None |
| C | One evening | About 3.5 h |
| D (T only) | One night, then the judge and your review | About 7.5 h |
| Deploy | 15 min | About 4 min |

Phase C waits for the MD decision (`PLAN_MD_REVIEW.md` §8), so that it tests the folder that is live.

The pull requests, each at most 400 lines:

1. Phase A: `vocab_set.py` and `test_vocab_set.py`.
2. Phase B: the module, the two diffs, the build script and the tests.
3. Phase C: `compose-vocab.yml` and `vocab_window.sh`.
