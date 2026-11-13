# Plan: fewer layers (depth pruning)

**Status:** 2026-11-13: refreshed for the live MD prune (§0); Phase A done (PRs #39 to #45); Phase B ready to start ·
**Author:** colin-spark · **Date:** 2026-11-10, refreshed 2026-11-13
**Source:** a note on depth pruning: rank the layers by Block Influence, confirm the best ones with the replay gate,
heal the cut with a linear map, then remove the layers from the files. This plan checks each step against this stack.
**Related:** `PLAN_SERVE_TUNE.md` (§7.1 the replay gate; H1 of the Step 2 window), `PLAN_COLD_EXPERTS.md` (the other
memory lever), `PLAN_BETTER_RANK.md` (the K 144 failure), `PLAN_PHYSICAL_PRUNE.md` (the prune tool and its test boot),
`PLAN_FINAL_REVIEW.md` (the deploy chain), `PRUNE.md` §5 (the 72 GiB cap)
**Scope:** a patch of the model file, new tools and run scripts in this repo, and test windows on port 8890. No change
to the live server.

| Step | Status |
|---|---|
| §6 Phase A: the patch, the tools and the tests (no downtime, about 2 days) | ✅ 2026-11-13: A1a to A3a (PRs #40 to #45); all tests pass; A3b (`skip_replay.sh`) waits for Phase B (§6.4) |
| §7 Phase B: the probe window (about 3 h downtime) | ⬜ |
| §8 Phase C: the replay window (about 4 to 5 h downtime) | ⬜ |
| §9 Phase D: heal maps for sets (only if §8.4 says so) | ⬜ |
| §10 Phase E: the physical removal and the deploy chain (only if a set passes) | ⬜ |

---

## 0. Refresh (2026-11-13): the MD prune is live

The plan was written on 2026-11-10 for the K 160 prune. Since 2026-11-12 the live server runs the MD prune
(`tp1-prunemd5-k149`, `compose-md149t5.yml`, `PLAN_MD_DEPLOY.md`). These facts change. Where a later section gives an
older number, this section is correct.

| Item | The plan of 2026-11-10 | Now |
|---|---|---|
| The live folder | `tp1-k160t5`, 74.09 GiB loaded | `tp1-prunemd5-k149`, 140 to 160 experts per layer, 69.79 GiB loaded |
| The 72 GiB cap of `PRUNE.md` §5 | 2.09 GiB above it | Met. The freed memory is for the second model of the router plan |
| The live replay NLL | 0.4524 (`prune5-k160`) | 0.4566 and 0.4558, a mean of 0.4562 (+2.75%; `lp-p1-a`, `lp-p1-b`) |
| Left under the gate (0.4573) | 0.0049 (+1.08% over the live replay) | **0.0011 (+0.24% over the live replay)** |
| The noise of one replay | 0.23% | Up to 0.56% between windows (`PLAN_COLD_EXPERTS.md` §7.3) |
| A whole layer (layers 3–39) | 1.55 to 1.57 GiB | 1.39 to 1.51 GiB (140 to 154 experts) |
| An F block | 1.44 GiB | 1.26 to 1.39 GiB |
| The model file of the boots | The image's file | The per-layer file of `PLAN_LOADER_PATCH.md`: `build_skip.sh` applies `per-layer-experts.diff`, then `layer-skip.diff` |
| The compose files | `compose-mask.yml`, `compose-tune.yml` | Also `compose-perlayer.yml`, under `compose-skip.yml` |
| The KV pin | 8 GiB | 7.25 GiB, as on the live server |
| The FP8 head (`HEAD=1`) | An option of `build_skip.sh` | Dropped: H1 did not deploy |

What this means for the plan:

1. **The budget is almost gone.** The cold cut S15 cost +0.82% against its reference and failed the gate
   (`PLAN_COLD_CUT2.md` §4.1). A whole layer is a larger change than S15. So a pass in Phase C needs a block whose
   rise is near zero. A likely result is still zero layers, now more likely than in §1.
2. **Phase B still runs.** It is the cheap test of §1, and it measures the rise of each block. Only a block near
   zero can go on. The thresholds of §7.6 are scaled to the new budget.
3. **Phase C, if it runs, uses two replays per point** and gates on the mean, as `PLAN_COLD_CUT2.md` §3 does. The S0
   check allows 1%, because a repeat check needs a limit above the noise (`PLAN_COLD_CUT2.md` §4.1, item 6).
4. **The prefix cache of the probe boot is off.** The reset endpoint of §6.2 needs `VLLM_SERVER_DEV_MODE=1`. The
   probe boot passes `--no-enable-prefix-caching` in its place. (`PREFIX_CACHE=0` does not work: the server's
   entrypoint sets `PREFIX_CACHE=1`, §7.7, attempt 3.) The client counts the shared prefix (the system prompt and the
   tools) once: it tells the patch the first position to count (`count_from` in the skip file). The screen requests
   ask for prompt logprobs, so vLLM skips the prefix cache for them in any case.
5. **The heal code waits.** `LAYER_HEAL` and `--fit` come in their own pull request, only if Phase B or C asks for
   them (§6.6). The Gram sums of the probe come in Phase A, so that Phase B can collect them.


## 1. Short answer

- **The idea:** skip a whole layer, or one half of a layer, at boot. A block that changes the state little can go.
- **Two levers:** a whole layer holds about 1.56 GiB and about 1.5% to 2% of the bytes of a decode step. The FFN half
  holds 92% of the memory of a layer. The attention half holds 8% of the memory, but about a third of the bytes that
  the layer reads per step.
- **The budget is small.** The replay gate allows +3% NLL over the stock model. The live model uses +1.89%, so 0.0049
  NLL is left: +1.08% over the live replay. H1 of `PLAN_SERVE_TUNE.md` (the FP8 head) did not deploy, so all of it is
  free.
- **The cheap test comes first.** One probe window (about 3 h) ranks all blocks by Block Influence (BI). It measures
  the NLL of the 28 best blocks on calibration data. Then it builds sets of 2 and 3 blocks, one block at a time. The
  result tells whether the rest of the plan is worth the work.
- **The goal:** two whole layers or two FFN halves bring the load from 74.09 GiB to about 71 GiB. That is under the
  72 GiB cap of `PRUNE.md` §5. A speed gain above the noise needs 3 or more whole layers.
- **A likely result is zero layers.** The K 144 cut failed by less than one point of NLL, and a whole layer is a much
  larger change than 16 cold experts per layer.
- **The price of a GiB.** H1 saved 0.49 GiB for +0.91% NLL against the live replay, and you chose no deploy. A layer
  set that passes the gate still needs your decision on its price (§4.2).

## 2. Terms

| Term | Meaning |
|---|---|
| Block | One candidate for a skip: a whole layer (`L17`), the attention half of a layer (`A17`), or the FFN half of a layer (`F17`) |
| Skip set | The blocks that one boot or one screen skips |
| State | The merged residual of the four mHC streams between two blocks: 4 × 4,096 values per token |
| BI | Block Influence (ShortGPT, https://arxiv.org/abs/2403.03853): 1 minus the mean cosine between the state before a block and the state after it. A small BI means that the block changes the state little |
| Screen | The NLL of the screen set in the probe boot, with a skip set that the client changes at run time |
| Screen set | A fixed set of calibration segments for the screen. No replay session is in it |
| Probe set | A second fixed set of calibration segments, for the BI pass and the Gram sums. It does not overlap the screen set |
| Rise | The change of the NLL against the reference with no skip, in percent |
| Heal map | A linear map at the boundary of a skipped block, fitted in closed form on the probe set (ReplaceMe, https://arxiv.org/abs/2505.02819; Ghosted Layers, https://arxiv.org/abs/2605.15491) |
| Ratio-4 layer | A layer with `compress_ratios` 4 and an indexer: the even layers 2–42 |
| Ratio-128 layer | A layer with `compress_ratios` 128 and no indexer: the odd layers 3–41 |
| Probe boot | An eager boot of the live folder with the probe and the run-time skip on. Only the client uses it |
| Skip boot | A normal boot (CUDA graphs, the DSpark draft) with a fixed skip set from the environment |

## 3. What the stack check changed

| Claim in the source | What this stack shows | Effect on the plan |
|---|---|---|
| Each layer holds about 1.6 GiB and about 2% of the bytes per step | 1.55 to 1.57 GiB of weights (§4.1). The routed experts are 1.41 GiB of that. The bytes per step are an estimate: 1.5% to 2% | Two blocks reach the 72 GiB cap. One block does not |
| Add a skip list to the layer loop | The layers use mHC. Each layer returns its FFN output unmerged, and the next layer merges it. The loop also builds the draft's input after layers 40–42. A `continue` in the loop breaks both | The skip goes inside the layer: a skipped layer returns its inputs unchanged. The half skips fit the same way (§6.1) |
| Read the list at boot, so that the CUDA graph is captured for the reduced model | The compile cache key does not include an environment variable. A boot can load the compiled graph of another skip set and run the wrong model with no error | Each boot with the patch sets `VLLM_DISABLE_COMPILE_CACHE=1`, and the run script checks the boot log (§6.1) |
| One instrumented pass over the replay set ranks all 43 layers | The replay is the held-out gate. A rank on the replay leaks the gate into the choice | The probe ranks on the calibration split (§7) |
| Leave layers 0 to 2 and the last couple out | Layers 0–2 route by hash. The draft reads the states after layers 40, 41 and 42 (`dspark_target_layer_ids`) | The candidates are layers 3–39: 19 ratio-128 layers and 18 ratio-4 layers |
| Keep the heal map as a small standalone matrix on the residual path | The state is 4 streams of 4,096 values. A map of the full state (16,384²) costs 0.5 GiB in BF16. A map that the 4 streams share (4,096²) costs 32 MiB | The heal map is shared by the streams (§9) |
| Physically remove the layer's experts and its KV share | A whole layer goes with a rewrite of the checkpoint. A removed half, or a heal map, needs a runtime patch that stays on the live server | Prefer whole layers when the costs are close (§10) |
| Single-layer scores do not add, so go greedy: remove the best, re-score, repeat | A boot for each re-score costs 10 minutes or more | The probe boot is eager and reads the skip set from a file at run time, so a re-score needs no boot (§6.2) |

## 4. The numbers

### 4.1 Memory per layer (measured)

From the safetensors headers of `data/tp1-k160t5` (2026-11-10):

| Part | Per layer | 43 layers |
|---|---|---|
| Routed experts (EXL3, 160 × 9.04 MiB) | 1,445.6 MiB | 60.71 GiB |
| Attention (FP8) | 102.0 MiB | 4.28 GiB |
| Compressor | 16.0 MiB (ratio 4), 8.3 MiB (ratio 128) | 0.49 GiB |
| Indexer | 12.5 MiB (ratio-4 layers only) | 0.26 GiB |
| Shared expert | 24.0 MiB | 1.01 GiB |
| Router gate | 1.3 MiB (6.2 MiB in layers 0–2, with the hash tables) | 0.07 GiB |
| mHC parameters | 3.0 MiB (1.5 MiB for each half) | 0.13 GiB |
| All | 1,584 MiB (ratio 128), 1,604 MiB (ratio 4) | 66.94 GiB |

| Block | Weights | KV | Bytes of a decode step (estimate) |
|---|---|---|---|
| L, ratio-4 layer | 1.57 GiB | Most of the KV share of a layer | 1.5% to 2% |
| L, ratio-128 layer | 1.55 GiB | Very small | 1.5% to 2% |
| F | 1.44 GiB | None | About 1% |
| A, ratio-4 layer | 0.13 GiB | As for L | About 0.6% |
| A, ratio-128 layer | 0.11 GiB | Very small | About 0.6% |

- **The KV.** A ratio-4 layer stores one KV entry for each 4 tokens, and its indexer stores keys too. A ratio-128
  layer stores one entry for each 128 tokens. So almost all of the KV saving comes from ratio-4 layers. The KV is
  pinned at 8 GiB, so a removed layer first gives a larger pool. Then the pin can go down (§10.3).
- **The bytes.** One verify step reads, in each layer, the attention weights and the shared expert (about 0.14 GiB).
  It also reads the routed experts that its 6 rows choose: about 20 experts, or 0.18 GiB, by estimate. The head and
  the draft add a fixed part. The noise floor of the bot decode is 4.2%, so one or two whole layers do not show in a
  speed test.

### 4.2 The NLL budget (measured)

| Model | Replay NLL | Rise against the baseline | Left under the gate |
|---|---|---|---|
| `baseline-k216` | 0.4440 | 0 | 0.0133 |
| The live folder (`prune5-k160`) | 0.4524 | +1.89% | **0.0049** (+1.08% over the live replay) |
| H1: the live folder with the FP8 head (2026-11-10, no deploy) | 0.4565 | +2.82% | 0.0008 |
| The gate | 0.4573 | +3.00% | — |

- A skip set must fit in 0.0049. H1 did not deploy (`PLAN_SERVE_TUNE.md` §7.4), so the full 0.0049 is free.
- **The price.** H1 cost +0.91% against the live replay for 0.49 GiB: 1.9% of NLL per GiB. You chose no deploy,
  because the loss was in the reasoning tokens. Two L blocks free 3.1 GiB. At the full budget (+1.08%), they cost
  0.35% of NLL per GiB. Phase C reports the price of each set, and the deploy stays your decision.
- The other gates also apply: the tool names reach at least 98% of the baseline (0.9277), and the decisions fall by
  no more than 3.
- The replay noise is 0.23% (run against repeat). A mask against a prune of the same set differs by 0.37%.
- The cold-expert cuts of `PLAN_COLD_EXPERTS.md` use the same budget.

### 4.3 The targets

- **Memory:** the cap needs 74.09 − 72 = 2.09 GiB. Two L blocks free 3.1 GiB (to about 71.0 GiB). Two F blocks free
  2.9 GiB (to about 71.2 GiB). One L block frees 1.55 GiB (to about 72.5 GiB), which is not enough.
- **Speed:** three L blocks give about +4.5% to +6% by bytes, near the noise floor. The draft acceptance can take the
  gain away (§8.3).

## 5. Order, gates and downtime

| Phase | Work | Downtime | Gate to go on |
|---|---|---|---|
| A | The patch, the probe, the client, the run scripts and the tests | None | All tests pass; the dry runs choose right |
| B | The probe window | About 3 h | The table of §7.6 |
| C | The replay window | About 4 to 5 h | The replay gate (§8.3) |
| D | Heal maps for sets | About 3 h, then about 4 h | The replay gate |
| E | The physical removal and its test boot | About 1.5 h | The match with the skip boot (§10.3) |
| F | The full runs, the judge and your blind review | As in `PLAN_FINAL_REVIEW.md` | Your decision |

CAUTION: Each window stops the bot's model server. The bot gives no answers until the run script restores it.

Rules for every window:

- Do not start a window between 05:00 and 10:00 on a weekday.
- Do not overlap a window with a window of another plan: `PLAN_SERVE_TUNE.md` (the `tune_draft.sh` window is open)
  or `PLAN_COLD_EXPERTS.md`.
- Use `run_lib.sh` for every run script.
- Compare speeds only inside one window. Each window boots its own reference point.

CAUTION: Bash reads a script while it runs. Do not edit `run_lib.sh` or a run script while a window runs.

## 6. Phase A: the patch and the tools (no downtime, about 2 days)

### 6.1 The skip patch

| File | Use |
|---|---|
| `image-patch/layer_skip.py` | The parser of the skip set, the probe sums, and the reader of the run-time skip file |
| `image-patch/layer-skip.diff` | The change in the image's `vllm/models/deepseek_v4/nvidia/model.py` |
| `image-patch/build_skip.sh` | Applies the diff to the image's model file and writes `image-patch/deepseek_v4_model_skip.py` (not in git). With `HEAD=1`, it applies `head-fp8.diff` first |
| `compose-skip.yml` | An override on top of `compose-mask.yml` and `compose-tune.yml`. It mounts the two files and passes the variables below |

How a skipped block runs inside `DeepseekV4DecoderLayer.forward`:

| Block | What the layer does |
|---|---|
| L | It returns its inputs `(x, residual, post_mix, res_mix)` unchanged. The next layer then merges the FFN output of the layer before |
| A | It does not run the attention pre and the attention. It runs the FFN pre on its inputs, with its own FFN mHC parameters, then the FFN |
| F | It runs the attention pre and the attention. Then it returns the attention output unmerged, with the mix values of the attention pre. The next layer merges it |

- The loop in `DeepseekV4Model.forward` does not change. A skipped layer still returns values, so the draft's input
  after layers 40–42 stays correct.
- The patch covers both mHC paths. (2026-11-13: the plan said that the b12x path is off on this server. That is
  wrong: the serve script sets `VLLM_USE_B12X_MHC=1` for the b12x backends, and the live server logs "b12x mHC
  enabled". The first patch refused the b12x path, so the first probe window failed at the boot, §7.7.)

The variables:

| Variable | Meaning |
|---|---|
| `LAYER_SKIP` | The skip set, for example `L17,F21`. Empty: no skip |
| `LAYER_SKIP_FILE` | Probe boot only: a JSON file with the skip set, the Gram blocks and the heal file. The patch reads it at the start of a model step when its mtime changes |
| `LAYER_PROBE` | `1`: the probe sums of §6.2. Probe boot only |
| `LAYER_HEAL` | A safetensors file with the heal maps of the skip set (§9) |

The rules in the patch:

- It refuses a block in layers 0–2 or 40–42, a block that occurs twice, and an L block together with an A or F block
  of the same layer. An A and an F block of the same layer become one L block.
- At boot, it logs one line: the skip set, the GiB of the skipped weights, and the sha256 of the heal file.
- `LAYER_SKIP_FILE` and `LAYER_PROBE` need `--enforce-eager`, because a CUDA graph does not run Python code. The patch
  stops the boot without it.

The compile cache:

- Every boot with the patch sets `VLLM_DISABLE_COMPILE_CACHE=1`. The boot then compiles the graph again. S0 of §8
  measures the extra boot time.
- The run script fails a point if the boot log shows a graph that came from the compile cache, or if the skip line
  differs from the skip set of the point.

### 6.2 The probe

In the probe boot, the patch adds these sums for the tokens of each model step:

- for each block, the sum of the cosines between the state before and after the block, and the token count. The
  blocks of layers 0–2 and 40–42 get sums too, as data;
- for each block, the sum of |Δ|² / |state|², the relative size of the change. It is a second rank;
- on request (the `gram` field of the skip file), XᵀX and XᵀΔ at up to 3 named blocks. X is the state before the
  block, Δ is the change that the block makes, and the rows are the 4 streams of each token. Each pair holds 128 MiB
  in FP32.

The states come from the kernels that run anyway. The `residual` that the attention pre returns is the state before
layer i. The `residual` that the FFN pre returns is the state after the attention of layer i. The final mHC post gives
the state after layer 42.

The patch writes the sums to `layer_probe.npz` after each step, as the capture patch writes `routing_sums.npz`. The
client reads the file before and after each request. The difference is the sums of that request.

The prefix cache:

- A cached prefix is not computed again, so it counts once, as in `capture.py`.
- A cached prefix from another skip set is wrong data. So the client calls `/reset_prefix_cache` after each change of
  the skip set. The first request after the reset must show 0 cached tokens, else the client stops.
- If the image has no reset endpoint, the probe boot runs with the prefix cache off. Then the shared prefix counts in
  every request.

### 6.3 The client: `layer_screen.py`

It imports the replay's request from the eval worktree, as `capture.py` does.

| Command | Use |
|---|---|
| `--pick` | Chooses the screen set and the probe set from the calibration split. The screen set holds about 150k prompt tokens, so that one screen takes about 3 minutes. The probe set holds about 500k tokens. The two sets do not overlap, and no replay session is in either. One target per segment, one skill after another. The screen set takes the longest target turns under 30k tokens first: on 2026-11-13 it held 6 targets in 6 skills with about 5.4k scored tokens, twice the scored tokens of a hash order |
| `--bi` | One pass over the probe set with the current skip set. It writes the BI table |
| `--screen` | The NLL of the screen set for a list of skip sets. It does the reset and the check of §6.2 at each change |
| `--greedy` | The rounds of §7.3 and §7.4 |
| `--fit` | Solves the heal map from the Gram sums (§9) and writes `heal-<block>.safetensors` |
| `--report` | The tables of §7: the BI of each block, the rise of each screened block, the sets, and the rank correlation of BI and the rise |

### 6.4 The run scripts

- `skip_probe.sh`: the window of §7. It boots the probe boot, runs `layer_screen.py --greedy`, and restores the live
  server.
- `skip_replay.sh`: the window of §8, with the point flow of `tune_replay.sh`. For each point, it boots, checks the
  boot log, runs the smoke test and the replay, applies the speed rule of §8.2, and stops the server. It keeps the
  two-failure rule and the restore of `run_lib.sh`.
- `run_lib.sh` gets one change: its memory watch also stops `layer_screen.py`. Make this change only when no window
  runs.
- `skip_probe.sh` (2026-11-13): it refuses to start when `results/skip-probe.jsonl` exists, because the client
  resumes from its log (`RESUME=1` resumes on purpose). It picks the sets if `results/skip-sets.json` is missing. If
  the eager boot fails with Model Runner V2, it boots once with Model Runner V1 (`compose-skip-v1.yml`). It checks the
  probe line of the boot log, runs the client to 15 min before `RESTORE_AT` (one rerun after a client error), stops
  the probe boot, restores the live server, and writes `results/skip-probe-report.md`.
- `skip_replay.sh` waits for Phase B. The refresh of §0 makes Phase C unlikely, so it is built only if §7.6 does not
  say "stop", before the Phase C window.

### 6.5 The tests and the gate

- `test_layer_skip.py` needs a GPU. `./test_layer_skip.sh` runs it in the server image, as `test_fp8_head.sh` does.
  It uses a stub stack of 4 layers with the real mHC kernels and random linear maps in place of the attention and the
  FFN. It checks these points:
  - an L skip gives the next layer exactly the inputs of the skipped layer;
  - an A skip and an F skip give the same output as a hand-written sequence of the same kernels;
  - a heal map of zero gives the same output as the plain skip, within the BF16 rounding;
  - the probe sums equal a PyTorch reference on random states;
  - the parser rules of §6.1.
- `test_layer_screen.py` uses a fake server, as `test_capture.py` does. It checks the set choice (no overlap, no
  replay session), the greedy rules and the stop rules, the reset check, the fit on synthetic data (it finds a known
  map), and the report.
- `bash -n` passes on both run scripts. A dry run with stub boots and stub replays handles these cases: a pass, a
  fail, a failed boot, a wrong skip line, and a graph from the compile cache.
- `test_skip_scripts.py` runs the dry run: the script sources a stub file (`SKIP_STUB`) after `run_lib.sh`. For
  `skip_probe.sh`: a normal window, the V1 boot after a failed V2 boot, two failed boots, a boot log with no probe
  line, and an old client log. The cases of `skip_replay.sh` come with it.
- `docker compose -f compose-mask.yml -f compose-tune.yml -f compose-skip.yml config` resolves.

**Gate.** All tests pass, and the dry runs choose right.

### 6.6 Pull requests

Each pull request changes at most 400 lines:

1. A1a: `layer_skip.py` (the parser, the probe sums, the skip file) and its CPU tests, `test_layer_skip.py`.
2. A1b: `layer-skip.diff`, `build_skip.sh`, `compose-skip.yml`, and the GPU tests of the patched layer with the real
   mHC kernels, `test_layer_skip_gpu.py` and `test_layer_skip.sh`.
3. A2: `layer_screen.py` and `test_layer_screen.py`, in three pull requests: A2a the data pick and the skip file,
   A2b the BI pass, the screen and the Gram sums, A2c the greedy rounds and the report.
4. A3a: `skip_probe.sh`, `compose-skip-v1.yml`, the change of `run_lib.sh`, and the dry run `test_skip_scripts.py`.
5. A3b: `skip_replay.sh` and its dry run, only if Phase B does not say "stop" (§6.4).

A1 and A2 grew past 400 lines, so they are five pull requests (2026-11-13).

If one part grows past 400 lines, split it again before you open the pull request. The heal code (§9) gets its own
pull request, and only if Phase B or C asks for it.

## 7. Phase B: the probe window (about 3 h downtime)

### 7.1 The boot

- The live folder (`MASK_NAME=prunemd5 MASK_K=149`, with `compose-perlayer.yml`) with `--enforce-eager`, no draft
  (`MODE=mtp0`), `LAYER_PROBE=1`, `LAYER_SKIP_FILE`, the prefix cache off (`PREFIX_CACHE=0`, §0 item 4), and the KV pin
  of 7.25 GiB.
- If Model Runner V2 refuses this boot, use `serve-v1.sh`, as `compose-capture.yml` does.

### 7.2 Order and times

| Order | Work | Minutes |
|---|---|---|
| 1 | Boot | 5 |
| 2 | The screen with no skip: the reference R | 3 |
| 3 | The BI pass over the probe set, with no skip | 8 |
| 4 | Round 1 (§7.3): 28 single blocks and 1 set of A blocks | 87 |
| 5 | The Gram sums of the 3 best L or F blocks of round 1 | 10 |
| 6 | Round 2 (§7.4): the sets of 2 | 26 |
| 7 | Round 3 (§7.4): the sets of 3 | 26 |
| 8 | The screen with no skip again: R2 | 3 |
| 9 | Stop the probe boot, restore the live server | 13 |
| | Sum | About 3 h |

The minutes per screen are an estimate, because the eager prefill speed is not measured. The client measures R and
drops the last screens of round 1 if the window cannot end before `RESTORE_AT`.

### 7.3 Round 1

- Screen the 10 L blocks, the 10 F blocks and the 8 A blocks with the lowest BI, each block alone.
- Then screen one set: the 6 A blocks with the lowest single rises, together.
- Both L and F blocks go to the screen. An F block frees 93% of the memory of its L block and keeps the attention. If
  F17 costs much less than L17, the attention of layer 17 matters, and F17 is the better cut.

### 7.4 Rounds 2 and 3 (the memory track)

1. Put the L or F block with the lowest rise into the set.
2. Run the BI pass with the set.
3. Screen the 6 L or F blocks with the lowest new BI, each added to the set.
4. Stop when the best set rises by more than 1% (3% before the refresh of §0; 1% is the stop row of §7.6). The Gram
   sums of §7.2 step 5 also run only when the best single L or F block rises by at most 1%, because only a heal
   uses them.

A blocks get no greedy rounds, because they free little memory.

### 7.5 The checks

- R and R2 differ by at most 0.1%. Else the screen is not stable, and its numbers do not count.
- Before the first request of each skip set, the patch log shows the new set, and the reset check of §6.2 passes.
- The report gives the rank correlation of BI and the rise in round 1. Below 0.5, BI ranks this model badly. Then a
  second probe window screens the other L and F blocks before any decision.

### 7.6 The decision after Phase B

The screen and the replay use different data. S1 of Phase C gives the first pair of numbers. Until then, use these
thresholds:

| Result of the screen | Meaning | Next step |
|---|---|---|
| The best single L or F block rises by more than 1% | No block is near the budget (0.24%, §0). A heal that halves the rise still does not fit | Stop the memory track. Record the result here and in `PRUNE.md` |
| The best single block rises by 0.25% to 1% | One block needs a heal | The heal pull request, then fit the heal map of that block (`--fit`). Phase C runs S0, S1 and S1h only |
| The best single block rises by at most 0.25% | One block can fit as it is | Phase C runs S0, S1, S1h, and S2 and S3 if each set rises by at most 0.35% |
| The set of 6 A blocks rises by at most 0.25% | The attention halves can buy speed | Add SA (the 6 A blocks) to Phase C as a speed point |

The thresholds of 2026-11-10 were 3%, 1% and 1.5% for a budget of 1.08%. The thresholds above keep the same ratios
to the budget of §0 (0.24%), rounded up.

The thresholds of the sets have a margin, because the screen and the replay use different data.

### 7.7 Result

**Attempt 1 (2026-11-13, 08:54 to 09:01, on your "start the window right now").** The live server stopped at 08:54:35.
Both probe boots failed at the model build, so no client ran. The script restored the live server, healthy at
09:01:09 (reply "OK.", gateway active). The downtime was about 7 minutes. Log: `logs/skip-probe.log`.

- The V2 boot: the patch refused the b12x mHC path (§6.1). The eager settings passed: "Using V2 Model Runner",
  CUDA graphs off.
- The V1 boot: a `TypeError` in `layer_skip.Controller`. The fix of the patch changed the mounted `layer_skip.py` while
  this boot loaded it. Do not edit a mounted file while a window runs.
- The fix (PR A1c): the skip and the probe on the b12x path too. The GPU tests run every case on both paths: 16 pass.

**Attempt 2 (2026-11-13, 09:02 to 09:13).** Both boots loaded the model (66.54 GiB) and failed in vLLM's kernel
warmup, before the KV cache: `libcudart_stub.so: undefined symbol: cudaDeviceReset`. The live server was healthy again
at 09:13:44 (reply "OK."). The downtime was about 11 minutes.

- The cause is the eager boot, not the skip. FlashInfer's `flashinfer.comm` binds the CUDA runtime at its import, and
  takes the first `libcudart` in `/proc/self/maps`. A compiled boot imports it in the compile passes, before any
  kernel runs. An eager boot runs the profile pass first. Its tilelang kernels load tilelang's `libcudart_stub.so`,
  and the warmup's import then binds that stub.
- The fix (PR A1c): `DeepseekV4Model.__init__` of an eager boot imports `flashinfer.comm` before any kernel runs. A GPU
  test reproduces the fault (a tilelang kernel first) and the fix (the import first) in subprocesses: 18 tests pass.

**Attempt 3 (2026-11-13, 09:15 to 09:23).** The probe boot was healthy after 180 s: 66.54 GiB loaded, a KV cache of
277,541 tokens, the probe line, 30 GiB available. R ran: NLL 0.5081 over 6,294 scored tokens in 120 s (about 1,200
prompt tokens per second, eager). The first request of the BI pass then had 47,360 cached prompt tokens, and the client
stopped, as §6.2 asks. The rerun stopped the same way. The live server was healthy again at 09:23:37. The downtime was
about 8 minutes. The client log of this attempt is `results/skip-probe-attempt3.jsonl`.

- The cause: the server repo's entrypoint (`entrypoint-no-download.sh`) sets `PREFIX_CACHE=1`, so the variable of
  `compose-skip.yml` had no effect. The boot ran with `--enable-prefix-caching`.
- The fix (PR A1c): `--no-enable-prefix-caching` after the script's own flag (vLLM's parser takes the last one;
  checked in the image), and a check of `enable_prefix_caching=False` in the boot log before the client starts. A new
  case of the dry run covers the check.

## 8. Phase C: the replay window (about 4 to 5 h downtime)

### 8.1 The points

All points boot the live folder with the bot draft and the live settings (as T0 of `PLAN_SERVE_TUNE.md` §6.2), with
`compose-skip.yml` and `LAYER_SKIP`.

| Order | Point | Skip set | Tests | Minutes |
|---|---|---|---|---|
| 1 | S0 | None (the patch on) | Boot check, smoke, replay, bot decode × 5, code test | 75 |
| 2 | S3 | The set of 3 | Boot check, smoke, replay, speed by §8.2 | 50 |
| 3 | S2 | The set of 2 | The same | 50 |
| 4 | S1 | The best single L or F block | The same | 50 |
| 5 | S1h | S1 with its heal map (`LAYER_HEAL`) | The same | 50 |
| 6 | SA | The 6 A blocks, only if §7.6 adds it | Boot check, smoke, replay, bot decode × 5 | 75 |
| 7 | The restore of the live server | | | 10 |

§7.6 decides which points run. The larger sets run first, so that the speed test runs on the largest set that passes,
on the boot of its replay.

### 8.2 The speed rule

- The bot decode × 5 runs on a point right after its replay, if the point passes the replay gate and no larger set
  passed before it.
- The test records the accepted tokens per step and per position. T0 had 2.20 per step.
- If no point passes, no speed test runs, except on S0 and SA.

### 8.3 The gates

- **The S0 check.** The NLL of S0 matches the mean of the live replays (0.4562, §0) within 1%. Else the patch changes
  the model with no skip. Then stop, and do not use the other points.
- **The replay gate** of `PLAN_SERVE_TUNE.md` §7.1, against `baseline-k216`: the NLL at most 0.4573, the tool names
  at least 98% of the baseline, and the decisions down by no more than 3.
- **The smoke test** passes: a JSON schema, a tool call with `auto`, and one with `required`.
- **The speed.** The bot decode of the set does not fall below S0 by more than the noise floor (4.2%). A gain counts
  only above the noise floor.
- **The acceptance.** The draft was built against the full model, and it reads the states after layers 40–42. A skip
  before them changes those states. Report the accepted tokens per step against S0.
- **The NLL split (a diagnostic).** Report the NLL of the answer tokens and of the reasoning tokens against S0, as
  `PLAN_SERVE_TUNE.md` §7.4 does for H1. H1 lost in the reasoning only.

### 8.4 How to read the result

| Result | Next step |
|---|---|
| S2 or S3 passes, and the speed holds | Your decision on the price (§4.2). Then Phase E with the largest set that passes |
| Only S1 passes | 1.55 GiB is not enough for the cap. Your decision: Phase E for one layer, or stop |
| S1h passes and S1 fails, or the heal removes at least half of the rise of S1 | The heal works. Phase D |
| No point passes, and the heal removes less than half of the rise | Stop. Record the result here and in `PRUNE.md` |
| A set passes, but the bot decode falls below S0 by more than the noise | The draft loses acceptance. The set gives memory only. Your decision |
| SA passes and its bot decode beats S0 by more than the noise | A speed point. It needs the loader patch of §10.2, so write that plan |

Also record, for each point, the rise on the replay divided by the rise on the screen. Phase D uses this ratio in
place of the margins of §7.6.

## 9. Phase D: heal maps for sets (only if §8.4 says so)

**The map.** It follows Ghosted Layers. At the boundary of a skipped block, the state s becomes s + s·M, with
M = (XᵀX + λI)⁻¹ XᵀΔ. The four streams share the map: 4,096 × 4,096 values in BF16, or 32 MiB. It costs one GEMM per
boundary per step, about 10% of the bytes of one layer.

**How the patch applies it.** After a skipped block, the patch builds the merged state with the mHC post kernel. The
draft's input path uses the same kernel. The patch applies the map, and it gives the next block a merged state. The
next block then runs the standalone mHC pre, the path for a merged input that the pipeline-parallel ranks use.

**The fit for a set.**

1. Fit the blocks one at a time, in the order of the greedy rounds.
2. Take the Gram sums of each block from a probe boot with the earlier blocks of the set skipped and healed.
3. Choose λ by the lowest error on a held-out quarter of the probe set.

**The limits.**

- The shared map cannot copy the stream mixing of the skipped block. If the shared map removes less than half of the
  rise of S1, try one map per stream (4 × 32 MiB) once.
- ReplaceMe merges its map into the MLP before the cut. That does not work here: the experts are EXL3 trellis codes,
  and a merge needs a new quantization.
- A heal map needs the patched model file on the live server permanently.

**The work.** The heal code in the patch and `--fit` in the client (one pull request). Then a probe window (about 3 h)
for the Gram sums of the sets, and a replay window (about 4 h) with the healed sets.

## 10. Phase E: the physical removal (only if a set passes)

A skip boot still loads the skipped layers. The memory falls only when the layers leave the files.

### 10.1 Whole layers: `prune_layers.py`

- It copies the live folder without the skipped layers. The kept layers get the new numbers 0 to 42 − n. Each layer
  file and the tensor names in it get the new number.
- It rewrites the entries per layer: `num_hidden_layers`, `compress_ratios`, `dspark_target_layer_ids` (40–42 become
  40 − n to 42 − n), `hybrid_tr3_tail.moe_layers`, the weight index, the tier bitmap, the manifests and the REAP plan.
- It writes a new draft folder with the new `dspark_target_layer_ids` and `num_hidden_layers`. The draft weights do
  not change.
- It does no new quantization. It copies the bytes, as `prune_experts.py` does.
- The self-check compares each kept tensor with its source, byte for byte. `--check` runs the self-check only.

Check first: how the runtime finds the MTP layer and the draft layers. If it uses `num_hidden_layers` as an index, the
new numbers must keep that index correct.

### 10.2 Halves

The runtime has no layer without an FFN or without an attention. So a removed F or A block needs a loader patch that
stays on the live server. Write that plan only if the set that passes holds halves and no set of whole layers passes.

### 10.3 The test boot (about 1.5 h downtime)

- The replay matches the skip boot of the same set within 0.37%, the noise of a mask against a prune.
- The boot log gives the GiB of the weights and the size of the KV pool. Lower the KV pin until the boot log gives
  1.02x or more for 256,000 tokens, as for #2 of `PLAN_SERVE_TUNE.md`.
- The bot decode × 5 runs against the live folder in the same window.

### 10.4 The deploy chain

The chain is the same as for the live pick: the full runs, the judge and your blind review (`PLAN_FINAL_REVIEW.md`).
The deploy follows only on your go.

## 11. Risks

| Risk | Effect | Guard |
|---|---|---|
| The compile cache loads the graph of another skip set | The point tests the wrong model, with no error | `VLLM_DISABLE_COMPILE_CACHE=1`, the boot-log check, the S0 check |
| The prefix cache keeps the KV of another skip set in the probe boot | The screen mixes two sets | The reset at each change and its check; else the prefix cache off |
| BI ranks this model badly (MoE, mHC, 3-bit experts) | The screen misses a good block | Round 1 screens 28 blocks; the rank correlation of §7.5 |
| The screen and the replay disagree | Phase C tests the wrong points | The margins of §7.6; the ratio of §8.4 |
| A skip changes the states that the draft reads | The acceptance falls, and the decode gain goes away | The acceptance per position against S0 (§8.3) |
| A rare skill, for example a tool call, loses more than the mean NLL shows | A tool call fails in the bot | The smoke test, the tool-name gate, and the full runs before a deploy |
| A set passes the gate, but its price is too high for you | No deploy, as for H1 | The price of §4.2 in the Phase C report |
| Model Runner V2 refuses an eager boot | The probe boot fails | `serve-v1.sh`, as for the capture |

## 12. Timeline

| Phase | Work | Downtime | When |
|---|---|---|---|
| A | About 2 days | None | Now: the Step 2 window of `PLAN_SERVE_TUNE.md` ended on 2026-11-10 |
| B | One evening | About 3 h | After Phase A and the deploy decision on #2 (the KV pin) |
| C | One night | About 4 to 5 h | After Phase B, if §7.6 does not say "stop" |
| D | About 1 day and two windows | About 3 h and about 4 h | Only if §8.4 says so |
| E | About 1 day and one window | About 1.5 h | Only if a set passes |
| F | As in `PLAN_FINAL_REVIEW.md` | As there | Your decision |

Phases A and B decide whether the rest is worth the work. If the result of §7.6 is "stop", the plan ends after one
window.
