# dsv4-specialize

Weight tools, test boots and run scripts that made DeepSeek V4 Flash smaller for the stock-research bot on one
DGX Spark. The result runs since 2026-11-06: 160 of 216 routed experts per layer, 5 experts per token, and 74 GiB
loaded in place of 95 GiB. An extension on 2026-11-09 and 2026-11-10 tested fewer experts per token with a router
patch (`PRUNE.md` §6, `plans/PLAN_FEWER_EXPERTS.md`). A round on 2026-11-10 tuned the serving settings of the live
server; it has its own folder, `serve-tune/`, with its plan, tools, results and a write-up (`serve-tune/README.md`). A test on the night of 2026-11-10 asked
whether the bot needs its coldest experts (`plans/PLAN_COLD_EXPERTS.md`). After a second window on 2026-11-11, the
answer is no: 486 cold experts (4.29 GiB after a prune) have no measurable cost on the replay.

Start with `PRUNE.md`: why, the grid, the final review, the physical prune and the open items. The details are in
`plans/`: `PLAN_OVERNIGHT_MASK_GRID.md`, `PLAN_OVERNIGHT_TOPK.md`, `PLAN_FINAL_REVIEW.md`, `PLAN_PHYSICAL_PRUNE.md` and
`PLAN_BETTER_RANK.md`. They moved here from the bot's repo on 2026-11-08; `plans/README.md` tells how to read their
paths. The scripts here refer to the plans by their section numbers.

Three documents stay in the bot's repo, `czhou578/stock-research-agent`: `MODEL_PRUNE.md` (what runs now, how to go
back to the stock model, how the harness measures a model), `plans/PLAN_MODEL_SPECIALIZE.md` (the method and the
harness) and `plans/PLAN_E2E_RUNNER.md` (the full runs). The eval harness, `model_eval/`, also stays there.

The live server is not in this repo. It is `compose-k160t5.yml` in `~/Projects/DeepSeek-v4-Flash-One-DGX-Spark`
(container `deepseek-v4-flash-k160t5-deepseek-v4-flash-1`). `start.sh` and `stop.sh` there control the stock model
only.

## Files

Run the Python files with the stock-research venv: `~/.hermes/stock-research/.venv/bin/python`.

**Weight tools.** Each tool writes a new folder and never changes its source.

| File | Use |
|---|---|
| `mask_checkpoint.py` | Builds `tp1-mask-k{K}/`: K of 216 experts per layer stay (`PLAN_MODEL_SPECIALIZE.md` §5.3). With `--rank <rank file>`, the bot rank decides the dropped experts (§6.5); the folder is then `tp1-botmask-k{K}/`. With `--keep-per-layer FILE` (a JSON list from `serve-tune/alloc_k.py`), each layer keeps its own K; `--out` is then needed. With `--drop-file FILE` (from `cold_sets.py`), it drops exactly the listed ids, a mandatory expert too |
| `cold_sets.py` | The drop files of `plans/PLAN_COLD_EXPERTS.md` §4: R0 (the live set as a mask), M1 (cold tool experts), D1 (cold ordinary experts) and MD (both). `--report R0.jsonl POINT.jsonl` compares a replay with R0 (§5.2) |
| `topk_checkpoint.py` | Builds a top-k copy of a tp1 folder (§8): `--source tp1-botmask-k160 --top-k 5 --out tp1-botmask5-k160`. `--check` runs the self-check only |
| `prune_experts.py` | Builds a physically pruned copy of a mask folder (`PLAN_PHYSICAL_PRUNE.md` §3): the dropped experts leave the files, and the kept experts get the new ids 0 … K−1. A mask with a different K in each layer gives a folder for the loader patch (`plans/PLAN_LOADER_PATCH.md` §4.1). `--check` runs the self-check only |
| `prune_run.sh` | Runs `prune_experts.py` in a root container from the server image, because the 43 expert files of `data/tp1` are `root:600`. The main server stays up |
| `check_routed.py` | The `PLAN_MODEL_SPECIALIZE.md` §5.2 check: no dropped expert in the routing of a test boot |
| `speed_result.py` | The §4.4 speed result of one test point, from the files in `results/` |
| `fewer_report.py` | The replay table of the points of `plans/PLAN_FEWER_EXPERTS.md` §7 against the baseline, and the decision for Phase D (`--decide`) |

**Rank tools.** They need the bot repo's eval worktree, `~/Projects/stock-research-model-eval`: `capture.py` and
`rank_data.py` import its `model_eval/replay.py`, and the rank data is its `model_eval/data/rank/` (out of git). They
moved here from the bot repo's `model_eval/` on 2026-11-08.

| File | Use |
|---|---|
| `capture.py` | The client of the capture boot (`PLAN_MODEL_SPECIALIZE.md` §6.2, `plans/PLAN_BETTER_RANK.md` §4): per segment, the counts of the routed experts and, with `--sums`, the router weights. `--rank` writes `expert_rank_<order>.json` |
| `rank_data.py` | Merges the exports into `model_eval/data/rank/sessions.jsonl` and keeps the held-out sessions out (`plans/PLAN_BETTER_RANK.md` §3) |
| `profile_report.py` | The routing profile of a capture with `ROUTING_PROFILE=1` (`plans/PLAN_FEWER_EXPERTS.md` §5): the share of the 5th and 6th experts per layer and skill, the experts per token and the window union of each drop rule, and the threshold for point C5 |
| `rank_report.py` | Compares the rank files in `results/`: the routing share that each K step removes, and the overlaps (`plans/PLAN_BETTER_RANK.md` §5) |

**Test boots.** Each compose file serves on port 8890, so the bot (port 8888) cannot reach a test model.

| File | Use |
|---|---|
| `compose-mask.yml` | Boots `tp1-${MASK_NAME}-k${MASK_K}`. `DRAFT_DIR` mounts another DSpark draft; `MASK_EXTRA_ARGS` adds vLLM flags |
| `compose-mem073.yml` | Override for `compose-mask.yml`: `GPU_MEMORY_UTILIZATION` 0.73 and 128k context (`PLAN_MODEL_ROUTER.md` §3) |
| `compose-drop.yml` | Override for `compose-mask.yml`: the router patch with its rule (`MOE_DROP`, `MOE_NORM`, `MOE_LAYERS`) and an 8 GiB KV cache (`plans/PLAN_FEWER_EXPERTS.md` §6) |
| `compose-capture.yml` | The expert capture: the stock model with Model Runner V1, no DSpark, `--enable-return-routed-experts` and the weights patch. `ROUTING_PROFILE=1` adds the routing profile; `CAPTURE_UTIL` and `CAPTURE_EXTRA_ARGS` change the boot check and add vLLM flags |
| `serve-v1.sh` | Runs the server's serve script with Model Runner V1 (for `compose-capture.yml` only) |
| `image-patch/` | The weights patch (`PLAN_BETTER_RANK.md` §4.1): the patched `base_router.py` and `routed_experts_capturer.py`, and `capture-weights.diff`, the change against the image. `patch -R -p1` with the diff gives back the original files. The router patch (`PLAN_FEWER_EXPERTS.md` §6): `moe_drop.py` (the drop rule and the gain anchor) and `router-drop.diff`; `build_router.sh` applies the diff to the image's router file (not in git) |

**Run scripts.** Each script stops the live server, runs its test boots and restores the live server. Read the
safety rules in [RUNBOOK.md](RUNBOOK.md) before you start one.

| File | Use |
|---|---|
| `run_lib.sh` | The shared functions and safety rules of the scripts below. A script sets `LOG`, `RESTORE_TAG` and, if it needs them, `TEST_COMPOSE`, `TEST_PROJECT` and `MASK`, then sources this file |
| `topk_run.sh` | The top-k test night (`PLAN_OVERNIGHT_TOPK.md` §5 and §6) |
| `prune_boot.sh` | The test boot of the pruned folder: replay, speed tests (`PLAN_PHYSICAL_PRUNE.md` §6) |
| `mem_boot.sh` | The memory test boot at 0.73 and 128k: boot lines, free memory at idle and under load (`PLAN_MODEL_ROUTER.md` §3) |
| `draft_test.sh` | The pruned model with another DSpark draft (`DRAFT_DIR`): the two decode tests (`PLAN_PHYSICAL_PRUNE.md` §7) |
| `capture_run.sh` | The capture boot (`PLAN_BETTER_RANK.md` §4): a smoke test of the sums file, the capture of `model_eval/data/rank`, the rank files |
| `rank_masks.sh` | Builds the mask folders of `PLAN_BETTER_RANK.md` §6 from the stock model (`RANK_NEW` picks the rank file). No downtime |
| `rank_night.sh` | Masks on the replay screen (`PLAN_BETTER_RANK.md` §6). `POINTS` lists them as `NAME:K`; the default is the three masks of §6.1 |
| `fewer_night.sh` | The test night of `plans/PLAN_FEWER_EXPERTS.md`: the routing profile (Phase A), then the replay points C1–C5 and C0 on `tp1-botmask-k160` with `compose-drop.yml` (Phase C), then 5 bot decode runs on the chosen point and 5 on the live server (Phase D). `BOOT_MIN_GIB` makes each boot wait for that much available memory |
| `fewer_stepf.sh` | Step F of `plans/PLAN_FEWER_EXPERTS.md` §9 on one point (`POINT`, `MOE_DROP`, `MOE_NORM`; `MASK_NAME=prune5` boots the live folder): optional replay (`REPLAY=1`) and speed tests (`SPEED=1`, gate `SPEED_MIN`), the 19 held-out prompts end to end, the restore, the checks, the judge and the blind page |
| `cold_replay.sh` | The replay window of `plans/PLAN_COLD_EXPERTS.md` §7: the masks R0, M1, D1 and MD (`cold_sets.py`), each with the smoke requests, the replay and the comparisons against R0 and the baseline. It stops after R0 if R0 does not match the earlier K 160 mask. The test boots pin the KV at 7.25 GiB |
| `overnight_guard.sh` | At 04:15 and 04:35, restores the live server (`MAIN_COMPOSE`, default `compose-k160t5.yml`) if port 8888 is down. `DRY_RUN=1` logs the actions only |

**Tests.** `test_mask_checkpoint.py`, `test_topk_checkpoint.py` and `test_prune_experts.py` run each tool on a small
synthetic checkpoint; `test_check_routed.py` uses synthetic routing data; `test_capture.py` and `test_rank.py` use a fake server and
synthetic sums. Run all of them with `python -m unittest`. `test_profile_report.py` uses synthetic profile rows. `test_serve_tune.py` runs the tests of `serve-tune/` (`serve-tune/README.md`). `test_cold_sets.py` uses a synthetic plan and rank. `test_moe_drop.py` (the router patch and the profile hook) needs torch and vLLM: `./test_moe_drop.sh` runs it in the server image.

**Not in git.** The test checkpoints (`tp1-*`), the kernel cache copy (`cache/`), the capture output (`capture-out/`), the wide drafts (`dspark-draft-*/`),
the result files (`results/`, except `results/expert_rank_bot.json`), the logs (`logs/`) and `history/hermes-e2e/`: the
step scripts of the full runs and their logs, moved from `~/.hermes-e2e/` on 2026-11-08. The step scripts keep their old
paths, so they do not run as they are.

## Run the scripts

CAUTION: Each run script stops the bot's model server. Read the safety rules in [RUNBOOK.md](RUNBOOK.md) before you
start a script.

[RUNBOOK.md](RUNBOOK.md) holds the operator steps:

- The safety rules of the run scripts, and how to start a script with its deadlines.
- The rebuild of the live checkpoint.
- One mask test by hand.
- The expert capture by hand.

## The mask

- Layers 3–42: `gate.bias = -1e4` for each dropped expert. The router never chooses it. The kept experts keep their
  choice and their weights.
- Layers 0–2 (hash routing): each `tid2eid` entry of a dropped expert moves to the kept expert with the most similar
  router row. That expert must not be in the token's row yet, and its load must be below twice the mean.
- `MASK.json` in each folder records the dropped experts (current and original ids) and the rule.

## License

MIT. See [LICENSE](LICENSE).
