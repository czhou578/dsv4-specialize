# dsv4-specialize

Weight tools, test boots and run scripts that made DeepSeek V4 Flash smaller for the stock-research bot on one
DGX Spark: 160 of 216 experts per layer, 5 experts per token, 74 GiB loaded in place of 95 GiB. The plans and the
results summary are in the bot's repo, `czhou578/stock-research-agent`: `MODEL_PRUNE.md` and
`plans/PLAN_MODEL_SPECIALIZE.md`, `PLAN_OVERNIGHT_MASK_GRID.md`, `PLAN_OVERNIGHT_TOPK.md`, `PLAN_FINAL_REVIEW.md`,
`PLAN_PHYSICAL_PRUNE.md`. The scripts here refer to the plans by their section numbers.

The folder on the Spark also holds the test checkpoints (`tp1-*`, 12 to 73 GB each) and a copy of the kernel
cache. Git ignores them. The stock server in `~/Projects/DeepSeek-v4-Flash-One-DGX-Spark` does not change; the
pruned model got its own compose file there (`compose-k160t5.yml`).

The tools in the order of the work:

| File | Use |
|---|---|
| `mask_checkpoint.py` | Builds `tp1-mask-k{K}/`: K of 216 experts per layer stay (plan §5.3). With `--rank expert_rank_bot.json`, builds `tp1-botmask-k{K}/` from the bot rank (plan §6.5); boot it with `MASK_NAME=botmask` |
| `test_mask_checkpoint.py` | Unit tests on a small synthetic checkpoint |
| `compose-mask.yml` | Test boot of a mask folder on port 8890 |
| `check_routed.py` | Plan §5.2 check: no dropped expert in the routing of a test boot |
| `compose-capture.yml` | Expert capture boot on port 8890: Model Runner V1, no DSpark, `--enable-return-routed-experts` |
| `serve-v1.sh` | Runs the server's serve script with Model Runner V1 (for `compose-capture.yml` only) |
| `test_check_routed.py` | Unit tests on synthetic routing data |
| `overnight_guard.sh` | At 04:15 and 04:35, restores the main server (`MAIN_COMPOSE`, default `compose-k160t5.yml`) if port 8888 is down (`DRY_RUN=1` to test) |
| `topk_checkpoint.py` | Builds a top-k copy of a tp1 folder (plan §8): `--source tp1-botmask-k160 --top-k 5 --out tp1-botmask5-k160`; boot it with `MASK_NAME=botmask5 MASK_K=160`. `--check` runs the self-check only |
| `test_topk_checkpoint.py` | Unit tests on a small synthetic checkpoint |
| `speed_result.py` | The plan §4.4 speed result of one test point, from the files in `model_eval/results` |
| `topk_run.sh` | The test boots of `PLAN_OVERNIGHT_TOPK.md` §5 and the restore of §6, with its deadlines |
| `prune_experts.py` | Builds a physically pruned copy of a mask folder (`PLAN_PHYSICAL_PRUNE.md` §3): `--source tp1-botmask5-k160 --out tp1-prune5-k160`. The dropped experts leave the files, and the kept experts get the new ids 0 … K−1. `--check` runs the self-check only |
| `test_prune_experts.py` | Unit tests on a small synthetic checkpoint |
| `prune_run.sh` | Runs `prune_experts.py` in a root container from the server image, because the 43 expert files of `data/tp1` are `root:600`. The main server stays up |
| `prune_boot.sh` | The test boot of `PLAN_PHYSICAL_PRUNE.md` §6: the pruned folder on port 8890, the replay set, the speed tests, then the restore |
| `compose-mem073.yml` | Override for `compose-mask.yml`: `GPU_MEMORY_UTILIZATION` 0.73 and 128k context, the DeepSeek footprint for two models (`PLAN_MODEL_ROUTER.md` §3) |
| `mem_boot.sh` | The memory test boot with that override: the boot lines, the free memory at idle and during one bot decode run, then the restore |
| `image-patch/` | `base_router.py` and `routed_experts_capturer.py` for the capture container (`PLAN_BETTER_RANK.md` §4): the router hands the weights to the capturer, which sums the counts and router weights per layer and expert and writes `capture-out/routing_sums.npz` after each step. `orig/` holds the unpatched files of the image; `diff -u orig/X X` shows the change |
| `capture_run.sh` | The capture boot of `PLAN_BETTER_RANK.md` §4: the stock model on port 8890 with the weights patch, a smoke test of the sums file, the capture of `model_eval/data/rank`, the three rank files, then the restore. Needs the deadlines set, like `prune_boot.sh` |
| `rank_masks.sh` | Builds the three mask folders of `PLAN_BETTER_RANK.md` §6 from the stock model: the floor rank at K 160 and K 144, the count rank at K 144 (`RANK_NEW` picks the rank file) |
| `rank_night.sh` | The test night of `PLAN_BETTER_RANK.md` §6: the three masks on the replay screen, then the restore. Needs the deadlines set |
| `rank_night2.sh` | One more point for §6: the 40 k floor at K 144 (a copy of `rank_night.sh` with one point) |
| `draft_test.sh` | The draft test of `PLAN_PHYSICAL_PRUNE.md` §7: the live pruned model on port 8890 with another DSpark draft (`DRAFT_DIR`, also a variable of `compose-mask.yml`), the two decode tests, then the restore |

Run the Python files with the stock-research venv: `~/.hermes/stock-research/.venv/bin/python`.

The run scripts (`topk_run.sh`, `prune_boot.sh`, `mem_boot.sh`) share the safety rules of `PLAN_OVERNIGHT_MASK_GRID.md`
§2: one model in memory, at least 100 GiB available before a boot, `restart: on-failure:1`, a memory watch that stops
the client below 1 GiB, deadlines, and the restore of the main server as the safe action. Start one in its own tmux
session with its deadlines set, for example
`tmux new -d -s prune-boot "START_AFTER='2026-11-06 19:05' BOOT_BY='2026-11-06 20:05' SPEED_BY='2026-11-06 20:20' RESTORE_AT='2026-11-06 21:35' ./prune_boot.sh"`.

## One mask test

The main server is the pruned model since 2026-11-06: `compose-k160t5.yml` in the server folder (container
`deepseek-v4-flash-k160t5-deepseek-v4-flash-1`). `start.sh` and `stop.sh` there handle the stock model only.

1. Print the plan: `python mask_checkpoint.py --keep 192 --dry-run`.
2. Build the folder: `python mask_checkpoint.py --keep 192`. It writes about 12.3 GB. Do not build
   while a speed test runs, because the writes slow the decode.
3. Copy the kernel cache once (done 2026-11-02). Some files are `root:600`, so a root container
   from the server image copies them: `mkdir cache && docker run --rm --pull=never --network none
   --entrypoint cp -v ../DeepSeek-v4-Flash-One-DGX-Spark/cache:/src:ro -v ./cache:/dst IMAGE -a /src/. /dst/`
   with IMAGE as in `compose-mask.yml`.

CAUTION: The next step stops the bot's model server. The bot gives no answers until step 8.

4. Stop the main server: `docker compose -f ../DeepSeek-v4-Flash-One-DGX-Spark/compose-k160t5.yml stop`. Do not stop it near the
   `daily-report` or `daily-followups` cron jobs.
5. Boot the test: `MASK_K=192 docker compose -f compose-mask.yml up -d`. For the plan §5.2 check, boot
   `CAPTURE_DIR=$PWD/tp1-mask-k192 docker compose -f compose-capture.yml up -d` instead, wait for
   `/health`, and run `python check_routed.py --keep 192`. It must print PASS. Model Runner V2 rejects
   the capture flag, so `compose-mask.yml` cannot do this check.
6. Score it from the stock-research repo:
   `python -m model_eval.replay --base-url http://127.0.0.1:8890 --out model_eval/results/mask-k192.jsonl`,
   then `--compare` against `baseline-k216.jsonl`.
7. Stop the test: `MASK_K=192 docker compose -f compose-mask.yml down`.
8. Start the main server again: `docker compose -f ../DeepSeek-v4-Flash-One-DGX-Spark/compose-k160t5.yml start`.

## The mask

- Layers 3–42: `gate.bias = -1e4` for each dropped expert. The router never chooses it. The kept
  experts keep their choice and their weights.
- Layers 0–2 (hash routing): each `tid2eid` entry of a dropped expert moves to the kept expert with
  the most similar router row. That expert must not be in the token's row yet, and its load must be
  below twice the mean.
- `MASK.json` in each folder records the dropped experts (current and original ids) and the rule.

## Expert capture (plan §6.2)

CAUTION: The capture boot stops the bot's model server, like a mask test.

1. Stop the main server: `docker compose -f ../DeepSeek-v4-Flash-One-DGX-Spark/compose-k160t5.yml stop`.
2. Boot the capture: `docker compose -f compose-capture.yml up -d`. It serves the base `data/tp1` on port
   8890 with Model Runner V1 and no DSpark (`serve-v1.sh`), so the decode is slower than the main server.
3. From the stock-research-model-eval repo:
   `python -m model_eval.capture --out model_eval/results/capture-k216.jsonl`, then
   `python -m model_eval.capture --rank model_eval/results/capture-k216.jsonl model_eval/results/expert_rank_bot.json`.
4. Stop the capture: `docker compose -f compose-capture.yml down`. Start the main server:
   `docker compose -f compose-k160t5.yml start` in its folder.
