# Runbook

This page holds the operator steps of the repo: the safety rules of the run scripts, the rebuild of the live
checkpoint, one mask test by hand and the expert capture by hand. [README.md](README.md) describes each file.

## Safety rules of the run scripts

CAUTION: Each run script stops the bot's model server. The bot gives no answers until the script restores it. Do not
run a script near the `daily-report` or `daily-followups` cron jobs (05:00 to 10:00 on weekdays).

The rules come from `PLAN_OVERNIGHT_MASK_GRID.md` §2. `run_lib.sh` applies them:

- Only one model is in memory. Before each boot, no server container runs and at least 100 GiB is available.
- A boot that is not healthy in 15 minutes goes down. After two failed boots in a row, the script restores the live
  server.
- A memory watch reads the free memory every 30 seconds. Below 1 GiB, it stops the client.
- The deadlines have no defaults: `START_AFTER`, `BOOT_BY`, `SPEED_BY` and `RESTORE_AT`. At `RESTORE_AT`, every client
  stops and the restore starts.
- On any other failure, the script does the safe action: the restore.

Start a script in its own tmux session, with its deadlines:

```bash
tmux new -d -s prune-boot "START_AFTER='2026-11-06 19:05' BOOT_BY='2026-11-06 20:05' SPEED_BY='2026-11-06 20:20' RESTORE_AT='2026-11-06 21:35' ./prune_boot.sh"
```

## Rebuild the live checkpoint

The live folder `data/tp1-k160t5` came from these steps. The rank file is `results/expert_rank_bot.json` (in git).

1. Build the bot-rank mask: `python mask_checkpoint.py --keep 160 --rank <rank file> --out tp1-botmask-k160`.
2. Build the top-k copy: `python topk_checkpoint.py --source tp1-botmask-k160 --top-k 5 --out tp1-botmask5-k160`.
3. Build the pruned folder: `SRC=tp1-botmask5-k160 OUT=tp1-prune5-k160 ./prune_run.sh`. It takes about 11 minutes
   and writes about 73 GB.

Steps 1 and 2 take about a minute each and write about 12 GB each.

## One mask test by hand

1. Print the plan: `python mask_checkpoint.py --keep 192 --dry-run`.
2. Build the folder: `python mask_checkpoint.py --keep 192`. It writes about 12.3 GB. Do not build while a speed
   test runs, because the writes slow the decode.
3. Copy the kernel cache once (done 2026-11-02). Some files are `root:600`, so a root container from the server image
   copies them. Use the image of `compose-mask.yml` as `IMAGE`:

   ```bash
   mkdir cache && docker run --rm --pull=never --network none --entrypoint cp \
     -v ../DeepSeek-v4-Flash-One-DGX-Spark/cache:/src:ro -v ./cache:/dst IMAGE -a /src/. /dst/
   ```

CAUTION: The next step stops the bot's model server. The bot gives no answers until step 8.

4. Stop the live server: `docker compose -f ../DeepSeek-v4-Flash-One-DGX-Spark/compose-k160t5.yml stop`.
5. Boot the test: `MASK_K=192 docker compose -f compose-mask.yml up -d`.
   - For the `PLAN_MODEL_SPECIALIZE.md` §5.2 check, boot
     `CAPTURE_DIR=$PWD/tp1-mask-k192 docker compose -f compose-capture.yml up -d` instead.
   - Wait for `/health`, then run `python check_routed.py --keep 192`. It must print PASS.
   - Model Runner V2 rejects the capture flag, so `compose-mask.yml` cannot do this check.
6. Score the test from the bot repo's eval worktree:
   `python -m model_eval.replay --base-url http://127.0.0.1:8890 --out ~/Projects/dsv4-specialize/results/mask-k192.jsonl`.
   Then compare it with `results/baseline-k216.jsonl` (`--compare`).
7. Stop the test: `MASK_K=192 docker compose -f compose-mask.yml down`.
8. Start the live server again: `docker compose -f ../DeepSeek-v4-Flash-One-DGX-Spark/compose-k160t5.yml start`.

## Expert capture

`capture_run.sh` does these steps with the safety rules above (`PLAN_BETTER_RANK.md` §4). The manual steps are:

CAUTION: The capture boot stops the bot's model server, like a mask test.

1. Stop the live server: `docker compose -f ../DeepSeek-v4-Flash-One-DGX-Spark/compose-k160t5.yml stop`.
2. Boot the capture: `docker compose -f compose-capture.yml up -d`. It serves the stock `data/tp1` with Model Runner
   V1 and no DSpark, so the decode is slower than on the live server. The patch writes the sums of the counts and
   the router weights to `capture-out/routing_sums.npz` after each step.
3. From this repo, capture and rank:
   `python capture.py --out results/capture-rank.jsonl --data ~/Projects/stock-research-model-eval/model_eval/data/rank --sources discord,subagent,e2e,cron --sums capture-out/routing_sums.npz`,
   then `python capture.py --rank results/capture-rank.jsonl results/expert_rank_weight.json --order weight`.
   The orders are `count`, `weight`, `skill_max` and `skill_floor`.
4. Stop the capture: `docker compose -f compose-capture.yml down`. Start the live server:
   `docker compose -f compose-k160t5.yml start` in its folder.
