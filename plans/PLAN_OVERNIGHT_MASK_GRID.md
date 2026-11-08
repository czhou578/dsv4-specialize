# Plan: Overnight mask grid, 2026-11-03 to 2026-11-04

**Status:** ready to run (second run) · **Author:** colin-spark · **Date:** 2026-11-03
**Related:** `PLAN_MODEL_SPECIALIZE.md` §5 (Phase 1: mask test) and §4.4 (speed baseline)
**Runs:** unattended, in a Claude Code session in tmux. The main server must serve on port 8888
again before the first cron job at 05:00.

**Changes after the first run** (2026-11-02, report `overnight-2026-11-03.md`; the first run measured no K):

- Every boot failed the 256k KV check at `GPU_MEMORY_UTILIZATION` 0.92 (6.4 < 6.99 GiB). The main
  `compose.yml` and `compose-mask.yml` now use 0.93. The main server gets 7.68 GiB of KV, and about
  2.7 GiB of host memory stays available (before: 4.3–4.5 GiB).
- The capture boot of §5.2 is gone: Model Runner V2 rejects `--enable-return-routed-experts`. §4.3 uses an
  indirect check on the replay results instead.
- The guard ran its 04:35 recheck a day late when the 04:15 restore took 20 minutes. The fix: the recheck
  now runs at once if 04:35 has passed.
- The five mask folders exist already.

---

## 0. Before you sleep (the user)

1. Start the guard in its own tmux session:
   `tmux new -d -s guard ~/Projects/dsv4-specialize/overnight_guard.sh`
2. Check the guard: `tail -2 ~/Projects/dsv4-specialize/overnight_guard.log` shows
   "waiting until 2026-11-04 04:15:00".
3. Start Claude Code in a second tmux session:
   `tmux new -s overnight`, then `cd ~/.hermes/stock-research` and `claude`.
4. Use a permission mode that does not stop for prompts, for example auto mode. A prompt that
   waits all night stops the grid. The guard still brings the main server back.
5. Give this prompt: *Read plans/PLAN_OVERNIGHT_MASK_GRID.md and run sections 2 to 6. Do not ask
   questions; when in doubt, take the safe action of §2.*
6. Detach with `Ctrl-b d`.

In the morning, read `~/Projects/stock-research-model-eval/model_eval/results/overnight-2026-11-04.md`.
`tmux attach -t overnight` shows the session.

---

## 1. Goal and time budget

Run Phase 1 of `PLAN_MODEL_SPECIALIZE.md` on the replay screen: K = 192, 176, 160, 144 and 128
experts per layer, at top-k 6. Also run the speed tests at K = 160 (the size cap of
`PLAN_MODEL_ROUTER.md`).

The main server boots in about 3 minutes (2026-11-01 log: 00:01:49 to 00:04:53, checksums
included). A test boot uses the same steps.

| Step | Estimate | Ends (start 22:00) |
|---|---|---|
| §3 preflight, check 5 folders | 5 min | 22:05 |
| K 192: boot, replay | 45 min | 22:50 |
| K 176: boot, replay | 45 min | 23:35 |
| K 160: boot, replay, speed tests | 80 min | 00:55 |
| K 144: boot, replay | 45 min | 01:40 |
| K 128: boot, replay | 45 min | 02:25 |
| Optional: speed tests at the smallest passing K | 40 min | 03:05 |
| §5 restore the main server | 10 min | 03:15 |

**Deadlines:**

| Time | Rule |
|---|---|
| 03:10 | Latest start of a boot with a replay run |
| 03:20 | Latest start of a speed test |
| 04:00 | Restore the main server, whatever the state of the grid |
| 04:15, 04:35 | The guard checks port 8888 and restores the server if it is down |
| 05:00 | First cron job (`daily-followups`, then `daily-report` at 05:30) |

Not in this run: Phase 4 (top-k needs a new tool), the §4.2 full runs (no tool yet) and Phase 2.

---

## 2. Rules

CAUTION: Only one model fits in memory. Two models at once can use all host memory and
hard-reset the Spark. Before each boot, `free -g` must show at least 100 GiB available, and
`docker ps` must show no running server container.

- **Main server: do not change it.** Do not edit or regenerate `start.sh`, `compose.yml`,
  `data/tp1`, `data/dspark-draft-k64` or `cache` in `~/Projects/DeepSeek-v4-Flash-One-DGX-Spark`.
  Do not run `start.sh`.
  - Stop the server with `./stop.sh`, with no argument.
  - Start it with `docker compose -f compose.yml start`.
- **The guard:** do not stop it or edit it.
- **Boots:** one boot per step. If `/health` on port 8890 does not answer within 15 minutes, run
  `down`, record the failure and go to the next K. After two failed boots in a row, stop the grid and
  go to §5.
- **Memory:** during each replay or speed run, check the free memory every minute. With a model at
  `GPU_MEMORY_UTILIZATION` 0.93, about 2.7 GiB is available. If it drops below 1 GiB, stop the client
  (`pkill -f "model_eval.replay|model_eval.speed|acceptance_c1"`).
- **Code:** change code in `model_eval/` or `~/Projects/dsv4-specialize/` only to fix a bug that
  blocks the run. Record each fix in the report.
- **Git:** do not commit, push or open pull requests.
- **Messages:** do not send messages to Discord or to other services.
- **The safe action:** if a step fails in a way that this plan does not cover, stop the tests and
  go to §5.

**Paths and commands used below:**

```bash
W=~/Projects/stock-research-model-eval          # model_eval code, branch model-eval-speed
S=~/Projects/dsv4-specialize                    # mask tool, compose-mask.yml, check_routed.py
SERVER=~/Projects/DeepSeek-v4-Flash-One-DGX-Spark
PY=~/.hermes/stock-research/.venv/bin/python
R=$W/model_eval/results
```

---

## 3. Preflight (the main server still runs)

1. Check the guard: `tmux ls` shows `guard`, and `$S/overnight_guard.log` waits for 04:15. If not,
   start it with the command of §0 step 1.
2. Check that the bot is idle: `curl -s http://127.0.0.1:8888/metrics | grep num_requests_running`
   shows 0. If a request runs, wait up to 30 minutes.
3. Check that the five folders `$S/tp1-mask-k192` to `tp1-mask-k128` exist (built 2026-11-02). If one
   is missing, build it in `$S`: `$PY mask_checkpoint.py --keep $K` (about 30 s and 12.3 GB).
   Check that `grep GPU_MEMORY $S/compose-mask.yml` shows 0.93.
4. Check each folder:
   `python3 $SERVER/image-patch/verify_tp1_manifest.py $S/tp1-mask-k$K --skip-checksums` prints
   48 tensor files.
5. Check that `$R/baseline-k216.jsonl` holds 60 rows.
6. Write the start time to the report file (§6).

---

## 4. The grid

### 4.1 Boot one K

1. If the main server runs: `cd $SERVER && ./stop.sh`.
2. Wait until `free -g` shows at least 100 GiB available (up to 3 minutes).
3. `cd $S && MASK_K=$K docker compose -f compose-mask.yml up -d`. Do not set `MASK_EXTRA_ARGS`.
4. Every 15 s, run `curl -fsS http://127.0.0.1:8890/health`, for up to 15 minutes.
5. From `docker logs dsv4-mask-deepseek-v4-flash-mask-1`, record the boot time, "Model loading took"
   and "GPU KV cache size".

### 4.2 Stop one K

`cd $S && MASK_K=$K docker compose -f compose-mask.yml down`. Then check that `docker ps` shows no
`dsv4-mask` container.

### 4.3 K = 192 and the §5.2 check

Do not run the §5.2 capture check. The image runs Model Runner V2, and V2 rejects
`--enable-return-routed-experts` ("Model Runner V2 does not yet support: routed experts capture",
2026-11-02). `check_routed.py` needs a new capture method first.

1. Boot K = 192 (§4.1), run the replay (§4.4) and stop (§4.2).
2. After K = 128, do the indirect check: if the NLL change of every K is within the noise floor
   (0.23%), the mask has no measurable effect. Record this in the report. The results then do not
   measure the prune.

The indirect check cannot show that no dropped expert runs. The hash layers 0–2 change the NLL also
when the `gate.bias` mask fails. Record "§5.2 not verified" in the report.

### 4.4 Replay of one K

```bash
cd $W
$PY -m model_eval.replay --base-url http://127.0.0.1:8890 --out $R/mask-k$K.jsonl > $R/mask-k$K.log 2>&1
$PY -m model_eval.replay --compare $R/baseline-k216.jsonl $R/mask-k$K.jsonl > $R/mask-k$K-compare.txt
```

The run scores 60 targets in about 40 minutes. A rerun with the same `--out` skips the finished
targets.

**Gates** (replay screen, `PLAN_MODEL_SPECIALIZE.md` §4.3):

- NLL at most +3%. The baseline noise is 0.23%.
- Tool names at least 98% of the baseline. The baseline has 77 of 83 tool names right, so a loss of
  one name passes, and a loss of two fails.

### 4.5 The other K values

Do K = 176, 160, 144 and 128, in this order. For each K: boot (§4.1), replay (§4.4), and stop (§4.2).
At K = 160, run the speed tests (§4.6) before the stop.

Run all five K values, also after a K fails a gate. The failures show how fast the quality drops.
The only early stops are the boot rule and the safe action of §2.

### 4.6 Speed tests of one K

1. Bot decode test (about 30 minutes). It prints its rate and its acceptance in the last log line:
   `cd $W && $PY -m model_eval.speed --base-url http://127.0.0.1:8890 --out $R/speed-bot-mask-k$K.jsonl --runs 3 > $R/speed-bot-mask-k$K.log 2>&1`
2. Code decode test, warm-up: run
   `$PY $SERVER/image-patch/acceptance_c1.py --skip-correctness --skip-prefill --base-url http://127.0.0.1:8890 --model deepseek-v4-flash-0731-mask-k$K > $R/speed-code-mask-k$K-0.json`.
   The script exits with an error when a prompt is below 35 tok/s, after it prints the JSON. Keep
   the JSON.
3. Save the counters:
   `$PY -c "import json; from model_eval.speed import counters; print(json.dumps(counters('http://127.0.0.1:8890')))" > $R/speed-code-mask-k$K-counters-before.json`
4. Run the same code test 3 more times, to `-1.json`, `-2.json` and `-3.json`.
5. Save the counters again, to `$R/speed-code-mask-k$K-counters-after.json`.
6. Compute the code rate the way §4.4 of the main plan does: all decoded tokens / all decode time
   for each run, then the median of runs 1–3. Compute the acceptance from the two counter files.
7. Compare with the baseline:
   - bot decode: 33.64 tok/s, noise floor 1.42 tok/s;
   - code decode: 38.86 tok/s, noise floor 1.65 tok/s.

   A change counts only if it is larger than the noise floor. Also compare the accepted tokens per
   draft: 2.25 for the bot test, 2.67 for the code test.

### 4.7 Optional speed point

If the time is before 03:20 after K = 128, take the smallest K that passed both replay gates. If it
is not 160, boot it, run §4.6 and stop it.

---

## 5. Restore the main server

Do this as soon as the grid ends, and at 04:00 at the latest.

1. Stop every test (§4.2). `docker ps` must show no `dsv4-mask` container.
2. Wait until `free -g` shows at least 100 GiB available.
3. `cd $SERVER && docker compose -f compose.yml start`.
4. Every 15 s, run `curl -fsS http://127.0.0.1:8888/health`, for up to 20 minutes.
5. Send one short request:
   `curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-v4-flash-0731","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"chat_template_kwargs":{"thinking":false}}'`
   It must return text.
6. Check `systemctl --user is-active hermes-gateway`: it must show `active`. Do not restart it.

If the server is not healthy after 20 minutes:

1. Save `docker compose -f compose.yml logs --tail 80` to the report.
2. Run `docker compose -f compose.yml restart` once, and wait 20 minutes again.
3. If it still fails, write the failure in the first line of the report. The guard tries again at
   04:15 and 04:35.

---

## 6. Report

Write `$R/overnight-2026-11-04.md`. Do not change `overnight-2026-11-03.md` (the first run).

1. **First line:** the main server state, for example "Main server healthy at 03:31", or the
   failure.
2. **§5.2 check:** "not verified" (§4.3), and the result of the indirect check.
3. **Grid table**, one row per K:

   | K | Dropped per layer | NLL base → new (Δ%) | Reasoning NLL | Answer NLL | Answer top-1 | Tool names | Gates | Boot time | Notes |
   |---|---|---|---|---|---|---|---|---|---|

4. **Speed table** for K = 160 and the optional point: bot rate, code rate, accepted tokens per draft,
   and the baseline values with their noise floors.
5. **Result:** the smallest K that passes both replay gates. Then apply `PLAN_MODEL_SPECIALIZE.md`
   §5.5 (stop gate) and the size cap of §4.5 (about 160 experts). No run has measured the full-run
   gates yet, so a pass is a pass of the replay screen only.
6. **Times and errors:** the start and end time of each step, failed boots, failed targets, code fixes.

Also:

- Write `$R/grid.csv` with the columns of `PLAN_MODEL_SPECIALIZE.md` §4.5: K, top-k, size, decode
  tok/s (median and spread), acceptance, NLL change, and each gate. Size = 106.8 − 0.45 × (216 − K) GB.
- In `$W/plans/PLAN_MODEL_SPECIALIZE.md`, update the §5 row of the progress table, and replace the results
  block of the first run at the end of §5.4. Copy the file to `~/.hermes/stock-research/plans/`.
- Do not commit.
