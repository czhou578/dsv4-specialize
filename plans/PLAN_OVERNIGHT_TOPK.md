# Plan: Overnight top-k test, 2026-11-04 to 2026-11-05

**Status:** ready to run · **Author:** colin-spark · **Date:** 2026-11-04
**Related:** `PLAN_MODEL_SPECIALIZE.md` §8 (Phase 4) and §4.5 (the grid); `PLAN_OVERNIGHT_MASK_GRID.md` §2
(rules) and §4.6 (speed tests); `~/.hermes-e2e/stepf-speed.sh` (tonight's speed run on bot K 160)
**Runs:** unattended, in a Claude Code session in tmux. The main server must serve on port 8888 again
before the first cron job at 05:00 (Thursday).

---

## 0. Before you sleep (the user)

1. Start a second guard for the morning, in its own tmux session:
   `tmux new -d -s guard2 'GUARD_AT=04:15 RECHECK_AT=04:35 ~/Projects/dsv4-specialize/overnight_guard.sh'`
2. Check it: `tail -1 ~/Projects/dsv4-specialize/overnight_guard.log` shows
   "waiting until 2026-11-05 04:15:00". The first guard (session `guard`, 23:19 and 23:39) stays.
3. Start Claude Code in a new tmux session:
   `tmux new -s overnight-topk`, then `cd ~/.hermes/stock-research` and
   `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1 claude`. Without the variable, Claude Code's memory check can
   stop a long replay job (it did on 2026-11-03).
4. Use a permission mode that does not stop for prompts, for example auto mode.
5. Give this prompt: *Read plans/PLAN_OVERNIGHT_TOPK.md and run sections 2 to 7. Do not ask
   questions; when in doubt, take the safe action of §2.*
6. Detach with `Ctrl-b d`.

In the morning, read `~/Projects/stock-research-model-eval/model_eval/results/overnight-2026-11-05.md`.

---

## 1. Goal and time budget

The bot-rank mask with K = 160 passes the replay screen and the full-run gates, but it saves no time:
each token still uses 6 experts. Phase 4 is the only change in the plan with an expected speed gain
above the 4.2% noise floor (§3.2: +9% for top-k 5, +21% for top-k 4). Tonight answers two questions:

1. How much faster is top-k 5 (and top-k 4) on the bot K 160 mask?
2. What does it cost on the replay screen?

| Time | Step |
|---|---|
| 22:04–23:04 | Tonight's speed run on bot K 160 (top-k 6) runs; it restores the main server at the end. **Do not touch the server** |
| now–23:05 | §3: write the top-k tool and its tests (code only, no large file writes) |
| 23:05–23:45 | §4: record the K 160 speed result; build the top-k folders; check them |
| 23:45 | Earliest server stop: the first guard checks at 23:19 and 23:39 and must find the main server up |
| 23:45–01:10 | §5.1: top-k 5 on bot K 160: boot, replay, speed tests |
| 01:10–02:35 | §5.2: top-k 4 on bot K 160, or bot K 144 (top-k 6) |
| 02:35–03:20 | §5.3 (optional): top-k 5 on K 216, replay only |
| by 03:50 | §6: main server back on port 8888 |
| 04:15, 04:35 | The second guard checks port 8888 |

**Deadlines:**

- No boot with a replay after 03:00.
- No speed test after 03:05.
- The restore starts at 03:45 at the latest, whatever the state.

---

## 2. Rules

All rules of `PLAN_OVERNIGHT_MASK_GRID.md` §2 apply. The most important ones:

CAUTION: Only one model fits in memory. Two models at once can use all host memory and hard-reset
the Spark. Before each boot, `free -g` must show at least 100 GiB available, and `docker ps` must show
no running server container.

- **Tonight's speed run:** do not stop it. Do not start a server step before 23:40. Also wait until
  `~/.hermes-e2e/stepf-speed.log` shows "speed tests done" (or an ERROR line).
- **Main server:** do not change it. Do not edit `start.sh`, `compose.yml`, `data/tp1`,
  `data/dspark-draft-k64` or `cache`. Do not run `start.sh`. Stop it with `./stop.sh`. Start it with
  `docker compose -f compose.yml start`.
- **Guards:** do not stop or edit them.
- **Boots:** if a boot is not healthy in 15 minutes, run `down`, record it and go on. After two failed boots
  in a row, stop and go to §6.
- **Memory:** during each run, check the free memory every 30 s. If it drops below 1 GiB, stop the client.
  `stepf-speed.sh` has the pattern.
- **Code:** new code goes to `~/Projects/dsv4-specialize/`. Change other code only to fix a bug that
  blocks the run, and record it.
- **Git:** do not commit, push or open pull requests.
- **Messages:** do not send messages to Discord or to other services.
- **The safe action:** on a failure that this plan does not cover, stop the tests and go to §6.

```bash
W=~/Projects/stock-research-model-eval   # model_eval
S=~/Projects/dsv4-specialize             # mask tools, compose-mask.yml
SERVER=~/Projects/DeepSeek-v4-Flash-One-DGX-Spark
PY=~/.hermes/stock-research/.venv/bin/python
R=$W/model_eval/results
```

---

## 3. Write the top-k tool (code only)

Write `$S/topk_checkpoint.py`. Reuse `header`, `read_tensor` and `sha256_file` from `mask_checkpoint.py`.

```
python topk_checkpoint.py --source tp1-botmask-k160 --top-k 5 --out tp1-botmask5-k160
```

1. **config.json:** write an edited copy with `num_experts_per_tok` = top-k. Change nothing else.
2. **tid2eid (layers 0–2):** vLLM sizes the table as (vocab, `num_experts_per_tok`), so cut each row
   from 6 to top-k ids. Drop the ids with the lowest estimated router score for that token:
   - score[t, e] = `gate.weight[e]` · (`embed.weight[t]` × `ffn_norm.weight`) of the same layer, all
     BF16 → F32. `sqrtsoftplus` is monotonic. The RMS norm scales the scores of a token by one positive
     number. Thus the order of the logits is the order of the router scores;
   - this uses the token embedding in place of the real FFN input (embedding plus attention, mixed by
     mHC). It is an estimate; record it as such;
   - keep the kept ids in their old column order.
   - **Memory:** with the main server up, only about 2 GiB is free. Read `embed.weight` in chunks of
     4,096 rows by file offset (32 MB BF16 each). Do not load the full 1 GB tensor.
3. **carried-001.safetensors:** its `tid2eid` tensors change shape, so rewrite the whole file:
   - a new header with the new shapes and data offsets, in the same tensor order, with `__metadata__`
     kept;
   - pad the header with spaces to a multiple of 8 bytes;
   - stream every other tensor's bytes unchanged from the source file.
4. **Other files:**
   - a symlink in the source becomes a symlink to its resolved target (in `data/tp1`, which the
     container mounts);
   - a real file in the source (the mask's `carried-002`, `carried-003`) becomes a **hardlink**
     (`os.link`). Do not symlink to the source folder: the container does not mount it.
5. **Manifest:** copy `rank-sliced-tp1-manifest.json` with the new `bytes` and `sha256` of
   `carried-001`. Copy `MASK.json`.
6. **TOPK.json:** write these values:
   - the top-k and the source folder;
   - the drop rule (with "estimate from the token embedding");
   - for each hash layer, the share of rows in which the dropped id is also the least-used of the six
     in `expert_rank_bot.json` (a diagnostic).
7. **Self-check after the write:**
   - read back each `tid2eid`: shape (129280, top-k), every row a subset of the source row, no
     duplicates added;
   - every other tensor of `carried-001` has the same bytes as in the source (compare by hash);
   - `verify_tp1_manifest.py --skip-checksums` passes;
   - `sha256sum carried-001` matches the manifest.
8. **Tests:** `$S/test_topk_checkpoint.py` on a small synthetic checkpoint, in the style of
   `test_mask_checkpoint.py`: the drop rule, the header rewrite (offsets, alignment, other tensors
   unchanged), hardlinks and symlinks, and the manifest. Run all tests in `$S`.

Do steps 1–8 while tonight's speed run goes on. Do not build the real folders before 23:05: a 4.3 GB
write slows the decode that the speed run measures.

---

## 4. After the speed run (23:05–23:45)

1. Wait for "speed tests done" in `~/.hermes-e2e/stepf-speed.log`, and check that port 8888 is healthy.
2. **The K 160 speed result** (top-k 6): read it from `$R/speed-bot-botmask-k160.log` (last line), from
   `$R/speed-code-botmask-k160-{1,2,3}.json` and from the two counter files. Compute it as in
   `PLAN_MODEL_SPECIALIZE.md` §4.4. Compare with the baseline: bot 33.64 ± 1.42 tok/s, accepted tokens per draft
   2.25; code 38.86 ± 1.65 tok/s, 2.67. This is the reference for the top-k steps.
3. Build the folders, one at a time:
   - `$PY topk_checkpoint.py --source tp1-botmask-k160 --top-k 5 --out tp1-botmask5-k160`
   - `$PY topk_checkpoint.py --source tp1-botmask-k160 --top-k 4 --out tp1-botmask4-k160`
   - `$PY topk_checkpoint.py --source $SERVER/data/tp1 --top-k 5 --out tp1-top5-k216` (for §5.3)
4. Check each folder with the self-check of §3 step 7.

---

## 5. Test boots

**Boot, stop and run:** as in `stepf-speed.sh`, with `MASK_NAME` and `MASK_K` set:

| Folder | `MASK_NAME` | `MASK_K` | Served name |
|---|---|---|---|
| `tp1-botmask5-k160` | `botmask5` | 160 | `deepseek-v4-flash-0731-botmask5-k160` |
| `tp1-botmask4-k160` | `botmask4` | 160 | `deepseek-v4-flash-0731-botmask4-k160` |
| `tp1-top5-k216` | `top5` | 216 | `deepseek-v4-flash-0731-top5-k216` |

- Stop the main server only when the bot is idle (`stepf-speed.sh` step 1).
- In the boot log, check that the model loads with the new top-k: no shape error on `tid2eid`, and the
  MoE scratch line, if any, shows the new top-k. A boot error is a result: record it, and go on.
- **Replay:** run `$PY -m model_eval.replay --base-url http://127.0.0.1:8890 --out $R/<label>.jsonl`. Then
  run `--compare` against `baseline-k216.jsonl`, and also against `botmask-k160.jsonl` (the top-k effect
  alone).
- **Speed:** run the bot decode test (`--runs 3`) and the code decode test (warm-up and 3 runs, counters
  before run 1 and after run 3), as in `stepf-speed.sh` step 3.

### 5.1 Top-k 5 on bot K 160

Boot, replay, speed tests, stop. Run the speed tests also when the replay fails. The speed gain tells if
a better drop rule or a fine-tune is worth the work.

### 5.2 Next point

- If top-k 5 passes both replay gates: top-k 4 on bot K 160 (boot, replay, speed tests, stop).
- If it fails: bot K 144 at top-k 6, replay only (the size axis). Build it first with
  `$PY mask_checkpoint.py --keep 144 --rank $R/expert_rank_bot.json --out $S/tp1-botmask-k144`. Boot it with
  `MASK_NAME=botmask MASK_K=144`.

### 5.3 Optional: top-k 5 on K 216

Run this point only if the replay can start before 03:00. Run the replay only. It separates the top-k
effect from the mask.

---

## 6. Restore the main server

Do the steps of `PLAN_OVERNIGHT_MASK_GRID.md` §5:

1. Stop the test.
2. Wait for 100 GiB free.
3. Run `docker compose -f compose.yml start` in `$SERVER`.
4. Wait for `/health`.
5. Send one short request.
6. Check that `hermes-gateway` is active.

If the restore fails, record the failure in the first line of the report. The second guard tries again at
04:15 and 04:35.

---

## 7. Report

Write `$R/overnight-2026-11-05.md`:

1. **First line:** the main server state ("healthy at 03:12") or the failure.
2. **Speed table:** baseline K 216 top-k 6; bot K 160 top-k 6 (tonight's run); each top-k point. Columns:
   bot rate, code rate, accepted tokens per draft (both tests), change against the baseline, and
   whether the change is above the noise floor.
3. **Replay table:** each point against the baseline and against bot K 160: NLL change, reasoning and
   answer NLL, answer top-1, tool names, gates.
4. **Boot notes:** load time, KV cache size, any top-k error.
5. **Result:** the fastest point that passes both replay gates, and the next step. If no top-k point
   passes, write that. Then the plan makes the model smaller, but not faster (§4.5), unless a better drop
   rule or a fine-tune helps.
6. **Times and errors.**

Also do these steps:

- Add rows to `$R/grid.csv`.
- In `$W/plans/PLAN_MODEL_SPECIALIZE.md`, update the progress rows of §4.4/§4.5 and Phase 4, and add a
  short results block to §8.
- Copy that file to `~/.hermes/stock-research/plans/`.
- Do not commit.
