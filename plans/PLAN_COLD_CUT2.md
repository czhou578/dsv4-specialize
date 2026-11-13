# PLAN_COLD_CUT2: the second cold window, S15 and S20

**Status:** 2026-11-12 22:40: proposal. The tools are ready, the two folders are built, and the window waits for your go.
**Author:** colin-spark · **Related:** `PLAN_COLD_EXPERTS.md` (the first two windows, the points R0 to MD, the gate of
§5), `PLAN_LOADER_PATCH.md` (the prune of a point that passes), `PLAN_MD_DEPLOY.md` (the live server since 2026-11-12).

## 1. The question

The live server runs the MD prune since 2026-11-12: 486 cold experts less than the K 160 set, 4.18 GiB freed, no
measurable cost. Both cold windows said that the cut can grow (`PLAN_COLD_EXPERTS.md` §5.3, last row). This window
asks how far. It tests two larger sets of cold experts, each a superset of MD, with the replay gate of the first windows.

## 2. The points

Each point drops the live drops of the K 160 set, plus a set of cold experts in layers 3–42. The weight share comes from
the rank file of the live set, `results/expert_rank_k160.json`. Layers 0–2 (hash routing) keep the live drops only.

| Point | Rule | Experts | Mandatory | GiB | Weight share lost per layer, mean (max) | K in layers 3–42 | Mean K | Folder |
|---|---|---|---|---|---|---|---|---|
| R0 | The live set as a mask: the reference | 0 | 0 | 0 | 0 | 160 | 160 | `tp1-coldr05-k160` |
| S15 | MD, plus every expert below 0.15% of the weight | 630 | 391 | 5.56 | 1.70% (2.28%) | 134–153 | 145 | `tp1-colds15t5-k145` |
| S20 | MD, plus every expert below 0.2% of the weight | 908 | 512 | 8.01 | 2.96% (5.38%) | 120–152 | 139 | `tp1-colds20t5-k139` |

MD, for comparison: 486 experts, 4.29 GiB, 1.25% (1.72%) lost, mean K 149. MD cost +0.46% and −0.10% against R0 in the
two windows: a mean of +0.18%.

**Why "MD plus".** The set of every expert below 0.2% (894 experts, `PLAN_COLD_EXPERTS.md` §3) does not contain MD:
14 experts of D1 sit above 0.2% in layers with a warm ordinary tail. The union keeps the points nested, so a result
reads as "MD and more", not as a different cut. S20 holds 908 experts for the same reason, not 894.

**The odds.** K 144 by rank failed the baseline gate with every rank (`PLAN_BETTER_RANK.md`: +3.29% to +3.90% against
K 216). S20 has a mean K of 139, and S15 of 145. The difference is the rule: those cuts kept every mandatory expert and
dropped warmer ordinary experts. S15 and S20 drop the coldest experts of each layer, mandatory or not. S15 loses 1.7%
of the weight per layer, S20 three percent. MD lost 1.25% for +0.18%. So S15 is the likely pass, and S20 the test of
the limit.

## 3. The gate

A small cut needs two replays, because one replay moves by up to 0.56% (`PLAN_COLD_EXPERTS.md` §7.3, the noise). Each
point boots twice, in the two halves of the window. Each half has its own R0. The script reads each replay against the
R0 of its half, and prints the mean of the two at the end (`cold_replay.sh`, `mean2`).

A point has **no measurable cost** when all of these hold:

1. The mean summed NLL against R0 is at most +0.5%.
2. The mean of the decisions against R0 is at least −1, and the mean of the tool names at least −1.
3. The smoke test (`prefill_probe.py --smoke`: the JSON schema and both tool choices) passes in both halves.
4. The point passes the baseline gate against K 216 (NLL at most +3%, tool names at least 98%) in both halves.

**The R0 check.** Each R0 must match `botmask5-k160` within 0.5% of the NLL. If one does not, the script restores the
live server and stops. Then the folders or the image changed, and the window gives no answer.

## 4. The decision

| Result | Meaning | Next |
|---|---|---|
| S20 has no measurable cost | About 8 GiB can go | The loader-patch chain for S20: the per-layer prune, the test window, the full runs, the judge, the blind review, the deploy (`PLAN_LOADER_PATCH.md`, `PLAN_MD_REVIEW.md`, `PLAN_MD_DEPLOY.md`) |
| S15 passes and S20 fails | The limit sits between 0.15% and 0.2% | The same chain for S15 (5.56 GiB). A point between them only if the memory is worth another window |
| Both fail | MD is the limit of the cold cut on this traffic | Stop the cold cuts. The fewer-layers plan is the next lever |
| S15 fails on the mean but passes in one half | The result sits in the noise | Treat it as a fail. The 0.5% budget is small, and the deploy chain is a day of work |

The full runs, the judge and the blind review stay the final gate before a deploy, as for MD.

## 5. The tools

- `cold_sets.py --share 0.15 0.2`: the new option writes `results/cold-s15.json` and `results/cold-s20.json`, the
  union of MD and every kept expert below the share. The check table prints the rows above. The four old files do not
  change: the regenerated R0 equals the live drops, and the regenerated MD equals the drops of `tp1-coldmd5-k149`.
- The folders: `mask_checkpoint.py --drop-file`, then `topk_checkpoint.py --top-k 5` and `--check`, then the
  intermediate folder is deleted, as in `PLAN_COLD_EXPERTS.md` §6 step 5. Each folder holds 12 GB.
- `cold_replay.sh`: the folder map has S15 and S20. A point with the suffix `-2` is a repeat: a new boot of the same
  folder, read against the R0 of its half. Both R0 points get the R0 check. The end of the window prints the mean of
  the two reports of each repeated point, with the gate of §3 on the mean.
- One test more in `test_cold_sets.py`: a share point is MD plus every expert below the share, and the hash layers
  keep the live drops only.

## 6. The window (about 4 h 35 min downtime)

CAUTION: this window stops the bot's model server for about 4 h 35 min. Start it only with your go. Do not start it
between 05:00 and 10:00 on a weekday, or during another test.

| Order | Point | Minutes |
|---|---|---|
| 1 | R0 | 44 |
| 2 | S20 | 44 |
| 3 | S15 | 44 |
| 4 | R0-2 | 44 |
| 5 | S20-2 | 44 |
| 6 | S15-2 | 44 |
| 7 | The restore of the live server | 10 |

Each point takes about 4 min to boot, 2 min for the smoke test, 35 min for the replay, and 3 min to stop. The script
skips a point that cannot end before `RESTORE_AT`. S15-2 is last, so a short window costs the repeat of S15, not the
repeat of S20.

Before the start:

1. The result files of the repeat window of 2026-11-11 move to `results/cold-w2/` and `logs/cold-w2/`. The replay
   skips the targets that its output file holds, so the old `cold-r0.jsonl` must not stay in `results/`.
2. The guard runs in tmux: `overnight_guard.sh` restores the live server at 04:15 and 04:35 if it is down.
3. The Hermes cron jobs run from 05:00 on a weekday. The window must end before that, or the jobs must be paused.

The start:

```
tmux new -d -s cold3 "POINTS='r0 s20 s15 r0-2 s20-2 s15-2' LOG=~/Projects/dsv4-specialize/logs/cold-replay-3.log
  START_AFTER='<now>' BOOT_BY='<start + 3 h 50 min>' SPEED_BY='<same>' RESTORE_AT='<start + 4 h 35 min>'
  ~/Projects/dsv4-specialize/cold_replay.sh"
```

The memory rules of `run_lib.sh` hold: no boot under 110 GiB free, and the memory watch stops a client below 1 GiB.
A mask of `data/tp1` loads 95.4 GiB and left 2 to 3 GiB free after the boot in the first windows.

## 7. What a pass gives, and what it does not

- **No memory yet.** A mask frees nothing. The memory comes from the per-layer prune and its loader patch, as for MD.
- **No speed.** The decode reads only the routed experts, so a cold expert costs memory, not time.
- **The deploy chain of a pass** is the chain of MD: the prune (about 1 h), a test window (about 1 h 30 min
  downtime), the full runs against the live model (about 7 h, no downtime), the judge, your blind review, and the
  deploy (about 4 min downtime).

## 8. Risks

| Risk | Effect | Answer |
|---|---|---|
| S20 boots with a layer at K 120 | A layer with few experts can fail a kernel path | The boot check and the smoke test see it; the two-failure rule restores |
| One half passes and one fails | The mean decides, but the result is in the noise | §4, last row: a fail |
| R0-2 drifts from `botmask5-k160` | The second half gives no answer | The R0 check stops the window; the first half still reads |
| The window runs past 04:15 | The guard and the script both start the live server | `docker compose start` on a running server is a no-op |
| The window runs past 05:00 on Friday | The cron jobs call a server that is down | The restore ends about 03:50 with a 23:15 start; the guard is the net |

## 9. Timeline

| Work | Time | Downtime |
|---|---|---|
| §5: the tools, the folders, the dry run | About 1 h | None |
| §6: the window | About 4 h 35 min | About 4 h 35 min |
| §4: the report and the decision | About 30 min | None |
