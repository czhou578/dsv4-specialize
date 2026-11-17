# Plan: a rank that counts coding traffic

Status (2026-11-17 11:10): done. The chain ran from 08:22 to 11:07 (§10). code1 (MD + 359 coding experts, +3.17 GiB)
cuts the junk from 35/60 on MD to 2/30 and the coding NLL by 2.92%. The next step is the memory for it.

## 1. The question

The night of 2026-11-16 (`PLAN_JUNK_CAPTURE.md` §10) found the source of the junk tokens:

- c3 comes from the 486 experts that the MD step removed.
- c2 comes from the cut to K 160, not from top-k 5.
- The coding traffic routes about twice the bot share to the removed experts. The rank of K 160 and MD used bot
  traffic only.

DeepSeek now serves coding too: the router sends code, math and long requests to it (`model-router`).

This plan asks: **which experts does coding need back, how much memory do they cost, and does that remove the c2 and
c3 junk without a cost on the bot?**

## 2. The constraints

- **The bot gate has almost no room.** MD costs +2.75% NLL against the stock model on the bot replay. The gate is +3%.
- **Memory has no room either.** Qwen K 208 runs beside DeepSeek. Beside MD, 27.4 to 29.2 GB is available, and Qwen
  needs 27,000 MB (`NEED_MB`; the default 28,600 failed twice on 2026-11-16). Each expert added to DeepSeek takes
  9.04 MiB from that margin. 113 experts take 1 GiB.

## 3. The estimate from the night data (2026-11-17)

This uses the routes that the stock model gave in window 1 of the night. Rank on 20 pilot tasks (C, D and L, without
C1, C2, C3 and D3); test on the 16 R and W tasks and on the probes. The numbers are the share of the router weight
that a set drops.

**A swap at MD's memory fails.** The rank "the larger of the bot share and the coding share" at MD's expert count swaps
about 550 experts. The dropped bot weight rises from 5.88% to 7.03%. In `PLAN_COLD_CUT2.md`, S15 (MD plus every kept
expert below a 0.15% weight share: 5.56 GiB dropped in place of 4.29 GiB) already failed the +3% gate.

**Experts added to MD are safe for the bot.** Each added expert lowers the dropped bot weight. The cost is memory:

| Added to MD | Memory | Bot dropped | Coding (R, W) dropped | c2 dropped | c3 dropped |
|---|---|---|---|---|---|
| 0 (MD) | 0 | 5.88% | 10.63% | 13.75% | 14.87% |
| 113 | +1.00 GiB | 5.65% | 9.88% | 10.86% | 11.41% |
| 226 | +2.00 GiB | 5.38% | 9.02% | 9.30% | 9.14% |
| 339 | +2.99 GiB | 5.13% | 8.46% | 7.80% | 7.45% |
| 486 | +4.29 GiB | 4.75% | 7.63% | 6.36% | 5.93% |
| K 160 (reference) | +4.29 GiB | 4.77% | 8.65% | 9.91% | 9.37% |

At K 160 (9.91% and 9.37% dropped), c3 was clean (0/10) and c2 was junk (6/10). So c3 likely needs about +2 GiB, and
c2 likely needs more. These numbers come from 40 synthetic tasks that MD wrote. The capture of §4 replaces them with
your own coding traffic.

## 4. The capture window (window 1, about 50 min; DeepSeek down about 35 min)

- **Data:** the calibration split of `qwen-data` (the set of `PLAN_QWEN_PRUNE.md` §3): 37 segments of your Claude Code
  transcripts (about 1.45M tokens) and 42 repository files (about 0.12M tokens), in the Hermes format that the router
  sends to DeepSeek. The pilot tasks stay out of the rank, so that they stay a test.
- **Tool:** `capture.py --data qwen-data --sources claude,repo --sums` (it reads this format unchanged), in
  `junk_capture.sh` with a new client mode. It uses the same stack as the night: stock model, KV pin, memory guard,
  smoke test.
- **Time:** about 1.6M tokens at about 1,050 tokens per second gives about 25 min of capture.

## 5. The rank and the candidates (no window)

1. For each expert that MD drops in layers 3-42, compute its share of the coding weight in its layer from the new
   capture. Layers 0-2 route by a hash and keep their drops, as in every cut so far.
2. Add the experts to MD in that order. Write the curve of §3 again, with these columns:
   - the bot dropped weight (from `results/capture-rank.jsonl`);
   - the coding dropped weight on all 40 pilot tasks, a held-out test;
   - c2 and c3 dropped.
3. Choose two candidates from the curve:
   - **C1:** the smallest addition where c3 drops at most 9.37%, the level of K 160 (c3 clean there);
   - **C2:** the smallest addition where c2 drops at most 7.5%, well below the 9.91% of K 160 (c2 junk there).
   - In §3's estimate, C1 is about +2 GiB and C2 about +3 GiB.
4. Build each candidate as on the night: a drop file (`results/cold-md.json` without the added experts),
   `mask_checkpoint.py --drop-file`, `topk_checkpoint.py --top-k 5` and `--check`, then `prune_run.sh`. Each build takes
   about 12 min and holds about 70 GB.

## 6. The probe window (window 2, about 1 h 30 min; DeepSeek down about 1 h 15 min)

- **Points:** MD, C1, C2, MD again. Use `bisect_junk.sh` with two new points that take their folders from variables,
  as `mdx` did.
- **For each point:**
  - the junk probes: 10 cold runs of c1, c2, c3 and d3, and the greedy pair;
  - the coding score: the NLL of 40 held-out turns of `qwen-data` (Claude Code, held-out split). Every model scores the
    same text. `qwen_gate.py` gets an option that takes these turns from the data, with no recording run.
- **Pass:** c2 and c3 each at 2/10 junk or fewer with sane mass 0.9 or more, while both MD controls give junk. Also the
  coding NLL is lower than MD's.

## 7. The decision

| Result | Next step |
|---|---|
| C1 passes | Plan the deploy of C1 (§8) |
| C2 passes, C1 fails | Plan the deploy of C2, and decide how to fund its memory (§8) |
| Both fail | The junk does not follow expert coverage alone. Stop the expert work, and probe the c2 position on K 160 with its experts and on stock to find the layers that matter |

## 8. The deploy, and the memory (a later plan)

A deploy plan, as `PLAN_MD_DEPLOY.md`, needs these steps:

- the bot replay gate against the stock model, two replays (NLL +3% at most, tool names 98% at least);
- the bot speed test;
- a junk check on a few pilot tasks run in full;
- the fit beside Qwen.

A candidate adds memory, so you must choose where it comes from:

- **The Qwen KV pin:** set it back from 3.58 GB (241k tokens) to 1.97 GB (133k tokens). This frees about 1.5 GiB. It
  is the only large lever.
- **The DeepSeek context cap:** about 0.35 GiB.
- **Run Qwen with less margin:** swap carries part of it today.

## 9. The work before window 1

Each PR stays at 400 lines or fewer, with its tests:

1. `junk_capture.sh`: a client mode for `capture.py` on `qwen-data`.
2. The rank tool: the curve of §5 from two capture files, and the drop file for a chosen addition.
3. `qwen_gate.py --score` with the held-out turns from the data, and DeepSeek settings.

`bisect_junk.sh` and `junk_night.sh` are still untracked in the bisect worktree. They can land as their own PR, or
stay there for one more night.

**Timeline:** about 3 h of tool work. Then window 1, the curve and the builds (about 1 h 30 min), then window 2.
Window 1, the builds and window 2 can run as one night chain, as on 2026-11-16, with the candidates chosen by the
fixed rules of §5.


## 10. Results (2026-11-17, the chain `code_night.sh`)

- **Window 1** (08:22 to 08:52): 79 rows of `qwen-data` (37 Claude Code segments, 42 repository files), no errors,
  24 min of capture.
- **The rank** (`results/code-rank/code-rank.md`): ranked by your own coding traffic, the additions help the pilot tasks
  less than the estimate of §3 (which ranked on pilot tasks).
  - code1 = +359 experts (+3.17 GiB, mean K 157): c3 drops 9.35%.
  - code2 = +486 experts (+4.29 GiB, the cap, mean K 160): c2 drops 8.74%; the 7.5% of the rule is not reached.
  - code2 has the memory of K 160 but beats it on every coding measure (pilot 8.91% against 9.87%, c2 8.74% against
    9.91%, c3 8.14% against 9.37%) and ties it on the bot (4.73% against 4.77%).
- **Window 2** (09:14 to 10:58, `results/code-probe/`), junk with the mean sane mass, and the coding NLL per token of
  the 40 held-out turns (34 Claude Code, 6 repository files):

  | Point | c1 | c2 | c3 | All | Coding NLL |
  |---|---|---|---|---|---|
  | MD control | 3/10 (0.67) | 3/10 (0.60) | 9/10 (0.16) | 15/30 | 0.7538 |
  | code1, +3.17 GiB | 1/10 (0.85) | 0/10 (0.96) | 1/10 (0.84) | 2/30 | 0.7318 (−2.92%) |
  | code2, +4.29 GiB | 1/10 (0.85) | 0/10 (0.93) | 2/10 (0.75) | 3/30 | 0.7264 (−3.63%) |
  | MD control again | 5/10 | 5/10 | 10/10 | 20/30 | 0.7532 (−0.08%) |

- **Against both MD controls** (35/60): code1 p = 9×10⁻⁷, code2 p = 6×10⁻⁶. c2: 0/10 against 8/20 (p = 0.022). c3: 1/10
  and 2/10 against 19/20. The pass rule of §6 (2/10 or fewer, sane mass 0.9 or more): c2 passes on both; c3 meets the
  junk count but its sane mass (0.84 and 0.75) stays under 0.9.
- **The coding score** moves by 0.08% between the two MD controls, so the −2.92% and −3.63% are real. Tool names stay
  at 100%; the decisions move by one.
- **The answer:** the junk follows expert coverage of coding traffic. code1 gives almost all of the gain of code2 for
  1.12 GiB less.
- **After the window:** DeepSeek healthy at 10:58; Qwen up at 11:07 (`NEED_MB` 27,000). The cron jobs stay paused.
