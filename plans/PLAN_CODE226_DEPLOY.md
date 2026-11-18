# Plan: the deploy of code226

Status (2026-11-17 20:10): you chose levers A and B. PR 3 (#99) and PR 1 are open; PR 2 is in progress. No window is
booked. The cron jobs stay paused (your choice).

## 1. Goal

Serve code226 on the live server in place of the MD prune. code226 is MD plus the 226 experts that coding traffic
needs most (`PLAN_CODE_RANK.md` §11). It is the folder `tp1-prunecode226t5-k154`.

| | Now: MD | After: code226 |
|---|---|---|
| Routed experts | 6,394 (K 140 to 160 per layer) | 6,620 (K 146 to 165 per layer, 15 counts) |
| Weights loaded, no draft | 66.54 GiB | 68.53 GiB |
| Weights loaded, with the bot draft | 69.79 GiB | about 71.78 GiB (the cap of the router plan is 72 GiB) |
| Junk on the probes (c1, c2, c3) | 14/30 and 19/30 | 1/30 |
| Coding NLL, 40 held-out turns | 0.7538 | 0.7335 (−2.69%) |
| Bot weight on removed experts (estimate) | 5.88% | 5.33% |

The bot replay is not measured yet. code226 keeps every expert of MD, so the bot replay should be equal to MD's or
better. MD costs +2.75% of the +3% budget. §5 measures it.

## 2. The memory (your decision before the switch)

code226 takes 1.99 GiB more than MD. That memory comes from the margin beside Qwen. Today, with MD and Qwen up, the
lowest available memory in an idle check is about 9.1 GB (18:40). The pair watch stopped Qwen 3 times in two days
(new VS Code windows), and a Qwen start beside MD already needs `NEED_MB=27000`.

Two levers can give the memory back:

- **A. Qwen's KV pin back to 133k tokens** (1,965,377,034 bytes, from 3,575,989,770): about 1.5 GiB. One 128k request
  still fits; the 4 sequences of `SEQS=4` share the cache.
- **B. A DeepSeek context cap of 160k** (now 256k), with a smaller KV pin: about 0.35 GiB, as measured in the Qwen plan.
  A prompt above 160k tokens then fails. The largest bot calibration prompt has 133k tokens.

| Choice | Margin beside Qwen against today |
|---|---|
| A and B | about the same (−0.1 GiB) |
| A only | about 0.5 GiB less |
| Neither | about 2 GiB less; expect more stops of Qwen by the watch |

You chose A and B on 2026-11-17.

## 3. What changes on the live server

A new compose file `compose-code226t5.yml` in the server repo. It is a copy of `compose-md149t5.yml` with these
differences:

1. `/models/tp1` is `data/tp1-code226t5`: the code226 folder, moved from this repo, as for MD.
2. Its own kernel cache `./cache-code226t5`: a copy of this repo's `cache` (815 MB), which holds the kernels of the 15
   expert counts from the test boots.
3. The project name `deepseek-v4-flash-code226t5`.
4. Lever B: `MAX_MODEL_LEN` 160000 and a KV pin of 7,431,600,000 bytes (6.92 GiB, the value that `cap_switch.sh`
   planned on 2026-11-14; the cap was never booted). The review window of §5 checks the pin. If vLLM says that it is
   too small, the window boots once more at the need plus 4%, and the compose file gets that value before the switch.

These stay the same:

- the image, the patches (the per-layer loader is already in the server repo since MD), and the draft;
- the port 8888;
- the two served names. The cron jobs, the gateway and the router need no change.

**Hermes** (lever B): after the checks of the switch pass, `deploy_md.sh` sets `context_length` 160000 for the
DeepSeek provider and for `auto` (the router sends long requests to DeepSeek), and the compression threshold 0.8 in
place of 0.5, with a backup of `config.yaml`; then it restarts `hermes-gateway`. DeepSeek and `auto` then still
compress at 128k tokens. Qwen's compression starts at about 102k tokens in place of 64k. A copy of the live config
takes exactly these three changes (tested 2026-11-17). A later rollback puts the backup back by hand.

## 4. The tools (PRs, no downtime)

Each PR changes 400 lines or fewer, with its tests:

1. **`deploy_md.sh`, general:** `DEPLOY` picks a profile (`md149t5`, `code226t5`) with the new name, the old compose
   file and the boot checks (the expected weights, the 15 counts, the minimum KV tokens). The env can set
   `EXPECT_GIB` after §5 measures it. New modes `vars` and `hermes-cap`. Add `deploy/compose-code226t5.yml`, and tests.
2. **`review_code226.sh`:** the review window of §5, built from `loader_window.sh`, with `compose-cap.yml` for the
   context length and the KV pin retry of `cap_switch.sh`.
3. **`pair_run.sh`:** land the `KV_BYTES` change that waits uncommitted in the main tree. Lever A needs it.
4. **After the switch:** the defaults of `run_lib.sh` and `overnight_guard.sh` name `compose-code226t5.yml`. Every
   window script then restores the new server.

## 5. The review window (about 2 h; DeepSeek down; Qwen stopped)

`review_code226.sh` on port 8890, with the bot draft and the loader patch:

1. **md** (the live folder at its live settings: 256k, the 7.25 GiB KV pin; the reference of the same window): boot,
   smoke, one replay, the speed test (3 runs).
2. **code226** at the settings of the deploy (160k, the 6.92 GiB KV pin, with one retry at the need plus 4%): boot,
   smoke, two replays, the speed test (5 runs), the prefill probe up to 128k tokens.

**The gates:**

| Check | Pass |
|---|---|
| Boot | about 71.78 GiB loaded with the draft; the pool line has 15 counts; the KV cache holds at least 160,000 tokens at the cap |
| Smoke | `prefill_probe.py --smoke` passes (the JSON schema and both tool choices) |
| Bot replay | the mean of the two against `baseline-k216`: NLL at most +3%, tool names at least 98%; and at most 0.5% above the md replay of the same window |
| Speed | the bot decode within 3% of md in the same window |

If one gate fails, the deploy stops, and MD stays live.

## 6. The prepare step (no downtime)

1. Move `tp1-prunecode226t5-k154` to `data/tp1-code226t5`, and leave links in this repo, as for MD.
2. Copy `cache` to `cache-code226t5`.
3. Copy `deploy/compose-code226t5.yml` into the server repo.

`./deploy_md.sh prepare` does these steps and checks them. It also takes the before readings: the `/metrics` counters
and the bot decode test (5 runs) on port 8888.

## 7. The switch (about 4 min downtime)

CAUTION: the bot does not answer while the server is down. You start this step with a `!` command, because auto mode
blocks it:

```
tmux new -d -s deploy "DEPLOY=code226t5 EXPECT_GIB=<the value of §5> ~/Projects/dsv4-specialize/deploy_md.sh switch"
```

The script runs these steps, as for MD:

1. Check the files and the live server.
2. Start the deploy guard. It restores MD if port 8888 is down 90 min after the start.
3. Wait for an idle bot, then stop MD.
4. Start code226, and run the checks of §5 (boot, KV, both served names, a chat reply, `hermes-gateway`).
5. If a check fails, save the logs, stop code226 and start MD again.

## 8. After the switch (no downtime)

1. **Qwen:** start it with the lever of §2 (`KV_BYTES=1965377034` for A) through `pair_run.sh` with `TESTS=idle`.
2. **The fit:** run `pair_run.sh` with the load tests (`qn`, `qnd`: Qwen and DeepSeek decode at the same time). Record
   the lowest available memory and the lowest free swap.
3. **The readings:** `./deploy_md.sh readings after`, and the time per model call of the next 24 hours, as in
   `PLAN_MD_DEPLOY.md` §8.
4. **The defaults:** merge PR 4 of §4.

## 9. The decision: keep, or roll back

| Reading | Roll back to MD when |
|---|---|
| The fit of §8 | the lowest available memory under load is below 2.5 GB, or the lowest free swap is below 4 GB |
| The bot decode | more than 3% slower than the before readings |
| The watch | it stops Qwen more than once a day in normal use |

The rollback is the same switch with the names swapped. MD's folder and compose file stay in place.

## 10. Risks

- **The bot replay is new.** code226 keeps all of MD's experts, so a fail is unlikely; §5 checks it before any switch.
- **The size cap:** with the draft, code226 is about 71.78 GiB, just under the 72 GiB of the router plan.
- **The junk evidence is 4 prompts.** The coding score of 40 turns and the pilot shares support it. A few full
  agent-eval tasks after the switch would add evidence.
- **Memory:** without levers A and B, Qwen loses about 2 GiB of margin.

## 11. Order and time

1. You choose the levers of §2.
2. PRs 1 to 3 of §4: about 2 h of tool work.
3. The review window of §5: about 2 h of downtime. It can run tonight or at a time you choose.
4. The prepare step and the before readings: about 30 min, no downtime.
5. The switch: about 4 min of downtime, started by you.
6. The after readings and the fit: about 1 h, then 24 h of watch.
