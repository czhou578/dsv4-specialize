# Plan: The deploy of the MD prune

**Status:** 2026-11-12: done. The MD prune is live since 18:00:26, and you kept it (§9.1). Reading C is due on
2026-11-13 · **Author:** colin-spark · **Date:** 2026-11-12
**Related:** `PLAN_MD_REVIEW.md` §8.1 (the decision that allows the deploy), `PLAN_LOADER_PATCH.md` §6.2 and §7.1
(the patch and its speed tests), `PLAN_FEWER_EXPERTS.md` §9.2 (the deploy steps of the last prune, not done),
`PLAN_MODEL_ROUTER.md` §3 and §8 in the bot's repo (the size cap and the second model)
**Scope:** one deploy script, its dry run, and the live-server names in `run_lib.sh` and `overnight_guard.sh` in this
repo. In the server repo: one new compose file, two patch files, the moved folder and a copy of the kernel cache.
About 4 minutes of downtime. No change to Hermes, its cron jobs, the gateway, `compose-k160t5.yml`,
`data/tp1-k160t5` or `cache-k160t5`.

| Step | Status |
|---|---|
| §4 The script and the names in this repo (PR, no downtime) | ✅ 2026-11-12: `deploy_md.sh`, `deploy/compose-md149t5.yml` and the settings (PR #35); the tests of §4.1 pass |
| §5 The before readings (no downtime) | ✅ 2026-11-12, 17:28 to 17:57: 35.27 tok/s, and the counters saved (§5.1) |
| §6 The server-repo files (no downtime) | ✅ 2026-11-12, 17:28: `prepare` took 3 s, and every file is in place (§6.1) |
| §7 The deploy (about 4 minutes of downtime) | ✅ 2026-11-12, 17:57:33 to 18:00:26, on your "start the deploy": 2 min 53 s of downtime, every check passes (§7.1) |
| §8 The after readings (no downtime) | ✅ A: 35.30 tok/s (+0.1%). B: 78.4 s per model call (+9.9%), no hard fault. C: ⏳ due 2026-11-13 at about 18:00 (§8.1) |
| §9 The decision: keep, or roll back | ✅ 2026-11-12, about 22:30: keep (your decision). The defaults of `run_lib.sh` and `overnight_guard.sh` are in this PR (§9.1) |

---

## 1. Goal

Serve `tp1-prunemd5-k149` on the live server in place of `tp1-k160t5`. Keep the KV pin at 7.25 GiB, so the
4.18 GiB that the prune frees stays free for the second model of the router plan (your decision, 2026-11-12).

The MD prune loads 69.79 GiB. This is below the size cap of 72 GiB of the router plan (§3 there), so the deploy also
completes Phase B of that plan.

| | Now: `tp1-k160t5` | After: the MD prune |
|---|---|---|
| Weights loaded | 74.09 GiB | 69.79 GiB |
| KV cache | 7.25 GiB (pinned) | 7.25 GiB (pinned, no change) |
| Memory available, server idle | about 23 GiB | about 27 GiB (26.8 GB at the boot of the review window) |
| Room for the second model, 10 GiB margin | about 13 GiB | about 17 GiB |
| Room for the second model, 4 GiB margin | about 19 GiB | about 23 GiB |

## 2. What changes on the live server

A new compose file, `compose-md149t5.yml`, in the server repo. It is a copy of `compose-k160t5.yml` with these
differences:

1. `/models/tp1` is `data/tp1-md149t5`: the MD prune folder, moved from this repo (§6).
2. Two more read-only mounts: the two built files of `image-patch/perlayer/`, copied into the server repo's
   `image-patch/`. The live server then does not depend on this repo.
   - `image-patch/vllm/model_executor/layers/quantization/exl3.py`
   - `image-patch/vllm/models/deepseek_v4/nvidia/model.py`
3. Its own kernel cache, `./cache-md149t5`: a copy of the test cache of this repo (397 MB). That cache holds the
   kernels of the 15 expert counts from the test boots, so the boot does not compile them again.
4. The project name `deepseek-v4-flash-md149t5`, so the container is `deepseek-v4-flash-md149t5-deepseek-v4-flash-1`.

These stay the same:

- The image (by digest), all other patch mounts, the draft `data/dspark-draft-k64-bot`, the KV pin, the port 8888
  and the other settings.
- The two served names, `deepseek-v4-flash-0731-k160t5` and `deepseek-v4-flash-0731`. Hermes, the two cron jobs
  (pinned to the first name since 2026-11-12), the gateway and the e2e runner then need no change. The first name no
  longer describes the folder. The header of the compose file records this.

The test boots of the review used the same image, folder, patch, draft and KV pin, on port 8890. The deploy changes
only the port, the names and the cache path.

## 3. The readings of the time per model call

The review runs of the MD prune took about 11% longer per model call than the live runs (`PLAN_MD_REVIEW.md` §6.2).
The decode test of the loader window found −1.3%. The MD runs ran at night on the test compose, so the cause is not
clear. Three readings separate the cause. None of them stops the live server.

| Reading | Before the deploy | After the deploy | Use |
|---|---|---|---|
| A. The bot decode test (`model_eval.speed`, 5 runs, about 10 min) | On port 8888 | On port 8888 | **The gate:** within 4.2% (the noise of the test, `PLAN_LOADER_PATCH.md` §6) |
| B. One full run of the 19 prompts (`review_live.sh`, about 3.5 h) | `live-1` and `live-2`: 71.3 and 71.4 s per model call | `deploy-1` on port 8888 | **The reading that you asked for:** the seconds per model call, with the same bot code (`a0fd27b`) |
| C. The server counters (`/metrics`) | One day of the live server | One day of the MD prune | The mean time per output token, the time to the first token and the draft tokens accepted, on the real traffic |

For reading B:

- At most +5% (74.9 s): the +11% of the review came from the test setup. Keep the deploy.
- More than +5%: the prune costs time on the bot's work. You decide (§9).

`deploy-1` also gets the checks of step F against `live-1` and `live-2`, as a test of the hard faults.

## 4. The script and the names in this repo (PR, no downtime)

1. `deploy_md.sh` has four modes. It uses `run_lib.sh` for the log, the idle wait and the restore.
   - `prepare`: the files of §6.
   - `readings TAG`: the readings A and C of §5 and §8.
   - `check [LOG]`: the checks of §7 step 4 on a boot log.
   - `switch`: the deploy of §7, with its own rollback. `DRY_RUN=1` with `switch` runs every check, then ends
     before any server action.
2. `deploy/compose-md149t5.yml`: the source of the compose file of §2. `prepare` copies it into the server folder.
3. `run_lib.sh` and `overnight_guard.sh`: the compose file and the container of the live server become settings
   (`MAIN_COMPOSE`, `MAIN_CONTAINER`). The defaults stay `compose-k160t5.yml` until the deploy is kept (§9).
   - Without this change, the next test window cannot use the new live server. Its safety rules stop it: a server
     container runs, and less than 100 GiB is available.
   - `overnight_guard.sh` also gets `STOP_PROJECTS`. The deploy guard stops the new server before it starts the old
     one.

### 4.1 The tests (2026-11-12, no server action)

| Test | Result |
|---|---|
| `DRY_RUN=1 ./deploy_md.sh switch` before `prepare` | Stops at the file check: it names the 5 missing files and starts no guard |
| `./deploy_md.sh check results/boot-md-review.log` (the MD boot of the review) | Every check passes: 69.79 GiB, 15 counts, 265,498 tokens, both names, "OK.", the gateway |
| `./deploy_md.sh check results/boot-lp-p0.log` (a K 160 boot) | Fails, as expected: 74.09 GiB and no pool line |
| `overnight_guard.sh`, `DRY_RUN=1`, with a down server and `STOP_PROJECTS` | Stops the containers of every listed project, then starts `MAIN_COMPOSE` |
| The guard command of `switch`, `DRY_RUN=1` | The quotes hold. The guard waits for a full date and time |
| The repo tests (the bot's venv) | 62 tests pass, 2 skipped |

The full dry run of `switch` comes after `prepare` (§6), before you start §7.

## 5. The before readings (no downtime)

`./deploy_md.sh readings before` (about 10 min):

1. Reading C: it saves `/metrics` to `results/metrics-deploy-before.txt`. `switch` saves the counters again just
   before the stop (`results/metrics-deploy-old-final.txt`). The counters started at the restore of 05:27 on
   2026-11-12, so they include the 114 judge pairs of the review.
2. Reading A: the bot decode test, 5 runs, to `results/speed-bot-deploy-before.jsonl`. A bot message waits behind
   a test request.

### 5.1 The results (2026-11-12, 17:28 to 17:57)

| Reading | Result |
|---|---|
| C: `results/metrics-deploy-before.txt` | Saved at 17:28:56. `switch` saved the final counters of the old server at 17:57:33 (`results/metrics-deploy-old-final.txt`): 445 requests since 05:27, 2.86 draft tokens accepted per decode step |
| A: `results/speed-bot-deploy-before.jsonl` | 35.27 tok/s (the 5 runs: 33.72 to 35.81), median 35.24 tok/s, 2.173 draft tokens accepted per step |

## 6. The server-repo files (no downtime)

**Caution:** the live server uses `compose-k160t5.yml`, `data/tp1-k160t5`, `cache-k160t5` and the draft. Do not
change them. The steps below add files only. `./deploy_md.sh prepare` does all four, then checks each file:

1. Move the folder: `mv ~/Projects/dsv4-specialize/tp1-prunemd5-k149
   ~/Projects/DeepSeek-v4-Flash-One-DGX-Spark/data/tp1-md149t5` (the same disk, seconds). Put a symlink at the old
   path, so that the test boots of `MASK_NAME=prunemd5` still work. The links in the folder are absolute paths into
   `data/tp1`, so they still resolve.
2. Copy the two built files of `image-patch/perlayer/` to the paths of §2. Compare the sha256 of each copy.
3. Copy `~/Projects/dsv4-specialize/cache` to `~/Projects/DeepSeek-v4-Flash-One-DGX-Spark/cache-md149t5`. Root
   owns 230 of its files (mode 600), so a container of the live image does the copy (`cp -a`, no network).
4. Copy `deploy/compose-md149t5.yml` into the server folder (§2). Check it with `docker compose config -q`.

### 6.1 The result (2026-11-12, 17:28)

`./deploy_md.sh prepare` ran at 17:28:45 with the live server up. It moved the folder, copied the two patch files,
copied the kernel cache (421 MB, 3 s) and the compose file, and found every file in place. The full dry run of
`switch` at 17:57:23 passed the file checks, the live-server checks and every check of §7 step 5 on the boot log of
the review window, and ended with no server action.

## 7. The deploy (about 4 minutes of downtime)

**Caution:** the bot does not answer while the server is down. The auto mode does not let Claude start this step.
You start it with a `!` command: `tmux new -d -s deploy "~/Projects/dsv4-specialize/deploy_md.sh switch"`.

The cron jobs run from 05:00 to 09:40 on weekdays. On weekdays, `switch` does not start from 04:15 to 09:45: the
idle wait can take 30 min. `deploy_md.sh switch` then:

1. Checks the files of §6, and that the live server is the only server and is healthy.
2. Starts the deploy guard (tmux `deploy-guard`), then saves `/metrics`.
3. Waits until the bot is idle (up to 30 min), and stops `compose-k160t5.yml`. Then it checks that no other server
   runs and that 110 GiB are available (the boot check of vLLM at 0.90).
4. Starts `compose-md149t5.yml`, and waits for the health check (up to 10 min; the test boot took 165 s).
5. Checks (`checks`, the same code as `./deploy_md.sh check`):
   - The boot log says "Model loading took 69.79 GiB", and the pool line has 15 counts.
   - The KV cache has at least 256,000 tokens (265,498 in the review window).
   - `/v1/models` lists both served names.
   - A chat request gets "OK.". A bot request can come first after the boot, so there are 3 tries of 5 min.
   - `hermes-gateway` is active.
6. If a check fails: it saves the logs, stops the new server, and starts `compose-k160t5.yml` again (the restore of
   `run_lib.sh`, with its own checks).

The deploy guard is a second safety: if port 8888 is down 90 minutes after the start, it stops the new server and
restores `compose-k160t5.yml`. The script can take up to about 80 minutes (the idle wait, the boot and a slow
rollback), so the guard acts only after that time.

### 7.1 The result (2026-11-12, 17:57 to 18:00)

You wrote "start the deploy", and the auto mode let Claude start `switch` in tmux. The log is
`logs/deploy-md.log`; the boot log is `results/boot-deploy-md149t5.log`.

| Time | Step |
|---|---|
| 17:57:33 | `switch` starts. The guard is set for 19:27:33 with `STOP_PROJECTS='dsv4-mask deepseek-v4-flash-md149t5'`. `/metrics` saved |
| 17:57:33 | The bot is idle, so there is no wait. `compose-k160t5.yml` stops in 2 s |
| 17:57:35 | `compose-md149t5.yml` starts, with 112 GiB available |
| 18:00:26 | Healthy after 171 s. Every check passes: 69.79 GiB, 15 counts, 265,498 tokens, both names, "OK." on the first try, the gateway active, 26 GiB available |
| 19:27:33 | The guard finds the server healthy and ends with no action (`logs/deploy-guard.log`) |

The downtime was 2 min 53 s. The old container stays as `Exited (0)`, for the rollback.

## 8. The after readings (no downtime)

1. Readings A and C: `./deploy_md.sh readings after`, as in §5.
2. Reading B: `LABELS=deploy-1 ./review_live.sh` (about 3.5 h), then the checks against `live-1` and `live-2`.
3. Reading C: save `/metrics` one day after the deploy. Compare the means with the before file.

### 8.1 The results (2026-11-12, 18:01 to 22:14)

**Reading A passes** (`./deploy_md.sh readings after`, 18:01 to 18:29). The gate is 4.2%.

| | Before (K 160) | After (the MD prune) |
|---|---|---|
| Rate, 5 runs | 35.27 tok/s | 35.30 tok/s (+0.1%) |
| Median | 35.24 tok/s | 35.95 tok/s (+2.0%) |
| Draft tokens accepted per step | 2.173 | 2.179 |

**Reading B is +9.9%**, more than the +5% line (`LABELS=deploy-1 ./review_live.sh`, 18:29:09 to 22:14:01, 3 h
45 min). All 19 prompts ended with exit 0, and the check has 0 FAIL lines.

| Run | Server | Model calls | Seconds per call | Against the live mean (71.35 s) | Run time | Output tokens per call | Ms per output token |
|---|---|---|---|---|---|---|---|
| `live-1` | K 160, live | 158 | 71.3 | — | 188 min | 1,837 | 38.8 |
| `live-2` | K 160, live | 151 | 71.4 | — | 180 min | 1,758 | 40.6 |
| `md-1` | MD, test compose at night | 156 | 80.9 | +13.4% | 210 min | 2,078 | 38.9 |
| `md-2` | MD, test compose at night | 166 | 78.3 | +9.7% | 217 min | 1,964 | 39.8 |
| `deploy-1` | MD, live | 172 | 78.4 | +9.9% | 225 min | 1,972 | 39.8 |

- All three MD runs are 9.7% to 13.4% slower per call. The extra time is real; the test setup of the review night
  is not the cause.
- The cause is longer answers, not a slower decode. The MD prune writes about 8% more output tokens per call (on the
  18 prompts with token counts: 2,031 against 1,877), and it makes more model calls (172 against 158 and 151). The
  time per output token is the same: 39.8 ms against 38.8 and 40.6 ms. Reading A says the same.
- The time per call varies a lot between runs of one prompt (one research prompt: 882 s on `live-1`, 471 s on
  `live-2`), so one run gives a rough figure.
- **The checks of step F find no hard fault.** Against `live-1`, every gate passes. Against `live-2`, one prompt gets
  a number-fidelity flag: the AMPX research prompt (`20260912_195336_669cff`), with 21 of 353 numbers without a match
  in the tool output. All 21 are figures from the 10-Q tables in thousands ("$101.7M" from the cell "101,713") or
  correct results from them (+113% revenue growth, 66% below the 52-week high). The check does not scale a table in
  thousands, so these are not errors. Over the run, `deploy-1` has the smallest share of numbers without a source:
  2.4%, against 3.4% and 3.7% on the live runs. Files: `results/review-deploy-1-check.txt`,
  `results/review-deploy-1-vs-live-1.txt` and `results/review-deploy-1-vs-live-2.txt`.

**Reading C is due on 2026-11-13 at about 18:00.** Caution: the before counters (`metrics-deploy-old-final.txt`)
include the 114 judge pairs of the review, which are short JSON answers with a high draft acceptance (2.86 tokens
per step, against 2.17 in the bot decode test). The two days have a different traffic mix, so read the counters
with that in mind. Readings A and B above hold the traffic constant. At 22:10, after 228 requests, the inter-token
latency per decode step was 92.8 ms, against 93.8 ms in the before counters.

## 9. The decision: keep, or roll back

| Result | Next |
|---|---|
| Reading A passes, and reading B is at most +5% | Keep. In a PR: the defaults of `run_lib.sh` and `overnight_guard.sh` become the new compose file and container. Phase B of the router plan is complete |
| Reading A passes, and reading B is more than +5% | You decide: the time that the bot loses, against the memory for the second model |
| Reading A fails | Roll back, then find the cause on a test boot |
| A hard fault in `deploy-1` that the live runs do not have | Roll back, then look at the fault |

**The rollback** (about 4 minutes of downtime): `docker compose -f compose-md149t5.yml down`, then
`docker compose -f compose-k160t5.yml start`. Keep `data/tp1-k160t5` (73 GB) and `cache-k160t5` until you delete
them. The disk has 2.8 TB free, so this can wait.

### 9.1 The decision (2026-11-12, about 22:30): keep

Reading A passes, and reading B is +9.9%: the second row of the table. You chose to keep the MD prune, for these
reasons:

- The extra time comes from longer answers at the same decode rate (§8.1).
- The judge (25 W / 16 L / 35 T) and your blind review (9 W / 7 L / 4 T) chose the MD prune more often than the K 160
  prune (`PLAN_MD_REVIEW.md` §7.1).
- The 4.18 GiB is for the second model of the router plan.

The cost: about 10% more time per model call, which is about 1 to 2 more minutes per research session.

This PR changes the defaults of `run_lib.sh` and `overnight_guard.sh` to `compose-md149t5.yml` and
`deepseek-v4-flash-md149t5-deepseek-v4-flash-1`. `deploy_md.sh` now sets the old server as `MAIN_COMPOSE` itself, so
a repeat of `switch` refuses with "runs already". Phase B of the router plan is complete; mark it in the bot's repo.
`compose-k160t5.yml`, `data/tp1-k160t5` (73 GB) and `cache-k160t5` stay as the rollback until you delete them.

## 10. Risks

| Risk | Effect | Answer |
|---|---|---|
| The new server does not boot (a mount, the cache) | About 8 min of downtime | `deploy_md.sh` restores `compose-k160t5.yml` (§7 step 5) |
| The deploy script ends early (for example, tmux ends) | The bot stays down | The deploy guard restores the old server 90 min after the start |
| The copied kernel cache does not match | The first boot compiles kernels: a few minutes slower | No fault. The health check waits up to 10 min |
| A test window after the deploy restores the old model | The bot runs the old model after the window | §4 step 2 and the defaults of §9 |
| The served name no longer describes the folder | A reader takes the live model for `tp1-k160t5` | The compose header, and this plan |

## 11. Effort and order

| Step | Time | Downtime |
|---|---|---|
| §4 The script, the names and the dry run (a PR) | About 1 hour | None |
| §5 The before readings | About 15 min | None |
| §6 The server-repo files | About 10 min | None |
| §7 The deploy | About 5 min | About 4 min |
| §8 The after readings | About 3.5 h, then one day for reading C | None |
| §9 The decision | — | About 4 min, only for a rollback |
