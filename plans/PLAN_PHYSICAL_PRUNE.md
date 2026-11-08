# Plan: Phase 3, the physical prune of bot K 160 at top-k 5

**Status:** 2026-11-07: done, §7 included. The bot runs on the pruned model since 2026-11-06 20:35 (§8.1). The
bot-rank draft is live since 2026-11-07 09:17 (§7.1) · **Author:** colin-spark · **Date:** 2026-11-06
**Related:** `PLAN_MODEL_SPECIALIZE.md` §7 (this plan carries it out), `PLAN_FINAL_REVIEW.md` §8 (the decision),
`PLAN_MODEL_ROUTER.md` §3 (the size cap), `MODEL_SWAP.md` §1 (a new served model)
**Scope:** a new tool `prune_experts.py` and its tests in `~/Projects/dsv4-specialize/`; a new folder
`tp1-prune5-k160/` there; one test boot on port 8890 (about 1 h of downtime). The main server, its `data/tp1` and its
`data/dspark-draft-k64` do not change. The deploy (§8) is your decision.

| Step | Status |
|---|---|
| §3 The tool `prune_experts.py` and unit tests | ✅ 2026-11-06: `prune_experts.py`, `test_prune_experts.py` (6 tests; all 21 tests of the folder pass, also in the server image), `prune_run.sh` |
| §4 Build `tp1-prune5-k160` (no downtime) | ✅ 2026-11-06 18:51–19:02: 46 files rewritten, 28,896 tensors removed (21.27 GiB of tensor data), 73 GB on disk; the main server stayed healthy |
| §5 Offline self-check | ✅ 2026-11-06: PASS. 82,560 expert tensors (43 × 160 × 12) and 5,463 other tensors byte-equal to the source; 86 router tensors in the keep order; `verify_tp1_manifest.py` with checksums passes on 48 files (83.26 GB of tensors). Script for §6: `prune_boot.sh` (set its deadlines first) |
| §6 Test boot: same routing, speed, memory (about 1 h downtime, your go) | ✅ 2026-11-06 19:05–20:04: 74.09 GiB loaded (baseline 95.39), 1,089,707 KV tokens; replay matches the pick within the run noise (§6.1); bot decode 33.13 tok/s, code decode 40.10 tok/s, both inside the noise against the baseline (§6.1) |
| §7 Optional: a new DSpark draft from the bot rank | ✅ 2026-11-07 08:42–09:12: `data/dspark-draft-k64-bot` (the bot's top 64 MTP experts of layer 42); bot decode +4.1%, code +1.5% against the stock draft on the same model, acceptance +4.7% / +3.5%: at the edge of the noise (§7.1). Switched 09:17 |
| §8 Deploy (your decision) | ✅ 2026-11-06 20:31–20:36: `compose-k160t5.yml`, served as `deepseek-v4-flash-0731-k160t5` (and the old name); Hermes switched; smoke test OK (§8.1) |

---

## 1. Goal

Remove the 56 dropped experts of each layer from the files. The pick (`tp1-botmask5-k160`) only hides them from the
router, so it still loads 95.39 GiB. The pruned folder must give the same routing and the same answers as the pick,
and it must load less memory.

## 2. Facts from the research (2026-11-06)

- **The expert files.** The 43 files `exl3-layer-{L}-tp1-rank0.safetensors` hold the routed experts of layers 0–42.
  Each expert is 12 separate tensors: `layers.{L}.ffn.experts.{E}.{w1|w2|w3}.rank0.{trellis|suh|svh|mcg}`.
  So the tool copies the bytes of the kept experts and gives them new names. It needs no new quantization.
- **The owner.** These 43 files are `root:root` with mode 600. Only a root container can read them. The tool runs
  in the server image with `--network none` and the base folder mounted read-only, as the kernel cache copy did.
- **The size.** The expert data is 81.95 GiB. The 56 × 43 dropped experts are 21.25 GiB (exact sum of the tensor
  bytes). The loaded size goes from 95.39 GiB to about 74.1 GiB. See §9, open point 1.
- **The router tensors** are in the carried files. The mask already edited three of them:
  - `carried-001`: `gate.weight` and `tid2eid` of layers 0–2 (hash routing);
  - `carried-002` and `carried-003`: `gate.weight` and `gate.bias` of layers 3–42.
- **The expert count.** vLLM and SparkInfer read `n_routed_experts` from `config.json`. The kernels use it only to
  choose a tile size. The EXL3 loader reads `experts_per_layer` from the rank-sliced metadata, and it stops if that
  value is not equal to `n_routed_experts`. 160 is a multiple of 16.
- **The MTP layer.** The target model skips all `mtp.*` tensors at load (`skip_substrs=["mtp."]`). The DSpark draft
  is a separate folder with its own 64 MTP experts and its own `n_routed_experts`. So the prune does not change the
  MTP tensors, and the current draft works with the pruned target.
- **The boot check.** The entrypoint runs `verify_tp1_manifest.py`: it compares the size and the SHA-256 of each file
  in `rank-sliced-tp1-manifest.json`.

## 3. The tool `prune_experts.py`

`python prune_experts.py --source tp1-botmask5-k160 --out tp1-prune5-k160`, with `--check` for the self-check
alone. It follows the style of `topk_checkpoint.py`: new headers, tensor bytes streamed from the source, never a
whole file in memory.

1. Read the keep map from `MASK.json` of the source: for each layer, the kept ids are the ids not in `drop`, in
   ascending order. The new id of a kept expert is its position in that list (0 … 159).
2. Expert files: for each layer, write a new `exl3-layer-{L}` file with the 12 × 160 kept tensors, renamed to the
   new ids, in the order of the source file. Keep `__metadata__`.
3. `carried-001`: slice the rows of `gate.weight` of layers 0–2 in the keep order. Map each `tid2eid` value to its
   new id. The tool stops if a `tid2eid` value is a dropped id (the mask must have moved all of them).
4. `carried-002` and `carried-003`: slice the rows of `gate.weight` and the entries of `gate.bias` of layers 3–42
   in the keep order. The masked bias (−1e4) leaves with the dropped rows, so the kept bias values are the original
   values. The tool checks this.
5. Other files: the same links as `topk_checkpoint.py` (symlink to the base folder, hardlink for a real file).
6. `config.json`: `n_routed_experts`, `hybrid_tr3_tail.experts_per_layer` and `reap_pruning.kept_experts_per_layer`
   become 160. `quantization_config.json`: `rank_sliced.experts_per_layer` becomes 160.
7. `model.safetensors.index.json`: remove the dropped names, rename the kept names, and set `total_size`.
8. `tier-bitmap-3.0bpw.json`: slice each layer list in the keep order.
9. `rank-sliced-tp1-manifest.json`: the new `bytes` and `sha256` of every rewritten file, and the new `tensor_bytes`
   and `tensor_count`. `EXL3_MANIFEST.json` describes the TP4 source repo (177 files), and the boot does not read it.
   So it stays a link, as in the mask folders. `tier-bitmap-3.0bpw.json` also gets the new `experts_per_layer`,
   `counts`, `total_expert_triplets` and the SHA-256 of the new plan.
10. `REAP_K160_PLAN.json` in the format of `REAP_K216_PLAN.json`: `keep_by_layer` in the original ids of the
    256-expert model (the REAP K 216 map, then the bot keep map). `mtp_keep` stays as in K 216.
11. `PRUNE.json`: the source, the keep map (old id → new id per layer), the byte sums and the self-check result.

Unit tests (`test_prune_experts.py`) on a small synthetic checkpoint, as in `test_topk_checkpoint.py`:

- the kept expert tensors are byte-equal to the source, under the new names;
- the router rows, the bias and `tid2eid` follow the keep map;
- a dropped id in `tid2eid` stops the tool;
- the manifests match the files;
- the vLLM rule `experts_per_layer == n_routed_experts` holds.

## 4. Build `tp1-prune5-k160` (no downtime)

- The build reads about 88 GB and writes about 70 GB. The main server stays up.
- Run it in a root container from the server image: `--network none`, `--memory 8g`, the base folder read-only at its
  host path, and `dsv4-specialize` read-write. At the end, the container sets the owner of the new files to
  `colin-spark`.
- Run it outside 05:00–10:00 on weekdays (the live cron jobs), and not during a speed test.
- Disk: 2.8 TB free. Memory: the build uses page cache only, which the kernel can free.

## 5. Offline self-check

`--check` reads the new folder and the source and confirms:

- each kept expert tensor of the new folder has the same SHA-256 as its source tensor;
- the router rows and the bias of each layer equal the source rows in the keep order;
- each new `tid2eid` value maps back to the source value;
- each manifest entry matches its file;
- `verify_tp1_manifest.py` from the server image passes on the new folder.

## 6. Test boot (about 1 h downtime, your go)

CAUTION: The bot gives no answers until the restore. This step stops the bot's model server.

1. Stop the main server. Boot `tp1-prune5-k160` on port 8890 with the current draft
   (`MASK_NAME=prune5 MASK_K=160`, the same compose file as the mask tests).
2. Read two lines from the boot log: "Model loading took … GiB" and "GPU KV cache size: … tokens".
3. Same routing: run the replay set and compare with `model_eval/results/botmask5-k160.jsonl`. The NLL and the tool
   names must match the pick within the noise (0.23%). A larger difference shows a bug in the rewrite.
4. Speed: the bot decode test and the code decode test. They must match the pick within the noise (4.2%).
5. Stop the test boot and start the main server. The overnight guard stays on.

This step needs no full-run e2e: the same routing gives the same answers, and the pick already passed the full-run
gates and the final review.

### 6.1 Result (2026-11-06)

`prune_boot.sh` stopped the main server at 19:05. The pruned folder was healthy after 210 s. The restore started at
20:04. Log: `~/.hermes-e2e/prune-boot.log`.

- **Memory.** "Model loading took 74.09 GiB" (baseline 95.39 GiB, estimate 74.1 GiB). With the same
  `GPU_MEMORY_UTILIZATION` (0.93), the KV cache grows to 1,089,707 tokens (baseline 295,424). Free memory during the
  replay: about 1.65 GB (the mask runs: about 4 GiB). It stayed above the 1 GiB stop line.

| Replay (60 targets) | NLL | Tool names | Against the baseline |
|---|---|---|---|
| Baseline K 216 | 0.4440 | 0.9277 | — |
| Pick `botmask5-k160` (mask) | 0.4541 | 0.9518 | +2.27%, PASS |
| `prune5-k160` | 0.4524 | 0.9398 | +1.89%, PASS |

- **Same routing.** The pruned folder is 0.38% lower in NLL than the pick. That is 0.15 points over the 0.23% noise
  line, in the better direction. A second view of the noise, per target:

  | Comparison | Change of the NLL sum | Median change per target | Largest | Tool-name changes |
  |---|---|---|---|---|
  | Baseline against its repeat (same model) | −0.22% | 1.64% | 36.9% | 0 |
  | Pick against `prune5-k160` | −0.38% | 1.92% | 28.3% | 1 (target `0d31caa5`) |

  The spread per target has the size of the run noise, and no target gets much worse. A wrong expert map makes the
  NLL clearly worse (the failed masks: +3% to +6%). One cause of the small change: the SparkInfer kernels choose
  their tile size from the expert count, so the order of the sums changes. Verdict: same routing, no bug in the
  rewrite. The strict rule of §6 step 3 misses by 0.15 points in the better direction.

| Speed | `prune5-k160` | Pick (mask) | Baseline | Against the baseline |
|---|---|---|---|---|
| Bot decode | 33.13 tok/s (acceptance 2.04) | 33.98 tok/s | 33.64 tok/s | −1.5%, inside the noise |
| Code decode | 40.10 tok/s | 42.07 tok/s | 38.86 tok/s | +3.2%, inside the noise |

- **The code test.** Against the pick, the code decode is 4.7% lower. The cause is one trial: the pick wrote the same
  `cuda_cpp` answer in all 4 runs (`322277d0`), which the draft predicts well (about 53.5 tok/s). The pruned folder
  writes a different `cuda_cpp` answer in each run (32.5–40.1 tok/s). The other four languages have the same speed
  as on the pick. So the change comes from a different text, not from slower compute.
- `acceptance_c1.py` exits 1 when one trial is below its own gate of 35 tok/s. The pick did the same in 2 of 4 runs.

## 7. Optional: a new DSpark draft from the bot rank

The current draft keeps the 64 MTP experts with the highest REAP rank. The mask lowered the DSpark acceptance (bot
test: 2.25 → 2.09). A draft that keeps the 64 MTP experts with the highest bot rank of layer 42 can raise it.

1. Build `dspark-draft-k64-bot` with `build_dspark_draft.py --source tp1-prune5-k160 --plan REAP_K160_PLAN.json`,
   with `mtp_keep` from the bot rank.
2. In the §6 boot, run the bot decode test once more with the new draft. Keep it only if the acceptance gain is
   larger than the noise.

### 7.1 Result (2026-11-07)

The stock draft keeps 64 of the 216 MTP experts: 39 REAP "tool" specialists of layer 42 plus a REAP fill. By the
2026-11-07 capture (`PLAN_BETTER_RANK.md`), those 39 carry 22% of the routing weight of layer 42. Only 16 of the 64
are in the bot's top 64, which carries 68%. `build_dspark_draft.py` built `data/dspark-draft-k64-bot` (3.0 GB, the
same size) from a plan whose `mtp_ranked` is the bot order of layer 42, with the per-category reservation at 0.

`draft_test.sh` booted the live model (`tp1-k160t5`) with the new draft on port 8890 (08:42, healthy after 180 s),
ran both decode tests, and restored the live server at 09:12. The downtime was 30 minutes.

| Test | Stock draft on the live model (2026-11-06) | Bot-rank draft | Change |
|---|---|---|---|
| Bot decode | 33.13 tok/s, 2.05 accepted per draft | 34.50 tok/s, 2.14 | +4.1%, acceptance +4.7% |
| Code decode | 40.10 tok/s, 2.52 accepted per draft | 40.69 tok/s, 2.61 | +1.5%, acceptance +3.5% |

- Every acceptance position improved in the bot test (0.78 → 0.80, 0.55 → 0.58, 0.36 → 0.39, 0.22 → 0.24). Rust and
  TypeScript gained 2 to 3 tok/s in the code test; CUDA and Go did not move.
- The bot gain (+1.37 tok/s) is just under the 4.2% noise floor (1.42 tok/s). The code gain is inside the noise. By
  the rule of §7 ("larger than the noise"), the draft does not qualify. But it costs nothing:
  - the draft changes no output (speculative decoding is exact);
  - the size is the same;
  - the switch is one mount line in `compose-k160t5.yml` and a 4-minute reboot.

  Both tests moved the same way, and the acceptance moved at every position. Thus the gain is most likely real and
  small. You decide.
- **Switched 2026-11-07 09:14–09:17** on your go: `compose-k160t5.yml` mounts `data/dspark-draft-k64-bot`. Docker
  recreated the container, and it was healthy after 154 s (74.09 GiB, 292,952 KV tokens, reply "OK."). The downtime
  was 2 min 34 s. The backup of the old file is `compose-k160t5.yml.bak.20261107_stockdraft`. To go back, copy it over
  and run `docker compose -f compose-k160t5.yml up -d`.

## 8. Deploy (your decision)

- A deploy changes the model server, so the decision is yours.
- Give the pruned folder its own place and compose file next to the main server. `./start.sh` keeps the stock
  model.
- Use a served name with `deepseek-v4-flash` in it, for example `deepseek-v4-flash-0731-k160t5`. Hermes takes the
  stall timeout and the reasoning rules from that part of the name (`MODEL_SWAP.md` §1.3).
- Put the new name in the three places of `MODEL_SWAP.md` §1.2, setting 1.

### 8.1 Deploy (2026-11-06)

You accepted 74 GiB and said go. Downtime: 20:31:45 to 20:35:04 (3 min 19 s). Log: `~/.hermes-e2e/deploy-k160t5.log`.

- **Folder.** `tp1-prune5-k160` moved to `~/Projects/DeepSeek-v4-Flash-One-DGX-Spark/data/tp1-k160t5` (a rename on the
  same disk). `dsv4-specialize/tp1-prune5-k160` is now a symlink to it. Kernel cache: `cache-k160t5` (a copy of the
  test-boot cache).
- **Compose file.** `compose-k160t5.yml` in the server folder: the config of the test boots (`compose-mask.yml`) with
  port 8888, project `deepseek-v4-flash-k160t5` and the folder and cache above. The stock `compose.yml`, `start.sh`
  and `stop.sh` do not change.
- **Two served names.** `--served-model-name deepseek-v4-flash-0731-k160t5 deepseek-v4-flash-0731` (in
  `EXTRA_VLLM_ARGS`, which comes after the script's own flag). Hermes uses the new name. The old name keeps the
  colinfer gateway (`--route deepseek=…:8888`), the e2e runner (`sandbox.py`) and older scripts working. This is the
  only change to the vLLM flags.
- **Hermes.** Setting 1 of `MODEL_SWAP.md` §1.2 only: `model.default`, `custom_providers[0].model` and the key
  under `custom_providers[0].models`. Settings 2–6 do not change (the same model family and template). Backups:
  `~/.hermes/config.yaml.bak.20261106_203145`, `~/.hermes/.env.bak.20261106_203145`.
- **Checks.** Healthy after 196 s; "Model loading took 74.09 GiB", 1,091,301 KV tokens; both names answer "OK.";
  `hermes chat -q` answers "OK" through the new name; the gateway restarted with Discord connected.
- **The container** is `deepseek-v4-flash-k160t5-deepseek-v4-flash-1`. 2026-11-06 20:39: `overnight_guard.sh`
  (new `MAIN_COMPOSE`, default `compose-k160t5.yml`), `topk_run.sh`, `prune_boot.sh` and the seven e2e scripts in
  `~/.hermes-e2e` (`stepf*.sh`, `final5.sh`) now stop, start and check this container and compose file. The README and
  the comments of `compose-mask.yml` and `compose-capture.yml` say the same. The guard passes a dry run in three cases
  (healthy, booting, down). On 2026-11-07, the shared code of the run scripts moved to `dsv4-specialize/run_lib.sh`, and the pre-switch copies were deleted.
- **KV cache, 2026-11-06 21:07–21:10.** `--kv-cache-memory-bytes 8589934592` in `EXTRA_VLLM_ARGS` pins the KV at 8 GiB
  (292,952 tokens; before: 29.8 GiB). 23.7 GB of host memory stays available for a second model. Backup of the old
  file: `compose-k160t5.yml.bak.20261106_kv30`; the reboot script: `kv8-reboot.sh` (`PLAN_MODEL_ROUTER.md` §3).
- **Boot check, 2026-11-07 15:05–15:12.** A start after a stop failed vLLM's boot check twice: "Free memory on device
  cuda:0 (112.92/121.69 GiB) on startup is less than desired GPU memory utilization (0.93, 113.17 GiB)". More host
  processes ran than at the deploy. With the KV cache fixed, `GPU_MEMORY_UTILIZATION` sets only this check, so it went
  to 0.90. The server was healthy after 110 s (74.09 GiB, 292,952 KV tokens, reply "OK."). The footprint did not
  change. Backup of the old file: `compose-k160t5.yml.bak.20261107_util093`.

**Go back to the stock model:**

1. `cd ~/Projects/DeepSeek-v4-Flash-One-DGX-Spark && docker compose -f compose-k160t5.yml stop`
2. `./start.sh` (up to 20 min).
3. `cp ~/.hermes/config.yaml.bak.20261106_203145 ~/.hermes/config.yaml`, then `hermes gateway restart`.

## 9. Open points

1. **The size cap.** Decision 2026-11-06: you accept 74 GiB. `PLAN_MODEL_ROUTER.md` §3 set about 72 GiB for about
   160 experts. The exact sum gives about 74.1 GiB. The §6 boot measured 74.09 GiB. If the cap is a hard rule,
   K 144 saves about 6.1 GiB more (about 68 GiB), but K 144 has no test results. You decide after §6.
   2026-11-06 20:51: the memory boot at 0.73 and 128k fails the KV check (5.82 GiB free, 6.54 needed); the minimum
   is about 0.75 (`PLAN_MODEL_ROUTER.md` §3).
2. **The KV cache.** With the same `gpu_memory_utilization`, vLLM gives the freed memory to the KV cache. That is
   correct for one model. For the router, lower the value later (`PLAN_MODEL_ROUTER.md`).

## 10. Effort and order

| Step | Time | Downtime |
|---|---|---|
| §3 Tool and tests | Half a day | None |
| §4 Build | About 1 h | None |
| §5 Self-check | About 30 min | None |
| §6 Test boot | About 1 h | About 1 h |
| §7 New draft | About 1 h | Inside the §6 boot |

Order: §3, §4, §5, then §6 in one downtime window, then §8.
