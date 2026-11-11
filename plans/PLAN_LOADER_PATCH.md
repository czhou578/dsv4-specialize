# Plan: a different expert count in each layer (the loader patch)

**Status:** 2026-11-11: plan only. Nothing is built · **Author:** colin-spark · **Date:** 2026-11-11
**Source:** `PLAN_COLD_EXPERTS.md` §5.5. MD, the cut of 486 cold experts, has no measurable cost on the replay in two
windows (+0.18% against R0). A prune of MD frees about 4.29 GiB, but each layer then keeps a different number of
experts (140 to 160). The runtime accepts one number for all layers. This plan patches the runtime.
**Related:** `PLAN_PHYSICAL_PRUNE.md` (the prune tool and the K 160 test boot), `PRUNE.md` (the live model),
`serve-tune/PLAN_SERVE_TUNE.md` §7.3 (the first per-layer K test), `PLAN_FINAL_REVIEW.md` (the deploy chain)
**Scope:** a patch of two image files, built from a diff in this repo and mounted on a test boot. A change of
`prune_experts.py`. One prune folder. One test window on port 8890. No change to the live server or the server repo.

| Step | Status |
|---|---|
| §3 Read the runtime code (no downtime) | ✅ 2026-11-11: five places tie the expert count to one value; one of them costs 1.28 GiB per count (§3) |
| §5.1 `prune_experts.py` with a K per layer (no downtime) | ⬜ |
| §5.2 The patch, its build script and the test compose file (no downtime) | ⬜ |
| §5.3 The kernel check on the GPU (no downtime, about 30 min) | ⬜ |
| §5.4 The prune of MD (no downtime, about 15 min) | ⬜ |
| §6 The test window (about 2 h 30 min downtime) | ⬜ |
| §7 The decision | ⬜ |

---

## 1. Short answer

- **The patch is small but has one trap.** The config, the router and the weight loader need only a per-layer
  lookup. The trap is the MoE runtime: the loader plans one runtime for each distinct expert count, and each runtime
  holds its own prefill arena of 1.28 GiB. MD has 15 distinct counts, so an unpatched runtime would hold about 19 GiB
  of arenas. That is four times the saving.
- **The fix:** one plan for each count, as SparkInfer requires, but one shared scratch buffer for all of them. All 40
  MoE layers of the live model already share one runtime and one scratch, because the layers run one after another.
- **The test:** a GPU check of the kernels with the bot up. Then one window: the patched image on the live K 160
  folder, and on the MD prune. The MD prune must equal the MD mask on the replay, hold about 69.80 GiB, and keep the
  speed.
- **No deploy in this plan.** A deploy also needs the full runs, the judge and the blind review.

## 2. Terms

| Term | Meaning |
|---|---|
| K | The number of routed experts that a layer keeps |
| Count | One distinct K value. MD has 15 counts: 140, 142 to 154, and 160 |
| Plan | The SparkInfer object for one MoE shape. It sizes the scratch, and `bind` checks it against the expert weights |
| Runtime | The loader's cache entry (`_RANK_SLICED_RUNTIMES`): a decode plan, a prefill plan and their scratch |
| Arena | The scratch of the prefill plan: 1277.1 MiB at the live settings (boot log, "arena=1277.1MiB") |
| Uniform folder | A model folder with the same K in each layer, such as the live `tp1-k160t5` |
| The patched image | The pinned server image with the two patched files of §4 mounted over its own |

## 3. What the runtime does now (2026-11-11)

The image is `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489…`. The line numbers below are
in its files. The live folder is uniform 3.0 bpw (`quantization_config.json`: `bits: 3.0`, no `bits_per_expert`), so
it uses the uniform EXL3 path.

| Place | What it does with the expert count | Change for a K per layer |
|---|---|---|
| `vllm/models/deepseek_v4/nvidia/model.py:609-655`, `DeepseekV4MoE.__init__` | Reads `config.n_routed_experts` for the gate rows, the router bias, and the range of the hash table (`tid2eid`) | Read the K of the layer |
| `model.py:735-757`, `_init_fused_moe_experts` | Gives `config.n_routed_experts` to `FusedMoE(num_experts=…)` | Give the K of the layer |
| `model.py:1828-1840`, `get_expert_mapping` | Lists the weight names for ids 0 to `n_routed_experts − 1` | None, if `n_routed_experts` is the largest K: a layer with fewer experts has no tensors for the higher ids |
| `model.py:1918` | `num_routed_experts` from the first MoE layer, for EPLB | None: EPLB is off, and layer 0 has the largest K |
| `vllm/model_executor/layers/quantization/exl3.py:505-558`, `_configure_rank_sliced` | Stores `experts_per_layer` as one integer | Accept an integer or a list of 43 values |
| `exl3.py:609-626`, `rank_sliced_layer_bitrates` | The uniform path returns `(bits,) * experts_per_layer` | Use the K of the layer |
| `exl3.py:1376-1383`, `create_weights` | Stops if `experts_per_layer` is not the `num_experts` of the layer | Compare with the K of the layer |
| `exl3.py:2088-2104`, `_rank_sliced_runtime` | The cache key holds `local_num_experts`, so each count gets its own runtime | Keep the key: each count needs its own plan (next row) |
| `exl3.py:2141-2151` | Each runtime allocates its own decode scratch and its own prefill arena | Take both from one shared buffer for each model and device |
| SparkInfer `moe/fused_moe/_impl.py:902` (the server repo's mounted copy) | `bind` stops if the expert weights come from a plan other than the scratch's plan | None: the patch keeps one plan for each count |

Four more facts:

- **The router accepts any K.** Its fast kernel supports only the counts 1, 2, 4, …, 128, 192, 256 and more
  (`fused_topk_bias_router.py:19-21`). 160 is not one of them, so the live model already uses the general path.
  A K of 140 to 154 uses the same path.
- **The routing has no groups** (`n_group: null`), so K needs no divisor.
- **The draft shares the target config.** `dspark.py:97-119` builds the draft layers with the target's config, and
  sets `n_routed_experts` to the draft's 64 while it builds them. The draft layers have the indices 43 and up. So the
  patch must use the per-layer list only for the layer indices 0 to 42, and `n_routed_experts` for all others. The
  draft has its own quantization config, so its runtime and scratch stay separate.
- **The comment at `exl3.py:141-144`** says that per-layer runtimes "would cost tens of GiB" and "are not
  affordable". The scratch is the cost, not the plan. This plan keeps the plans per count and shares the scratch.

`prune_experts.py` also stops on a different K per layer (`keep_lists`), and it writes one integer to each count
field (`edit_json`).

## 4. The design

### 4.1 The folder format

- `config.json`: `n_routed_experts` is the largest K (160: layers 0–2 keep the live set, because the hash remap of a
  smaller K has no test). A new key, `n_routed_experts_per_layer`, holds the 43 values.
- `quantization_config.json` and `config.json`: `rank_sliced.experts_per_layer` and
  `hybrid_tr3_tail.experts_per_layer` hold the same list in place of an integer.
- `tier-bitmap-3.0bpw.json`: each layer list in the keep order, and `experts_per_layer` as the list.
- A folder without the new key loads exactly as now. So the patched image can serve the live uniform folder.

### 4.2 The patch of `model.py`

1. In `DeepseekV4MoE.__init__`, set `self.n_routed_experts` to the value of the layer, if the config has the list and
   the layer index is below 43. Else keep `config.n_routed_experts` (the draft, and every uniform folder).
2. Use `self.n_routed_experts` for the gate rows, the router bias, the `tid2eid` range and `FusedMoE(num_experts=…)`.
3. Stop with a clear error if the list is present on the MegaMoE path. This model does not use that path.

### 4.3 The patch of `exl3.py`

1. `_configure_rank_sliced` accepts an integer or a list of `last − first + 1` values. A helper
   `experts_for_layer(layer_name)` returns the K of the layer.
2. `create_weights` and `rank_sliced_layer_bitrates` use that helper.
3. The mixed-bitrate path stops with a clear error if it gets a list. The live model does not use that path.
4. **The shared scratch.** A new pool holds one decode scratch and one prefill scratch for each owner scope and device.
   At the first runtime of a model, the patch plans all counts of that model, because the list is in the metadata.
   Then it allocates each pool buffer once, at the largest size of all plans. Each runtime gets a view of the pool
   with the shape of its own plan. The pool never grows, so the pointers in the CUDA graphs stay valid.
5. One log line at the first runtime: the counts, the number of plans, and the pool size in MiB.

**Why the shared scratch is safe:** today all 40 MoE layers of the target share one runtime and one scratch, and the
layers run one after another. The patch keeps that order and that sharing. Only the plan changes from layer to layer.
The side streams of the model belong to the attention (`model.py:874-899`, `DeepseekV4DecoderLayer`), not to the
routed experts.

### 4.4 The files

- `image-patch/per-layer-experts.diff`: the diff of both files against the pinned image.
- `image-patch/build_perlayer.sh`: copies the two files out of the pinned image, applies the diff, and runs
  `py_compile`, as `serve-tune/image-patch/build_head.sh` does. The output files are not in git.
- `compose-perlayer.yml`: an overlay on `compose-mask.yml` that mounts the two built files.

The server repo does not change. A deploy would copy the mounts into its compose file (§7).

### 4.5 The K values

| K values | Counts | Experts removed | GiB freed |
|---|---|---|---|
| Exactly as MD | 15 | 486 | 4.29 |
| MD, each K rounded up to a multiple of 4 | 6 | 428 | 3.78 |
| MD, each K rounded up to a multiple of 8 | 3 | 336 | 2.97 |

A rounded K keeps some experts that MD drops, so it keeps a subset of a tested cut. Use the exact K. Fall back to the
multiple of 4 only if §5.3 shows that each count costs much boot time or memory.

## 5. The work before the window (no downtime)

### 5.1 `prune_experts.py` with a K per layer

1. `keep_lists` accepts a different K in each layer.
2. `edit_json` writes the lists of §4.1, and `n_routed_experts` as the largest K.
3. The self-check compares each count field with the kept experts of each layer.
4. The folder name uses the mean K: `tp1-prunemd5-k149`. `REAP_K149_PLAN.json` holds the keep map.
5. Tests on the synthetic checkpoint of `test_prune_experts.py`: a K per layer, the lists, and a uniform folder that
   stays byte-equal to the output of the current tool.

### 5.2 The patch

1. Write the diff of §4.2 and §4.3, the build script and `compose-perlayer.yml`.
2. Tests in the image, as `serve-tune/image-patch/test_fp8_head.py` runs:
   - a uniform config gives the same counts, the same runtime keys and one plan, as now;
   - a list gives each layer its K, and the draft layer indices get `n_routed_experts`;
   - the pool takes the largest scratch, and each runtime gets a view with its own shape.

### 5.3 The kernel check on the GPU

A script in the patched image builds one MoE layer with random EXL3 codes and runs it on the GPU. The live server
stays up.

**CAUTION:** the check uses about 5 GiB of GPU memory, and the live server leaves about 23 GiB free. Run it only
when the bot is idle, and stop it if the free memory falls below 4 GiB.

1. Build 160 experts. Run the uniform runtime of 160 with top-k ids below 140.
2. Build the first 140 of the same experts, and run the patched runtime of count 140 with the same ids.
3. The two outputs must be equal, bit for bit: the same kernels read the same experts.
4. Run the counts 140, 154, 160 and 140 again in turn on the shared scratch. Each output must be equal to its first
   run. This shows that one count does not damage the scratch of the next.
5. Record the plan time of each count, the pool size, and whether the kernel cache gets new files for each count.
   The test boot mounts the same `./cache`, so this also warms it.

### 5.4 The prune of MD

`prune_experts.py --source tp1-coldmd5-k149 --out tp1-prunemd5-k149` in a root container, as for K 160 (about
11 min, with the bot up). Expect about 69.80 GiB of tensors (74.09 − 4.29). Run the self-check and
`verify_tp1_manifest.py`.

## 6. The test window (about 2 h 30 min downtime)

**CAUTION:** this stops the bot's model server. Start it only on your go. Pause the Hermes cron jobs first, and do
not run it during another test.

| Order | Point | Work | Minutes |
|---|---|---|---|
| 1 | P0: the patched image on `tp1-prune5-k160` (equal to the live folder, file by file) | Boot, smoke, the speed test (5 runs) | 25 |
| 2 | P1: the patched image on `tp1-prunemd5-k149` | Boot, smoke, two replays, the speed test, the prefill probe | 105 |
| 3 | The restore of the live server | | 10 |

- The second replay of P1 needs its own label: the replay skips the targets that its output file holds.
- The boot limit of `run_lib.sh` is 15 min. If §5.3 shows long plan times, warm the cache first or raise the limit
  for P1.
- The window script follows `cold_replay.sh`: the KV pin of 7.25 GiB, `BOOT_MIN_GIB=110`, the memory watch, and the
  deadlines.

### 6.1 The gates

| Gate | P0 | P1 |
|---|---|---|
| Boot | Healthy; 74.09 GiB loaded; one plan in the log line | Healthy; about 69.80 GiB loaded; 15 plans and one pool |
| Memory | The same free memory as the live server | At least 4 GiB more free memory than P0 after the boot |
| Smoke | The JSON schema and both tool choices pass | The same |
| Quality | — | The mean of the two replays is within 0.6% of the mean of the two MD mask replays (`PLAN_COLD_EXPERTS.md` §5.5). 0.6% is the noise of one folder between windows. The baseline gate also passes (NLL ≤ +3% and tool names ≥ 98% against K 216) |
| Speed | The bot decode test within 4.2% of the live server | The bot decode test within 4.2% of P0; the draft acceptance within the same noise |

## 7. The decision

| Result | Next step |
|---|---|
| P0 and P1 pass | The deploy chain of `PLAN_FINAL_REVIEW.md`: the full runs, the judge and your blind review. Then decide the use of the 4.29 GiB (KV room, or the second model of the router plan) |
| P0 passes, P1 fails the quality gate | The prune differs from the mask: compare the tensors of one layer, then test again |
| P1 fails the memory gate | The pool or a plan holds more than §5.3 measured: read the log line, and try the K values rounded to 4 |
| P0 fails | The patch changes the uniform path: fix it before any other test |

After a pass, the larger cut of `PLAN_COLD_EXPERTS.md` §5.5 (all experts below 0.2%, 7.89 GiB) needs no new code.
Each of its points needs two replays.

## 8. Risks

| Risk | Effect | Answer |
|---|---|---|
| A layer index error sends the per-layer list to a draft layer | The draft gets the wrong expert count and stops at boot | §4.2 uses the list only for the indices 0 to 42; a test in §5.2 covers the draft indices |
| The plans compile a kernel for each count | A longer boot, and a larger kernel cache | §5.3 measures it and warms the cache; the fallback is the K values rounded to 4 |
| The SparkInfer tile choice depends on the count (`_impl.py:1568-1590`) | The prefill speed of some layers changes | The speed test and the prefill probe of P1 |
| An image update changes the two files | The diff does not apply | The image is pinned by digest; the build script stops if the diff does not apply |
| The shared scratch is used by two layers at the same time | Wrong outputs | The routed experts run one layer at a time; step 4 of §5.3 tests the sharing |

## 9. The pull requests for the code

Each one stays under 400 lines and passes its own tests:

1. `prune_experts.py` with a K per layer, and its tests (about 150 lines).
2. The diff, the build script and `compose-perlayer.yml` (about 250 lines).
3. The tests of the patch and the kernel check script (about 200 lines).
4. The window script and the README rows (about 120 lines).

## 10. Timeline

| Work | Time | Downtime |
|---|---|---|
| §5.1 to §5.2: the code and the tests | About one day | None |
| §5.3: the kernel check | About 30 min | None |
| §5.4: the prune of MD | About 15 min | None |
| §6: the window | About 2 h 30 min | About 2 h 30 min |
| §7: the report and the decision | About 30 min | None |
