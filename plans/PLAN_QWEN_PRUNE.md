# Plan: prune the experts of Qwen3.6-35B-A3B for the fast route

**Status:** 2026-11-14: K 208 with NVFP4 draft experts (§7.7 to §7.9) uses 25.42 GiB at the peak with a 120k
prompt, at 101.8 tok/s: a margin of 3.4 to 4.1 GiB beside DeepSeek, or 3.7 to 4.4 GiB with the context cap. The
facts and side-by-side test (§7.6): K 208 loses a third of the right answers on 80 rare facts, and a blind judge finds
no loss on your turns. You chose K 208. Phase D (both servers at the same time) waits for your go.
2026-11-13: every pruned point failed G1 (§7.2) ·
**Author:** colin-spark · **Date:** 2026-11-13
**Related:** `PLAN_SECOND_MODEL_FIT.md` (§5.5 the measured fit and speed of the candidates), `PLAN_PHYSICAL_PRUNE.md`
and `PRUNE.md` (the method on DeepSeek), `PLAN_BETTER_RANK.md` (the ranks), `PLAN_MODEL_ROUTER.md` in the bot's repo
**Scope:** a data builder, a capture client, a prune tool and two window scripts in this repo, and three test
windows. The model runs with the vLLM 0.33 venv, alone on port 8889. No change to the DeepSeek server.

| Step | Status |
|---|---|
| §5 Phase A: the data, the tools and the tests (no downtime, about 2 days) | ✅ 2026-11-13: PRs #54 to #57 (§5.1 to §5.3) |
| §6 Phase B: the capture window (about 85 min downtime) | ✅ 2026-11-13, 20:20 to 21:39 (§6.1) |
| §7 Phase C: the builds and the gate window (about 80 min downtime) | ✅ 2026-11-13, two windows (§7.1, §7.2): no point passes G1 |
| §8 Phase D: both servers at the same time (`PLAN_SECOND_MODEL_FIT.md` §7) | ⬜ |

---

## 1. Short answer

- **The goal:** `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` with its MTP draft, at a context of 128,000 tokens, beside the
  live DeepSeek checkpoint, with a free margin of at least 4 GiB. You chose this model for its balance of reasoning
  and speed: 99 tok/s with the draft, against 35 tok/s for DeepSeek.
- **The gap:** with the draft it uses 32.3 GiB at 256k. About 29 GiB is available. A context of 128k frees about
  1.5 GiB. The rest must come from the weights.
- **The lever:** the routed experts hold 77% of the weights. A prune from 256 to 176 experts per layer frees
  5.7 GiB, with the experts of the draft. Then the model needs about 25 GiB, and the margin is about 4 GiB with no
  change to DeepSeek.
- **The method** is the method of the DeepSeek prune: record the experts that your own traffic uses, rank them, remove
  the cold ones, and gate the result against the full model.
- **It is simpler here.** Each expert is a set of plain tensors, vLLM 0.33 returns the routed experts with a flag,
  and a pruned folder with one expert count for all layers needs no loader patch.
- **The cost is not known.** DeepSeek lost 1.9% NLL at −26% of its experts and failed at −33%. Only the gate window
  gives the number for Qwen. A likely result is that 192 passes and 160 fails.
- **Your traffic:** coding, web search and tool calls, with Hermes or your own agent interface.

## 2. The numbers (measured)

### 2.1 The checkpoint (from the safetensors headers, 2026-11-13)

| Part | GiB | Note |
|---|---|---|
| Routed experts | 16.88 | 40 layers × 256 experts × 1.69 MiB. 12 tensors per expert (NVFP4 packed weights and scales) |
| MTP draft | 1.57 | One layer. Its 256 experts are stacked BF16 tensors: 1.50 GiB |
| Attention | 1.20 | |
| Embedding, head | 1.42 | Vocabulary 248,320 |
| Vision encoder | 0.83 | Not loaded (`--language-model-only`) |
| Router gate | 0.04 | `gate.weight`, BF16, 256 × 2,048 per layer |

Each token uses 8 of the 256 experts of a layer (`num_experts_per_tok`).

### 2.2 The run (measured in the fit window, `PLAN_SECOND_MODEL_FIT.md` §5.5)

| Setup, 256k context | Weights | KV need | Used at the peak | Bot decode |
|---|---|---|---|---|
| No draft | 19.65 | 2.64 | 28.81 | 57.9 tok/s |
| MTP draft, K 3 | 21.22 | 3.10 | 32.29 | 99.3 tok/s, 2.28 tokens accepted per step |

### 2.3 What a prune frees

One expert slot (one expert in each of the 40 layers) is 66 MiB. The draft layer must use the same expert count,
because vLLM reads one `num_experts` for the model and its draft. One slot of the draft is 5.9 MiB.

| Experts per layer | Target experts | Draft experts | Sum | Use at the peak, with the draft, at 128k | Margin beside DeepSeek (29 GiB) |
|---|---|---|---|---|---|
| 256 (now) | 0 | 0 | 0 | About 30.7 | None |
| 192 (−25%) | 4.22 | 0.38 | 4.60 | About 26.1 | About 2.9 |
| 176 (−31%) | 5.27 | 0.47 | 5.74 | About 25.0 | About 4.0 |
| 160 (−37%) | 6.33 | 0.56 | 6.89 | About 23.9 | About 5.1 |

The DeepSeek context cap (`PLAN_SECOND_MODEL_FIT.md` §3.1, D1) adds about 0.35 GiB to each margin.

**Correction (2026-11-14).** This page first gave D1 as 1.1 GiB, and the margins with D1 in §6.1, §7, §7.2 and §7.4
used that value. The 1.1 GiB is the saving against the old KV pin of 8 GiB (`serve-tune/PLAN_SERVE_TUNE.md` §3). The
live pin has been 7.25 GiB since 2026-11-10, and the pin for 160k is about 6.9 GiB (most of the KV need of DeepSeek
is fixed: about 6.5 GiB at 128k and 7.0 GiB at 256k). So D1 frees about 0.35 GiB. The margins with D1 below use
0.35 GiB.

## 3. The data

The prune keeps the experts that the calibration data uses. So the data must look like the traffic of the fast
route: an agent prompt, tool definitions, code, tool calls and their results, web search results, and reasoning.

| Source | What it gives | State |
|---|---|---|
| Your Claude Code transcripts (`~/.claude/projects`, with your yes of 2026-11-13) | Real coding sessions: 106 sessions, about 14,900 tool calls (shell, edits, file reads, web fetches and searches) and their results, about 11M tokens of text | On disk |
| The bot's recorded sessions (`model_eval/data/sessions.jsonl`) | Web search and tool calls in the Hermes format, with the Hermes system prompt and tools | On disk, already split |
| Your repositories: `v3`, `stock-research-agent`, `model-benchmarks`, `llm-reader`, `train-gpt2`, `nanoGPT-inference`, `colinfer`, `vitesse-platform` | Your code, as the content of file reads and as coding tasks | 6 on disk; `nanoGPT-inference` to clone |
| Qwen's own output in the capture window | The model's own reasoning and tool-call style. The capture counts generated tokens too | Phase B |

Rules for the data:

- **The transcripts are converted, not copied.** Each session becomes chat messages in the OpenAI format: a tool call
  of the transcript becomes an assistant tool call, and its result becomes a tool message. The system prompt and the
  tool definitions are those of Hermes, because that is the agent of the fast route. Thinking blocks and images are
  left out.
- **The assistant turns of the transcripts count as calibration.** The routing depends on the tokens, not on the
  model that wrote them. So a recorded tool call or code edit exercises the experts that write such text.
- **The data stays local and out of git.** The transcripts can hold secrets. The builder writes to a folder that
  `.gitignore` covers, and nothing leaves the machine.
- **The split is by project.** Calibration: most sessions of each project. Held-out: whole sessions that the
  calibration never sees, and one repository (`nanoGPT-inference`) that no calibration session touches.
- **The size:** about 1.5M to 2M calibration tokens, as for DeepSeek (1.75M), balanced over the projects. The
  prefill of Qwen runs at about 1,900 tokens per second, so the capture takes about 15 to 20 minutes.

## 4. The gates

The reference is the full model (K 256) in the same window. A pruned model must pass all of these:

| Gate | Measure | Limit |
|---|---|---|
| G1 replay NLL | The full model writes 60 held-out turns at temperature 0 (the recorded set). The pruned model scores them with prompt logprobs, as the DeepSeek replay does | At most +3% summed NLL |
| G2 tool names and decisions | On the same 60 turns: every token of each tool name ranks first; the first answer token ranks first | Tool names at least 98% of the reference; decisions down by no more than 3 |
| G3 tool-call suite | The tool-calling benchmark of `model-benchmarks` (85 tasks) | No more than 3 tasks below the reference of the same window |
| G4 smoke | A short request and a tool call (`qwen_fit.py --smoke`) | Pass |
| G5 speed with the draft | The bot decode test with the MTP draft, against K 256 with the draft in the same window | Report. A loss above the noise is your decision: a pruned target can lower the acceptance of the draft |
| G6 memory | The use at the peak at 128k with the draft | A margin of at least 4 GiB beside DeepSeek, with the DeepSeek savings that you accept |

- The replay noise of one run is up to 0.56% on DeepSeek. A point within 0.5% of the NLL limit gets a second replay,
  and the mean decides.
- G1 measures how far the pruned model moves from the full model. It does not measure whether the full model is good
  enough for your work. That is the skill test of the router plan.
- The last gate is your own use: a few days of real work on the pruned model before it becomes the default.

## 5. Phase A: the data, the tools and the tests (no downtime)

| Tool | Use |
|---|---|
| `qwen_data.py` | Builds the calibration and held-out files from the three sources of §3. `--stats` prints the sessions, the tokens per project and the tool mix. No model is needed |
| `capture.py` (change) | The expert count becomes an option (now 216). The data folder and the model name already are options |
| `qwen_prune.py` | Copies the checkpoint with K experts per layer: it keeps the top K of each layer by the rank, renumbers them, cuts the rows of `gate.weight`, slices the stacked tensors of the draft, and writes `num_experts` to the config. It copies bytes and does no new quantization. `--check` compares each kept tensor with its source |
| `qwen_capture.sh` | The window of §6 |
| `qwen_gate.sh` | The window of §7 |

- The rank is the weight rank of `PLAN_BETTER_RANK.md` if the reply holds the router weights, else the count rank.
- Tests: the conversion of a transcript (a tool call, a tool result, a skipped thinking block, no secret-shaped
  field copied from the metadata), the split (no session in both sets), the prune on a tiny synthetic checkpoint
  (kept tensors equal their source byte for byte, the gate rows match the kept ids, the draft slices match), and dry
  runs of both window scripts with stub boots.
- Each pull request changes at most 400 lines: the data builder; the capture change and the prune tool; the two
  window scripts.

### 5.1 The data builder (2026-11-13)

`qwen_data.py` writes `qwen-data/sessions.jsonl` and `tools.json` (not in git) in under 2 seconds. The result on this
machine, in estimated tokens (characters / 4):

| Source | Calibration | Held-out |
|---|---|---|
| Claude Code transcripts | 33 sessions, 37 segments, 1.13M tokens, from 13 projects | 13 sessions, 25 segments, 0.89M tokens |
| The bot's sessions | 22 sessions, 26 segments, 0.63M tokens | 18 sessions, 24 segments, 1.00M tokens |
| Repository files | 42 files of 7 repositories, 0.09M tokens | 6 files of `nanoGPT-inference` |

- The calibration holds about 1.86M tokens. A budget per source (`--budget`) takes one segment at a time from each
  project in turn, so the largest project (2.2M tokens of transcripts) does not fill it alone.
- The tool calls of the transcript part: 1,901 `terminal`, 248 `patch`, 97 `write_file`, 77 `read_file`, 21
  `web_extract`, 3 `web_search`. The web searches come mostly from the bot's sessions.
- No tool result is left without its call, the longest segment has about 91,000 tokens, and the secret pattern
  replaced 18 strings.
- The system prompt is the Hermes prompt of the bot's data. Give me the prompt of your coding agent when it exists,
  and the builder takes that.

### 5.2 The prune tool (2026-11-13)

`qwen_prune.py` ran on the real checkpoint with a dummy rank (no downtime, no model boot):

- K 176 wrote 16.25 GiB in 75 seconds, with the check. That is 5.74 GiB less than the source, as §2.3 says.
- It kept 85,956 tensors and dropped 38,400 (80 experts × 40 layers × 12 tensors). Every kept tensor equals its
  source byte for byte.
- The safetensors library opens all five files. The router has 176 rows in each layer, and the stacked tensors of
  the draft have 176 experts.
- The tool holds one source file in memory at a time (at most 4.7 GiB).
- `capture.py --experts 256` sets the expert count (the default stays 216).

What only a boot can show: that vLLM 0.33 loads the folder. The gate window boots it first.

### 5.3 The clients and the window points (2026-11-13)

- `qwen_gate.py`: the record, the score and the comparison of §4 (G1, G2). On the real data, the pick gives 60
  distinct targets: 30 from the transcripts, 24 from the bot's sessions and 6 from the held-out repository.
- `capture.py --generate 256`: after each segment the model writes up to 256 tokens, and their routing counts.
- The windows are points of `qwen_fit.sh`, not two new scripts: the script already has the boot, the KV pin retry,
  the memory samples and the restore. `c1`, `c2` and `c3` are the capture window. `g256`, `g192`, `g176` and `g160`
  are the gate window. Each point has its own context (128k), tests and time need.
- vLLM 0.33 accepts the four new command lines (its argument parser, no boot).
- **No `suffix` point.** The suffix draft of vLLM needs the package `arctic_inference`, which is not in the venv. The
  window tests the `ngram` draft only.
- The sampling of the recorded set is the thinking-mode setting of the Qwen model card (temperature 0.6, top-p 0.95,
  top-k 20) with a fixed seed per turn. The +3% limit of G1 comes from the DeepSeek replay, which also scored sampled
  turns. Read it as a first threshold: the window reports the number, and a result near the limit is your decision.

The commands:

```
# Phase B, the capture window (about 100 min of downtime)
POINTS='c1 c2 c3' START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... ./qwen_fit.sh
# after it, with no downtime: the rank and the builds
python capture.py --rank results/qwen-capture.jsonl results/qwen-rank.json
python qwen_prune.py --source <snapshot> --rank results/qwen-rank.json --keep 176 --out qwen-k176
# Phase C, the gate window (about 100 min of downtime)
POINTS='g256 g176 g192 g160' START_AFTER=... BOOT_BY=... SPEED_BY=... RESTORE_AT=... ./qwen_fit.sh
```

Two questions that Phase A cannot answer, with the fallback for each:

| Question | Check | Fallback |
|---|---|---|
| Does the capture cover the draft layer? | The layer count of the routed experts with the draft on (40 or 41) | A small hook that sums the choices of the draft layer, as the capture patch of DeepSeek does |
| Does vLLM 0.33 return routed experts with its default model runner? | The first boot of the capture window | Model Runner V1 by the environment variable, as for DeepSeek |

## 6. Phase B: the capture window (about 85 min downtime)

All boots: the full model, 128k context, one sequence, port 8889, the default loader.

| Order | Work | Minutes |
|---|---|---|
| 1 | Boot with `--enable-return-routed-experts`, no draft | 5 |
| 2 | The capture over the calibration set: the prompt tokens, and up to 512 generated tokens per segment | 25 |
| 3 | The recorded set: 60 held-out turns at temperature 0 | 10 |
| 4 | The tool-call suite: the reference of G3 | 10 |
| 5 | Boot with the MTP draft; the capture of the draft layer on a part of the set; the bot decode test: the reference of G5 | 15 |
| 6 | One point with the `ngram` draft, which needs no weights: a boot and the bot decode test | 9 |
| 7 | The restore of the live server | 3 |

Step 6 tells whether a draft without weights gives a useful part of the gain of the MTP draft. If one does, the
prune can be smaller.

**Gate.** The capture has at least 1.5M tokens with no failed segment, the rank file has 256 experts for each of the
40 layers (and for the draft layer), and the reference files of G1, G3 and G5 exist.

After the window, with no downtime: the report of the routing (the share of the tokens that the top 160, 176 and
192 experts of each layer cover), and the builds. A coverage that is far lower than on DeepSeek at the same ratio is
an early sign that the prune costs more.

### 6.1 Result (2026-11-13, 20:20 to 21:39, on your "start the capture window")

The live server stopped at 20:20 and was healthy again at 21:39:29 (reply "OK.", gateway active). The downtime was
79 minutes. Files (not in git): `results/qwen-capture.jsonl`, `results/qwen-rank.json`, `results/qwen-recorded.jsonl`,
`results/qwen-fit-c{1,2,3}*`, `logs/qwen-capture.log`.

**A fault, fixed in the window.** Every capture request of the window failed: the Qwen 3.5/3.6 chat template refuses a
message list with no user message ("No user query found in messages."), and the capture tokenized the shared prefix
as the system message alone. The DeepSeek template accepted that. The fix adds a short user message to that list;
the common prefix stops where the two token lists differ, so it does not count. I reran the capture by hand on the
same boot (c1), beside the window's other tests, and stopped the window's failing capture process.

| Point | Result |
|---|---|
| c1: the capture | 99 of 99 segments, 40 layers, 8 experts per token, no unfilled row. 2.44M prompt tokens counted (each shared prefix once), and 20,450 tokens that Qwen wrote, with 174k tokens of tool results after the targets |
| c1: the recorded set | 60 turns of the full model: 48 with tool calls, all with reasoning, 377 tokens on average, 12 at the limit of 800 tokens |
| c1: the tool-call suite (G3 reference) | 51 of 85 tasks, composite 4.93 |
| c1: the bot decode, no draft | 58.8 tok/s |
| c2: the MTP draft | 98.2 tok/s, 2.24 tokens accepted per step. The use at the peak at 128k: **29.9 GiB** (the load: 30.0 GiB) |
| c2: the capture with the draft | 40 layers: vLLM does not return the routing of the draft layer |
| c3: the `ngram` draft | 62.7 tok/s, 0.47 tokens accepted per step: little gain over no draft |

c1's peak (31.4 GiB) holds the memory of the capture clients, which read replies of up to 90k tokens. The peak of c2
is the clean number for the model with its draft at 128k.

**The coverage of the rank** (the share of the routed slots that the top K experts of a layer carry):

| Point | Mean over the layers | Lowest layer (layer 2) |
|---|---|---|
| Qwen K 208 of 256 | About 96% | About 88% |
| Qwen K 192 of 256 | 93.9% | 82.8% |
| Qwen K 176 of 256 | 91.0% | 77.7% |
| Qwen K 160 of 256 | 87.6% | 72.4% |
| DeepSeek K 160 of 216, the live prune (passed: +1.9% NLL) | 95.4% | 86.5% |
| DeepSeek K 144 of 216 (failed: +3.3% to +3.9%) | 92.5% | 81.9% |

- Every expert of every layer is routed at least once. The routing of this traffic is spread more evenly than on
  DeepSeek: K 176 keeps less than the DeepSeek point that failed. By this measure, K 192 is near the limit, and K 176
  and K 160 are likely to fail. The measure is a proxy: Qwen mixes 8 experts per token, DeepSeek 6.
- The transcripts and the bot's sessions agree on 154 of their top 176 experts per layer.

**Three changes for Phase C:**

1. **The draft stays whole.** vLLM does not report the routing of the draft, so there is no rank for its experts.
   vLLM builds an MTP draft from the `model` field of its speculative config, so the pruned points take the draft
   from the original snapshot, with all 256 experts. The draft then saves nothing (0.47 GiB less than §2.3 at K 176),
   and a speed change in the gate window comes from the target prune alone. `qwen_prune.py --draft keep` copies the
   draft tensors unchanged.
2. **K 208 replaces K 160.** K 160 keeps far less than the failed DeepSeek point. K 208 gives a safer point.
3. **The margins with the measured peak** (29.9 GiB at 128k with the draft; about 29 GiB available beside DeepSeek):

| Point | Saving | Use at the peak | Margin |
|---|---|---|---|
| K 208 | 3.16 GiB | About 26.7 GiB | About 2.3 GiB (2.7 with the DeepSeek context cap) |
| K 192 | 4.22 GiB | About 25.7 GiB | About 3.3 GiB (3.7 with the cap) |
| K 176 | 5.27 GiB | About 24.6 GiB | About 4.4 GiB |

The builds (no downtime): K 176 (16.72 GiB), K 192 (17.78 GiB) and K 208 (18.84 GiB), each in about 75 seconds, and
every kept tensor equals its source.

## 7. Phase C: the builds and the gate window (about 3 hours downtime)

Builds (no downtime): K 208, K 192 and K 176 (§6.1), with `--check`.

Each point runs the smoke test, the score of the 60 recorded turns, the tool-call suite (33 min on c1) and the bot
decode test. The window gives each point 45 minutes. K 176, the point most likely to fail (§6.1), runs last: if the
time runs out, the window skips only that point.

| Order | Point | Tests | Minutes |
|---|---|---|---|
| 1 | K 256 with the draft, 128k | G3, G4, G5, G6, and the score: the reference of the window | 45 |
| 2 | K 208 with the draft (in place of K 160, §6.1) | G1 to G6 | 45 |
| 3 | K 192 with the draft | G1 to G6 | 45 |
| 4 | K 176 with the draft | G1 to G6 | 45 |
| 5 | The restore of the live server | | 3 |

How to read the result:

| Result | Next step |
|---|---|
| K 176 passes all gates | Phase D with K 176 |
| K 192 passes and K 176 fails | Phase D with K 192 (margin about 3.3 GiB; 3.7 with the DeepSeek context cap, D1) |
| Only K 208 passes | Phase D with K 208 and D1 (margin about 2.7 GiB). Your decision on the smaller margin |
| No point passes G1 or G3 | Stop the prune. The choices are Qwen without its draft at 128k (about 27.4 GiB), or Nemotron |
| A point passes the quality gates, but the speed with the draft falls | Rebuild the draft experts by the rank of the pruned target, as the bot-rank draft of DeepSeek. Else your decision |

### 7.1 Gate window 1 (2026-11-13, 21:50 to 22:23)

The live server stopped at 21:50 and was healthy again at 22:22:53 (reply "OK.", gateway active): 33 minutes of
downtime. Files (not in git): `results/qwen-fit-g256*`, `logs/qwen-gate.log`, and the two failed boots in
`results/qwen-gate1-g{208,192}-server.log`.

**The reference (g256: the full model with the MTP draft, 128k):**

| Measure | Value |
|---|---|
| The score of the 60 recorded turns | NLL 0.3144 per token; 89.3% of the tokens rank first; 60 of 60 decisions; 100% of the tool names |
| The tool-call suite | 52 of 85 tasks, composite 5.06 (c1 in the capture window: 51 of 85). With the draft, the suite took 10 minutes |
| The bot decode test | 97.8 tok/s (c2: 98.2) |
| The use at the peak | 31.69 GiB, with the score and suite clients (c2 with no clients: 29.9 GiB) |

**The fault.** K 208 and K 192 did not boot ("IndexError: index 208 is out of bounds for dimension 0 with size
208"), and the window restored the live server after two failed boots in a row, as designed. The cause is in vLLM
0.33: the proposer builds the draft with the `vllm_config` of the target, and `Qwen3_5MTP` reads `num_experts` from
the target config. The `model` field of the speculative config changes only the folder of the draft weights. The draft
of 256 experts then goes into tensors of 208 rows. In stock vLLM, the draft of a pruned target must have the same
expert count as the target.

**The fix: `qwen_mtp_patch/sitecustomize.py`.** On `PYTHONPATH`, it wraps `Qwen3_5MTP.__init__`: when the draft config
has another expert count than the target, the draft gets a shallow copy of `vllm_config` with a copy of the text
config that has the expert count of the draft. It changes nothing else, and nothing when the counts are equal. It was
checked with the real configs of vLLM 0.33 (the draft view has 256 experts, the target keeps 208, and the quant and
compile settings stay shared). The first GPU boot with the patch is in window 2. For Phase D, the server of the
pruned model needs the same `PYTHONPATH`.

**Gate window 2:** the points g208, g192 and g176 with the patch; p208, p192 and p176 (the same folders with no
draft: smoke, score and tool-call suite) run only when their g point has no score, so the quality gates G1 to G3 are
measured even if the patch fails. The draft does not change the outputs of the target, so G1 to G3 do not depend on
it. G5 and G6 need the draft. The last point, r256, scores the full model again, for the noise between the two
windows. The pruned points are compared with g256 of window 1: the vLLM build, the flags and the recorded set are the
same.

### 7.2 Gate window 2 and the decision (2026-11-13, 22:30 to 23:56)

The live server stopped at 22:30 and was healthy again at 23:55:47 (reply "OK.", gateway active): 85 minutes of
downtime. Files (not in git): `results/qwen-fit-{g208,g192,g176,r256}*`,
`logs/qwen-gate2.log`. The patch of §7.1 worked: each pruned point booted with the draft of 256 experts on the first
try ("qwen_mtp_patch: the MTP draft keeps 256 experts (the target has 208)"), so the p points did not run.

| Point | Loaded | NLL change (G1, limit +3%) | Tokens that rank first | Decisions, tool names (G2) | Tool-call suite (G3) | Speed with the draft (G5) | Use at the peak (G6) |
|---|---|---|---|---|---|---|---|
| K 256, window 1 (g256, the reference) | 21.22 GiB | | 89.3% | 60 of 60, 100% | 52 of 85 | 97.8 tok/s | 31.69 GiB |
| K 256, window 2 (r256) | 21.22 GiB | −0.10% | 89.2% | 60 of 60, 100% | | 98.5 tok/s | 31.14 GiB |
| K 208 | 18.09 GiB | **+4.15%** (+4.25% against r256) | 88.7% | 60 of 60, 100% | 50 of 85 | 101.2 tok/s | 28.85 GiB |
| K 192 | 17.00 GiB | **+8.04%** (+8.15%) | 88.4% | 60 of 60, 100% | 51 of 85 | 101.4 tok/s | 27.55 GiB |
| K 176 | 15.98 GiB | **+11.11%** (+11.23%) | 88.2% | 60 of 60, 100% | 51 of 85 | 94.2 tok/s | 26.56 GiB |

The use at the peak holds the score and suite clients; the clean peak of the full model with its draft at 128k is
29.9 GiB (§6.1). Each point frees in use what it frees in weights.

**What the numbers say:**

- **The loss is real.** The full model repeats within 0.1% between the windows, so +4.15% is 40 times the noise.
- **The loss is broad.** At K 208, 37 of the 60 turns get worse, and most skills lose 4% to 6%: the repositories,
  the stock research, the subagents, and the held-out repository (nanoGPT-inference, +4.0%). No single kind of work
  causes it, so a narrower calibration does not fix it.
- **Qwen loses more than DeepSeek at the same coverage.** K 208 keeps about 96% of the routing and loses 4.2%; the
  live DeepSeek prune keeps 95.4% and loses 1.9%. Qwen is smaller and mixes 8 experts per token from a more even
  routing, so each expert carries more.
- **The decisions, the tool names and the tool-call suite do not move** (50 to 52 of 85, inside the noise of the
  suite). The loss is in the wording of Qwen's turns, not yet in the choice of a tool. The gates of §4 put G1 first,
  as on DeepSeek, because a broad NLL loss shows up in long sessions before it shows up in a short suite.
- **The patched draft keeps its speed** at K 208 and K 192 (about 101 tok/s). At K 176 the acceptance falls (94 tok/s).

**The decision: no prune.** The first 48 experts cost +4.15% and the next 16 cost +3.9%, so G1 holds only above about
K 224. K 224 saves about 2.1 GiB, and the full model with its draft at 128k leaves about −0.9 GiB (29.9 GiB against
about 29 GiB available): the margin at K 224 is about 1.2 GiB, or about 1.6 GiB with the DeepSeek context cap (D1).
G6 asks for 4 GiB. No K passes both G1 and G6, so nothing is accepted and nothing is deployed. On your request,
the folders `qwen-k176`, `qwen-k192` and `qwen-k208` were deleted on 2026-11-14. The rank file
(`results/qwen-rank.json`) stays, and `qwen_prune.py --draft keep` rebuilds a folder in about 75 seconds.

**The choices for the fast route, none taken:**

1. **Nemotron-3.5-Lightning-30B-A3B** (`PLAN_SECOND_MODEL_FIT.md` §5.5): 74 tok/s in 25.5 GiB with no draft at 256k
   (a margin of about 3.5 GiB), 127 tok/s in 28.8 GiB with its DSpark draft. Its quality on your work is not tested.
2. **Qwen3.6-35B-A3B in full with a shorter context or no draft:** 58.8 tok/s with no draft at 128k (about 27.4 GiB),
   or the draft at a context short enough for the margin.
3. **K 208 with a looser G1:** it passes every other gate and saves 2.8 GiB, but its margin is about 2.3 GiB (2.7
   with D1). This changes a gate, so it is your decision.

### 7.3 The overhead window: the memory of vLLM itself, then K 208 (your goal of 2026-11-14)

Your goal: test whether the overhead of vLLM frees memory; if it frees nothing, test K 208 with the draft.

**What the overhead is.** The full model with its draft at 128k uses 29.9 GiB at the peak: 21.22 GiB of weights
(1.57 GiB of them the draft), 1.83 GiB of KV pin, and about 6.9 GiB more. Of the 6.9 GiB, 6.5 GiB stays after the
boot with no request, so a request adds little. `nvidia-smi` gives the engine process 25.2 to 26.8 GiB: about 2 to 3.5
GiB more than the weights, the pin and the CUDA graphs (0.24 GiB). The rest, about 4.5 GiB, is outside the GPU
allocations: mostly the two processes of `vllm serve` (the API server and the engine core) with Python, PyTorch, the
CUDA libraries and the tokenizer. vLLM already frees the allocator cache after the warm-up.

**The levers.** Two levers drop out before the window: `vllm serve` always runs the engine core in its own process
(`VLLM_ENABLE_V1_MULTIPROCESSING=0` applies to the offline engine only), and a shorter context saves little (the KV
pin at 128k is 1.83 GiB, because only 10 of the 40 layers use attention). The window tests the rest, all on the full
model with its draft at 128k, each with the smoke test, one prompt of 120,000 tokens and the bot decode test. Each
point also logs the resident and peak resident memory of each server process.

| Point | Change from c2 | What it can free |
|---|---|---|
| o0 | None: the baseline of this window | |
| o1 | Prefill budget 512 (from 2,048) | The activations of a prefill step. Prefill gets slower |
| o2 | `--enforce-eager` | The CUDA graphs and the torch.compile code. Decode gets slower |
| o3 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | The fragments that the allocator holds |
| o4 | o1 + o3 | Both, with no loss of decode speed |
| k0 | K 208 with the draft of 256 experts (`qwen_mtp_patch`), the settings of c2 | The prune: 3.13 GiB of weights |
| k1 | K 208 with the settings of o4 | The prune and the levers |

K 208 runs in the same window, whatever the result of the o points: the auto mode of Claude Code lets only you start a
window, and you are asleep. The K 208 folder was rebuilt for this window (`--draft keep`, every kept tensor equal to
its source).

**The decision rule, written before the window:**

- A lever frees memory when its peak is at least 0.75 GiB below o0 and its decode rate is within 5% of o0.
- If the full model with its draft and the levers that free memory leaves a margin of at least 4 GiB beside DeepSeek
  (the memory that the window logs before the first stop, alone or with the context cap D1), that is the choice: no
  loss of quality.
- Else, K 208 with the draft (k0 or k1) is the choice to report, with its margin and its G1 loss (+4.15%, §7.2). The
  G1 limit is your decision. Nothing is deployed.

### 7.4 Result of the overhead window (2026-11-14, 00:08 to 00:56)

The live server stopped at 00:08 and was healthy again at 00:56:20 (reply "OK.", gateway active): 48 minutes of
downtime. The memory available beside DeepSeek: 28.83 GiB before the stop (29,524 MiB in `free -m`), three minutes after
the K 208 build wrote 19 GB; 29.56 GiB at 00:05 and 29.48 GiB at 00:58. The margins below use this range,
28.8 to 29.5 GiB. Files (not in git): `results/qwen-fit-{o0..o4,k0,k1}*`, `logs/qwen-overhead.log`.

| Point | Change | Use at the peak | GPU process | Long prompt (119,745 tokens) | Decode with the draft |
|---|---|---|---|---|---|
| o0 | None (full model with its draft) | 29.64 GiB | 25.2 GiB | 46.1 s | 101.8 tok/s |
| o1 | Prefill budget 512 | 29.60 GiB | 25.0 GiB | 59.0 s | 98.9 tok/s |
| o2 | `--enforce-eager` | 29.62 GiB | 25.3 GiB | 51.6 s | 72.6 tok/s |
| o3 | Expandable segments | 29.71 GiB | 25.2 GiB | 46.1 s | 96.5 tok/s |
| o4 | o1 + o3 | 29.57 GiB | 25.0 GiB | 59.6 s | 99.7 tok/s |
| **k0** | **K 208 with the draft of 256 experts, the settings of o0** | **26.51 GiB** | **22.1 GiB** | **45.5 s** | **97.7 tok/s** |
| k1 | K 208 with the settings of o4 | 26.60 GiB | 21.8 GiB | 57.5 s | 99.2 tok/s |

**The overhead frees nothing.** No lever moves the peak by more than 0.07 GiB (the rule asks 0.75 GiB), and eager mode
costs 29% of the decode speed. The overhead (about 6.6 GiB at the peak, 6.2 to 6.6 GiB after the boot) is a fixed
cost in two parts:

- **About 2.2 GiB in the engine process on the GPU**, beyond the weights and the KV pin: the CUDA context, the
  kernel workspaces and the CUDA graphs (0.24 GiB). The prefill budget, the graphs and the allocator do not change it.
- **About 4.4 GiB of host memory of the two processes of `vllm serve`**: the API server holds 1.9 GB resident and the
  engine core 3.1 GB (7.2 GB during the load), for Python, PyTorch, the CUDA libraries and the tokenizer. `vllm serve`
  cannot run the engine in its own process.

**K 208 with the draft (k0) is the choice to report.** Its use at the peak, with one prompt of 120k tokens and the bot
decode test, is 26.51 GiB: 3.13 GiB below o0, as the prune predicts. Its margin beside DeepSeek:

| Setup | Peak | Margin (28.8 to 29.5 GiB available) | With the DeepSeek context cap D1 (+0.35 GiB) |
|---|---|---|---|
| Full model with its draft (o0) | 29.64 GiB | −0.8 to −0.1 GiB | −0.5 to +0.2 GiB |
| K 208 with its draft (k0) | 26.51 GiB | 2.3 to 3.0 GiB | 2.7 to 3.4 GiB |

K 208 with the draft and D1 does not reach the 4 GiB of G6: read it as a margin of about 3 GiB. Its quality cost is the
G1 loss of +4.15% (§7.2); the decisions, the tool names and the tool-call suite do not move, and the draft keeps its
speed (97.7 to 101 tok/s). k1 gains nothing over k0, so k0 keeps the default settings.

**What K 208 needs, if you choose it** (none of it is done): the K 208 folder (`qwen_prune.py --draft keep`, about 75
seconds; this window's copy was deleted after the test, as you asked on 2026-11-14); `qwen_mtp_patch` on the
`PYTHONPATH` of its server; the DeepSeek context cap D1; and Phase D, both servers at the same time with the memory
watch, because the margin is at or below the 4 GiB of G6. A looser G1 limit and the smaller margin are your decisions.

### 7.5 The facts and the side-by-side test (your request of 2026-11-14)

Your question: how much visible loss, and how many more hallucinations, does K 208 bring? §7.2 measured the NLL, the
decisions, the tool names and the tool-call suite, but no test of facts. This window adds two tests, on the full model
(h256) and on K 208 (h208), both with the draft:

1. **Facts.** 200 questions of SimpleQA (OpenAI's public set of 4,326 short questions with one known answer;
   `qwen-data/simpleqa/`, not in git): 120 of the topic "Science and technology", the topic nearest to your work, and
   80 of the others, in a fixed order that mixes both. The model answers with no tools, at the sampling of the fast
   route, with a fixed seed per question and at most 2,048 tokens, for at most 20 minutes per model. The questions are
   rare facts: the knowledge that a prune removes first. After the window, the live DeepSeek grades each answer
   against the known answer: correct, incorrect (a hallucination) or not attempted. The grader runs with no thinking:
   it graded 6 of 6 known cases right that way. The report counts only the questions that both models answered.
2. **Side-by-side.** Each model writes a new sample (seed + 1) of the 60 recorded turns of §4. After the window,
   DeepSeek (with thinking, about 45 seconds a call) judges 30 of them blind: the end of the context, the recorded
   turn of the full model, and the new turn, in both orders. Two comparisons: the full model's new turn against its
   recorded turn (the noise of the judge and of the sampling), and K 208's new turn against the same recorded turn.

How to read it: K 208 hallucinates more if its incorrect answers rise above the noise of 200 questions (about ±5);
its loss is visible if the judge prefers the recorded turn over K 208's turn clearly more often than over the full
model's own new turn.

### 7.6 Result of the facts and side-by-side test (2026-11-14, 09:48 to 10:59)

The live server stopped at 09:48 and was healthy again at 10:58:54 (reply "OK.", gateway active): 70 minutes of
downtime. The judging ran after the window on the live DeepSeek, from 11:00 to 12:25. Files (not in git):
`results/qwen-fit-h{256,208}-{facts,graded,pairs,judged}.jsonl`, `logs/qwen-facts.log`.

**Facts.** Each model had 20 minutes: the full model answered 87 questions, K 208 answered 80 (its answers are a
little longer: 1,485 tokens on average against 1,385, and 26 of 80 hit the limit of 2,048 tokens against 22 of 87).
On the 80 common questions:

| | Full model | K 208 |
|---|---|---|
| Correct | 15 | 10 |
| Incorrect (a hallucination) | 43 | 46 |
| Not attempted (an "I don't know", or no answer in 2,048 tokens) | 22 | 24 |
| Incorrect of the attempted answers | 74% | 82% |
| Science and technology: correct, incorrect | 9, 21 | 4, 25 |

Per question: the full model knows 7 answers that K 208 gets wrong or leaves open, and K 208 knows 2 that the full
model misses (a sign test gives p = 0.18: the sample is too small to prove a loss, and its direction agrees with the
G1 loss). Five of the 7 lost answers are science and technology. Some lost answers are confident inventions: "Dolly
the sheep was cloned into three identical copies" (she gave birth to triplets), 450 hp for an engine of 550 hp. The
full model itself is wrong on 3 of 4 of these rare facts, so neither model should answer such questions without a
search.

**Side-by-side.** The judge compared a new turn with the recorded turn of the full model, on 30 held-out turns in
both orders:

| New turn | New better | Recorded better | Tie | Unjudged | Score (wins + ties / 2) |
|---|---|---|---|---|---|
| Full model, a new sample (the noise) | 28 | 17 | 14 | 1 | 0.58 |
| K 208 | 24 | 24 | 11 | 1 | 0.49 |

Per turn (both orders added), K 208 scores better than the full model's new sample on 8 turns, worse on 10, and the
same on 12 (p = 0.81). The judge prefers the turn that it reads first (17 against 11 wins for the new turn), which is
why each pair runs in both orders. The score falls by 0.09, but the per-turn test cannot tell it from the noise.

**What it means for your use:**

- On your own work (coding, tool calls, research with its context), the judge does not see a difference. This agrees
  with §7.2: the decisions, the tool names and the tool-call suite did not move.
- On rare facts with no search, K 208 knows less: about one third fewer correct answers in this sample, with a few new
  confident errors. The extra hallucinations are small (+3 of 80) and inside the noise.
- The risk is in questions that need stored knowledge of rare things: niche libraries, versions, numbers, names. With
  web search or the content in the context, the risk falls.

The K 208 folder was deleted again after the window.

### 7.7 K 208 with NVFP4 draft experts (your choice of 2026-11-14)

You chose K 208, then the vLLM path to more margin: the 256 experts of the MTP draft in NVFP4 instead of BF16.
Speculation stays exact, so the outputs of the target do not change; only the acceptance of the draft can fall.

**What vLLM needs** (read in vLLM 0.33):

- vLLM builds the draft with the quant config of the *target* (`VllmConfig.quant_config` comes from the target model
  config), so the ignore list and the groups of `qwen-k208/config.json` decide how the draft layers are built. The
  draft folder gives the weights and, with `qwen_mtp_patch`, the expert count.
- The checkpoint ignores the draft (`re:^mtp.*`). The patch of `qwen_draft_quant.py --patch-config` changes it to
  `re:^mtp\.(?!layers\.0\.mlp\.experts\.).*` (the rest of the draft stays BF16), keeps the expert group of the
  target off the draft (`re:^(?!mtp\.).*mlp\.experts...`), and adds a group for the draft experts with NVFP4 weights
  and no input quantization (NVFP4A16). vLLM takes the first matching target, and the tests use its own functions.
- Weight-only NVFP4 MoE needs a backend with a weight-only mode. The target's `flashinfer_cutlass` has none, so the
  draft runs its MoE on Marlin (`"moe_backend": "marlin"` in the speculative config). Weight-only mode needs no
  activation scales, so no calibration of the draft.
- The fused gate and up of an expert need one global scale; the checkpoint does the same (12,992 for both in layer 39,
  expert 0).

**The build** (`qwen_draft_quant.py`, about 20 seconds): the draft folder `qwen-draft-nvfp4` links the four shards with
no draft tensor and rewrites the fifth (3.6 GB to 2.4 GB). The draft experts take **0.42 GiB instead of 1.50 GiB**:
1.08 GiB less. The scales and the packing come from compressed-tensors. Checks:

- The format: one expert of the checkpoint, unpacked and packed again with its global scale, gives 100% of its scales
  and the same values (only the sign of some zero codes differs).
- The error of the 256 new experts against BF16: 9.5% relative (mean and worst), the usual error of 4 bits with a
  scale per 16 weights.
- vLLM 0.33 builds the engine config of the point on the CPU: the draft with 256 experts on Marlin, the target with
  208, the new group with no input quantization.

**The window** (point d1, about 10 minutes): K 208 with the NVFP4 draft, the smoke test, one prompt of 120,000
tokens and the bot decode test. The baseline is k0 of §7.4 (the BF16 draft): 26.51 GiB at the peak, 97.65 tok/s, 2.24
tokens accepted per draft step.

**The decision rule, written before the window:** the NVFP4 draft is the choice if d1 boots, passes the smoke test,
uses at least 0.75 GiB less than k0 at the peak, and decodes within 5% of k0 (at least 92.8 tok/s). Then K 208 uses
about 25.5 GiB: a margin of about 3.3 to 4.0 GiB beside DeepSeek, or 3.7 to 4.4 GiB with the context cap.

### 7.8 Result of d1, and the next window (2026-11-14, 13:48 to 13:57)

The live server stopped at 13:48 and was healthy again at 13:57:07: 9 minutes of downtime. d1 booted on the first
try: the target's MoE on FLASHINFER_CUTLASS, the draft with 256 experts (`qwen_mtp_patch`) and its MoE on MARLIN
(weight-only FP4).

| | k0 (BF16 draft, §7.4) | d1 (NVFP4 draft) |
|---|---|---|
| Model loading | 18.09 GiB | **17.01 GiB** (−1.08) |
| GPU process (`nvidia-smi`) | 22,639 MiB | 22,461 MiB (−0.17 GiB) |
| Use at the peak | 26.51 GiB | 26.44 GiB (−0.07) |
| Overhead X at the peak / after the boot | 6.59 / 6.21 GiB | 7.60 / 7.23 GiB |
| Prompt of 119,745 tokens | 45.5 s | 45.4 s |
| Decode with the draft | 97.65 tok/s | 97.92 tok/s |
| Tokens accepted per draft step | 2.24 | 2.19 |

**By the rule of §7.7, d1 fails:** its peak is only 0.07 GiB below k0, and the rule asks 0.75 GiB. The speed and the
acceptance hold.

**The cause, found in the second window: a cold compile.** The patched quant config changes the graph, so d1
compiled it cold (torch.compile took 17.6 s, against 0.4 s from the cache in k0), and the compile held about 1 GiB
in the engine process. My first guess (blocks of the Marlin repack that the allocator keeps) was wrong.

### 7.9 Result of d2 and d3, and the decision (2026-11-14, 14:02 to 14:15)

The live server stopped at 14:02 and was healthy again at 14:15:26: 13 minutes of downtime. You started this window.
Both points loaded the compiled graph from the cache (0.4 s).

- d2 was meant to free the allocator cache after the load (`QWEN_EMPTY_CACHE=1`), but the hook patched the V1
  model runner, and vLLM 0.33 runs the V2 runner (`vllm.v1.worker.gpu.model_runner`): the hook did not run. So d2 is
  d1 again with a warm cache. The hook is removed (it is not needed), and d2 is now d1 with no change.
- d3 is d1 with the allocator's expandable segments.

| | k0 (BF16 draft) | d1 (cold compile) | **d2 (warm)** | d3 (warm, expandable) |
|---|---|---|---|---|
| Model loading | 18.09 GiB | 17.01 GiB | 17.01 GiB | 16.96 GiB |
| GPU process | 22,639 MiB | 22,461 MiB | **21,655 MiB** | 21,647 MiB |
| Use at the peak | 26.51 GiB | 26.44 GiB | **25.42 GiB** | 25.51 GiB |
| Overhead X at the peak | 6.59 GiB | 7.60 GiB | 6.58 GiB | 6.72 GiB |
| Prompt of 119,745 tokens | 45.5 s | 45.4 s | 45.1 s | 45.2 s |
| Decode with the draft | 97.65 tok/s | 97.92 tok/s | 101.78 tok/s | 99.31 tok/s |
| Tokens accepted per draft step | 2.24 | 2.19 | 2.31 | 2.23 |

**The decision: the NVFP4 draft passes the rule of §7.7.** With a warm cache, K 208 with the NVFP4 draft uses
25.42 GiB at the peak, 1.09 GiB less than with the BF16 draft, at the same speed and acceptance (2.19 to 2.31 against
2.24). Expandable segments add nothing. The setup: `qwen-k208` with the patched quant config, the draft folder
`qwen-draft-nvfp4`, `"moe_backend": "marlin"` in the speculative config, and `qwen_mtp_patch` on the `PYTHONPATH`.

| Margin beside DeepSeek (28.8 to 29.5 GiB available) | Without the context cap | With the cap (+0.35 GiB) |
|---|---|---|
| K 208 with the NVFP4 draft (25.42 GiB) | 3.4 to 4.1 GiB | 3.7 to 4.4 GiB |

**A caution for Phase D.** The first boot after any change of the config or of vLLM compiles cold and needs about
1 GiB more for a minute. Boot a new setup once alone (DeepSeek stopped) to fill the compile cache, then beside
DeepSeek. The cache now holds this setup.

## 8. Phase D: both servers at the same time

### 8.1 The procedure (your go of 2026-11-14: first without the context cap, then with it)

- `pair_watch.sh` (tmux `pair-watch`): every second, the available memory to `results/pair-mem.log`. Below 2 GiB it
  kills the Qwen server, and only it: the process group of the pid that `pair_run.sh` records, and only when that
  command line is the vLLM 0.33 venv with `--port 8889`. DeepSeek also shows on the host as a vLLM process (port
  8888), so a plain name match would be unsafe.
- `pair_run.sh TAG=nocap|cap`: checks (DeepSeek healthy, no Qwen, the watch on, at least 28.6 GB available, not in the
  cron window), boots K 208 with the NVFP4 draft beside DeepSeek (`--gpu-memory-utilization 0.005`: with a pinned KV
  cache, vLLM uses it only for the free-memory check at the boot), then the tests of §7.3 of the fit plan, each with
  the lowest available memory: both idle; Qwen with 120,000 tokens while DeepSeek decodes; DeepSeek with 128,000
  tokens while Qwen decodes; the bot decode test on Qwen alone, on DeepSeek alone, and on both at once. Qwen then
  stays up for the 24-hour gate.
- `cap_switch.sh`: stops Qwen, waits for an idle bot, and recreates the DeepSeek container from
  `compose-md149t5-c160.yml` (the live file with `MAX_MODEL_LEN` 160000 and a KV pin of 7,431,600,000 bytes; the
  project name and the container name stay). A pin that is too small gets one more boot with the need plus 4%; a
  failed boot rolls back to `compose-md149t5.yml`. After a healthy boot, Hermes gets `context_length` 160000 and the
  compression threshold 0.8 (backup of `config.yaml`), and the gateway restarts.
- The order: watch, `pair_run.sh TAG=nocap`, `cap_switch.sh`, `pair_run.sh TAG=cap`.

**The gate.** The lowest available memory of every test stays at or above 2 GiB (the watch never fires), and both
servers stay healthy. Then 24 hours of normal use with the cron jobs and no out-of-memory event.

### 8.2 Result without the context cap (2026-11-14, 14:25 to 14:48)

DeepSeek did not stop. 28,799 MB was available beside it before the boot. Qwen booted in 3 minutes from the warm
compile cache (`gpu_memory_utilization` is not part of vLLM's compile key). The lowest available memory during the
boot was 8,505 MB.

| Test | Lowest available | Result |
|---|---|---|
| Both idle, 60 s | 9,852 MB | |
| Qwen: 119,745 tokens, while DeepSeek decodes | 8,220 MB | Qwen 52.7 s (45 s alone); DeepSeek 133 tokens in 18.6 s |
| DeepSeek: 127,837 tokens, while Qwen decodes | 8,477 MB | DeepSeek 245 s; Qwen 44.9 tok/s |
| Bot decode test, Qwen alone | 8,327 MB | 99.1 tok/s |
| Bot decode test, DeepSeek alone | 7,517 MB | 36.9 tok/s |
| Bot decode test, both at once | 7,459 MB | Qwen 46.8 tok/s, DeepSeek 19.3 tok/s |

**The gate holds.** The lowest available memory of the run is 7,459 MB (7.3 GiB), far above the floor of 2 GiB;
the watch never fired; both servers stayed healthy. Most of that memory is reclaimable file cache (with both idle:
1.6 GB free, 8.9 GB cached). The use of Qwen beside DeepSeek (about 18.7 GB of available memory) is lower than the
25.4 GiB that the windows measured with DeepSeek stopped: the windows count the fall of `MemAvailable` from a machine
with about 118 GB available, and that fall is not the same as the cost beside a full server.

**Speed.** One model alone keeps its speed. When both decode at the same time, they share the memory bandwidth:
Qwen falls from 99 to 47 tok/s and DeepSeek from 37 to 19 tok/s. A router that sends one request at a time to one
model keeps full speed.

**A fault of the script.** The last test waited with a bare `wait`, which also waits for the Qwen server (a child of
the script), so the script did not write its summary; the results above come from its files. The fix waits for the
client by its pid. Qwen stayed up.

### 8.3 The context cap is not applied; the 24-hour gate (your decision of 2026-11-14)

You asked for the cap when the projected margin was 3.4 to 4.1 GiB. Without it, the run left 7.3 GiB at its lowest,
and the cap adds only about 0.35 GiB, at the cost of the 256k context of DeepSeek, a change to Hermes and a restart.
You chose to skip it. `cap_switch.sh` stays in the repository, unused; its dry run wrote
`compose-md149t5-c160.yml` into the server folder, and that file is removed. DeepSeek and Hermes are unchanged.

**The 24-hour gate** started at 14:28 on 2026-11-14: Qwen K 208 with the NVFP4 draft stays up on port 8889 (pid in
`results/pair-qwen.pid`), with the watch in tmux `pair-watch` (`results/pair-mem.log`, `logs/pair-watch.log`). It
ends at 14:28 on 2026-11-15, before the cron jobs of 2026-11-16. The gate passes if `logs/pair-watch.log` has no
WATCH line, both servers are healthy, and `journalctl -k` shows no out-of-memory event. Nothing routes requests to
Qwen yet; the gate covers the normal traffic of DeepSeek beside an idle Qwen.

If DeepSeek restarts by itself (`restart: on-failure:1`) while Qwen runs, its boot needs the memory that Qwen holds:
the watch then stops Qwen below 2 GiB. To stop Qwen by hand: `kill -TERM -- -$(cat results/pair-qwen.pid)`, then
`tmux kill-session -t pair-watch`.

As `PLAN_SECOND_MODEL_FIT.md` §7: DeepSeek first, then the pruned Qwen with a very small utilization value or after
a drop of the page cache, a memory watch that stops Qwen below 2 GiB, the load tests on both, and the 24-hour gate
of the router plan.

### 8.4 The image test (your request of 2026-11-14)

You want to paste screenshots and diagrams into the chat. Every window so far ran Qwen with `--language-model-only`.
The vision tower is intact in the K 208 folder (27 blocks, 0.83 GiB in BF16; the prune copied it unchanged), but the
rank of the prune counted text tokens only, so the experts that image tokens use were never counted.

**The test** (`qwen_vision.py`): 12 images rendered with headless Chromium, 3 questions each with a known answer that
is visible in the image: 4 code screenshots (Python, shell, a diff, a traceback), 3 charts, 3 UI screens (a form, a
table, a dialog) and 2 diagrams. A model answers each question about its image, at the sampling of the fast route
with a fixed seed. The live DeepSeek (text only) then grades each answer against the known answer.

**The window** (it stops the Qwen of the 24-hour gate, which starts again after it):

| Point | Model | Tests |
|---|---|---|
| i208 | K 208 with the NVFP4 draft, vision on (one image per prompt) | Smoke, the 36 image questions. The first boot compiles the new graph |
| i256 | The full model with its draft, vision on | Smoke, the 36 image questions: the reference |
| m208 | i208 again, with a warm compile cache | Smoke, 6 image questions, a prompt of 120,000 tokens, the bot decode test: the memory with vision |

**The decision rule, written before the window:**

- K 208 keeps its vision if it answers at most 3 questions fewer than the full model (the noise of 36 sampled
  questions), and no kind of image loses more than 2.
- The memory of m208 is reported against d2 (25.42 GiB with no vision). The margin beside DeepSeek falls by the cost
  of the tower; it is below the 4 GiB of G6 either way, so whether vision stays on is your decision.
- After the window, Qwen starts again beside DeepSeek (`pair_run.sh`, with `VISION=1` if the test passes), and the
  24-hour gate starts again. In its first 90 minutes (14:28 to 15:57) the gate had no WATCH line, `oom_kill` 0, and at
  least 8.6 GiB of swap free.

### 8.5 Result of the image test (2026-11-14, 15:59 to 16:18)

The live server stopped at 15:59 and was healthy again at 16:17:59: 18 minutes of downtime. All three points booted
on the first try. Files (not in git): `results/qwen-fit-{i208,i256,m208}*`, `logs/qwen-vision.log`.

**The images: no loss shows.**

| | Full model (i256) | K 208 (i208) |
|---|---|---|
| Correct, graded by DeepSeek | 36 of 36 | 36 of 36 |
| Code / charts / UI / diagrams | 12 / 9 / 9 / 6 | 12 / 9 / 9 / 6 |
| Time for the 36 questions | 75 s | 78 s |

One image costs 354 to 519 prompt tokens. m208 repeated 6 questions of i208 and gave the same text 6 of 6 times.
**The limit of this test:** both models reach the ceiling, so it rules out a gross loss and cannot show a small one.
`qwen_vision.py` now also builds a hard set (5 images, 15 questions: a full-HD screenshot of dense code, a table of 16
rows, a chart of two series with no value labels, a state diagram, a dense dark panel). It has not run yet.

**The memory: vision costs 2.66 GiB, three times the weights of the tower.**

| | d2: no vision (§7.9) | m208: vision, warm cache |
|---|---|---|
| Model loading | 17.01 GiB | 17.87 GiB (+0.86) |
| GPU process | 21,655 MiB | 22,667 MiB (+0.99 GiB) |
| API server, resident | 1.97 GB | 3.01 GB (+1.04 GB) |
| Engine core, resident | 3.24 GB | 3.52 GB (+0.28 GB) |
| **Use at the peak** | **25.42 GiB** | **28.08 GiB (+2.66)** |
| Prompt of 119,745 tokens | 45.1 s | 45.4 s |
| Decode with the draft | 101.78 tok/s | 101.57 tok/s |

- The text speed does not change.
- The margin beside DeepSeek falls from 3.4 to 4.1 GiB to **0.7 to 1.4 GiB**, before any swap.
- A hidden cost: vLLM keeps a cache of processed images, `--mm-processor-cache-gb`, 4 GiB by default, in the API
  server and again in the engine core. With the default, the two processes can grow by up to 8 GiB as images arrive.
  Any use beside DeepSeek needs `--mm-processor-cache-gb 0`.

**By the rule of §8.4:** the quality passes (0 questions fewer). The memory is your decision. Not yet tested: the trims
(`--mm-processor-cache-gb 0`, a limit on the image size with `--mm-processor-kwargs` or the width and height options of
`--limit-mm-per-prompt`), and the hard set.

**The state after the window:** at 16:18, Qwen K 208 started again beside DeepSeek in its text-only setup
(`TAG=gate2 TESTS=idle pair_run.sh`), and the 24-hour gate of §8.3 starts again.

### 8.6 Vision with two cuts, and the hard set (your go of 2026-11-14)

Two cuts for the memory of §8.5:

1. `--mm-processor-cache-gb 0`: no cache of processed images. It stops the growth of up to 8 GiB; the memory at the
   boot may not change, because the cache starts empty.
2. `--mm-processor-kwargs {"max_pixels":3686400}`: at most 2560 x 1440 pixels per image (about 3,600 tokens). The
   checkpoint allows 16.7 megapixels (16,384 tokens), and vLLM sizes its encoder cache and its warm-up run for that.
   A larger screenshot is scaled down.

| Point | Model | Tests |
|---|---|---|
| t208 | K 208 with the NVFP4 draft, vision with the two cuts | Smoke, the hard set (15 questions) |
| t256 | The full model, vision with the two cuts | Smoke, the hard set: the reference |
| u208 | t208 again, warm compile cache | Smoke, the hard set, a prompt of 120,000 tokens, the bot decode test: the memory |

**The rule, written before the window:** the cuts help if the peak of u208 is at least 0.5 GiB below m208 (28.08 GiB).
K 208 passes the hard set if it answers at most 2 questions fewer than the full model. The gate of §8.3 stops for the
window and starts again after it.

## 9. Risks

| Risk | Effect | Guard |
|---|---|---|
| The prune costs more than on DeepSeek: the model is smaller, and each expert is 1.69 MiB | No point passes | Three counts in one window; the coverage report before the builds |
| The calibration misses a kind of work that you do | A skill fails that the gates do not test | The held-out sessions and the held-out repository; your own use as the last gate; keep the full folder for a rollback |
| The transcripts hold secrets | A leak through a file in git or a request | Local files only, covered by `.gitignore`; the model runs on this machine |
| The assistant turns of the transcripts are another model's text | The rank fits that style more than Qwen's | Qwen's own generated tokens in the capture; G1 uses Qwen's own turns |
| The draft loses acceptance on a pruned target | Less speed | G5; the rebuild of the draft experts |
| vLLM refuses a folder with 176 experts | A boot fails | The first build is tested on a boot in the gate window; the config change is one number |
| The pair has no margin for a peak | Out of memory, a hard reset | Phase D with the memory watch, as in the fit plan |

## 10. Timeline

| Phase | Work | Downtime |
|---|---|---|
| A | About 2 days | None |
| B | One window | About 85 min |
| C | The builds: about 30 min. One window | About 80 min |
| D | One evening, then 24 hours of normal use | None, if the margin holds |

Phase B decides early: the coverage report shows whether the traffic leaves a cold tail of experts, before any build.
