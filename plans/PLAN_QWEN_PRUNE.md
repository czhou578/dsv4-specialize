# Plan: prune the experts of Qwen3.6-35B-A3B for the fast route

**Status:** 2026-11-14: the overhead of vLLM frees nothing (§7.4): the four levers leave the full model with its
draft at 29.6 GiB, about −0.1 GiB of margin. K 208 with its draft uses 26.5 GiB at the peak with a 120k prompt: a
margin of 2.3 to 3.0 GiB, or 3.4 to 4.1 GiB with the DeepSeek context cap (D1), at 98 tok/s. Its cost stays the G1
loss of +4.15% (§7.2). It is the choice to report; the G1 limit and Phase D are your decision. Nothing is deployed.
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

The DeepSeek context cap (`PLAN_SECOND_MODEL_FIT.md` §3.1, D1) adds 1.1 GiB to each margin. So K 192 with D1 also
reaches 4 GiB.

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
| K 208 | 3.16 GiB | About 26.7 GiB | About 2.3 GiB (3.4 with the DeepSeek context cap) |
| K 192 | 4.22 GiB | About 25.7 GiB | About 3.3 GiB (4.4 with the cap) |
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
| K 192 passes and K 176 fails | Phase D with K 192 (margin about 3.3 GiB; 4.4 with the DeepSeek context cap, D1) |
| Only K 208 passes | Phase D with K 208 and D1 (margin about 3.4 GiB). Your decision on the smaller margin |
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
about 29 GiB available): the margin at K 224 is about 1.2 GiB, or about 2.3 GiB with the DeepSeek context cap (D1).
G6 asks for 4 GiB. No K passes both G1 and G6, so nothing is accepted and nothing is deployed. On your request,
the folders `qwen-k176`, `qwen-k192` and `qwen-k208` were deleted on 2026-11-14. The rank file
(`results/qwen-rank.json`) stays, and `qwen_prune.py --draft keep` rebuilds a folder in about 75 seconds.

**The choices for the fast route, none taken:**

1. **Nemotron-3.5-Lightning-30B-A3B** (`PLAN_SECOND_MODEL_FIT.md` §5.5): 74 tok/s in 25.5 GiB with no draft at 256k
   (a margin of about 3.5 GiB), 127 tok/s in 28.8 GiB with its DSpark draft. Its quality on your work is not tested.
2. **Qwen3.6-35B-A3B in full with a shorter context or no draft:** 58.8 tok/s with no draft at 128k (about 27.4 GiB),
   or the draft at a context short enough for the margin.
3. **K 208 with a looser G1:** it passes every other gate and saves 2.8 GiB, but its margin is about 2.3 GiB (3.4
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

| Setup | Peak | Margin (28.8 to 29.5 GiB available) | With the DeepSeek context cap D1 (+1.1 GiB) |
|---|---|---|---|
| Full model with its draft (o0) | 29.64 GiB | −0.8 to −0.1 GiB | +0.3 to +1.0 GiB |
| K 208 with its draft (k0) | 26.51 GiB | 2.3 to 3.0 GiB | 3.4 to 4.1 GiB |

K 208 with the draft and D1 reaches the 4 GiB of G6 only at the top of the range, so read it as a margin of about
3.5 GiB. Its quality cost is the
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

## 8. Phase D: both servers at the same time

As `PLAN_SECOND_MODEL_FIT.md` §7: DeepSeek first, then the pruned Qwen with a very small utilization value or after
a drop of the page cache, a memory watch that stops Qwen below 2 GiB, the load tests on both, and the 24-hour gate
of the router plan.

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
