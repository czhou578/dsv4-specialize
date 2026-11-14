# Plan: prune the experts of Qwen3.6-35B-A3B for the fast route

**Status:** 2026-11-13: plan written; the sizes of §2 are measured; nothing built ·
**Author:** colin-spark · **Date:** 2026-11-13
**Related:** `PLAN_SECOND_MODEL_FIT.md` (§5.5 the measured fit and speed of the candidates), `PLAN_PHYSICAL_PRUNE.md`
and `PRUNE.md` (the method on DeepSeek), `PLAN_BETTER_RANK.md` (the ranks), `PLAN_MODEL_ROUTER.md` in the bot's repo
**Scope:** a data builder, a capture client, a prune tool and two window scripts in this repo, and three test
windows. The model runs with the vLLM 0.33 venv, alone on port 8889. No change to the DeepSeek server.

| Step | Status |
|---|---|
| §5 Phase A: the data, the tools and the tests (no downtime, about 2 days) | 🔨 2026-11-13: the data builder (§5.1), the capture option and the prune tool (§5.2) are done; next the two window scripts |
| §6 Phase B: the capture window (about 85 min downtime) | ⬜ |
| §7 Phase C: the builds and the gate window (about 80 min downtime) | ⬜ |
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
| 6 | Two points with a memory-free draft: `ngram` and `suffix`. A boot and the bot decode test for each | 2 × 9 |
| 7 | The restore of the live server | 3 |

Step 6 tells whether a draft without weights gives a useful part of the gain of the MTP draft. If one does, the
prune can be smaller.

**Gate.** The capture has at least 1.5M tokens with no failed segment, the rank file has 256 experts for each of the
40 layers (and for the draft layer), and the reference files of G1, G3 and G5 exist.

After the window, with no downtime: the report of the routing (the share of the tokens that the top 160, 176 and
192 experts of each layer cover), and the builds. A coverage that is far lower than on DeepSeek at the same ratio is
an early sign that the prune costs more.

## 7. Phase C: the builds and the gate window (about 80 min downtime)

Builds (no downtime): K 192, K 176 and K 160, each about 12 to 17 GB on disk, with `--check`.

| Order | Point | Tests | Minutes |
|---|---|---|---|
| 1 | K 256 with the draft, 128k | G4, G5, G6: the reference of the window | 15 |
| 2 | K 176 with the draft | G1 to G6 | 20 |
| 3 | K 192 with the draft | G1 to G6 | 20 |
| 4 | K 160 with the draft | G1 to G6, only if K 176 passes G1 and G2 | 20 |
| 5 | The restore of the live server | | 3 |

How to read the result:

| Result | Next step |
|---|---|
| K 176 passes all gates | Phase D with K 176. K 160 is a reserve for a larger margin |
| Only K 192 passes | Phase D with K 192 and the DeepSeek context cap (D1) |
| No point passes G1 or G3 | Stop the prune. The choices are Qwen without its draft at 128k (about 27.4 GiB), or Nemotron |
| A point passes the quality gates, but the speed with the draft falls | Rebuild the draft experts by the rank of the pruned target, as the bot-rank draft of DeepSeek. Else your decision |

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
