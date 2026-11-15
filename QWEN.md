# The Qwen3.6-35B-A3B prune

Since 2026-11-14, a pruned copy of Qwen3.6-35B-A3B (`unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`) runs on port 8889 beside
the live DeepSeek V4 Flash. It keeps 208 of 256 routed experts per layer. It runs with its MTP draft and its vision
tower. This page records why, how the tests chose that point, and what is open. The details and every measurement are
in [plans/PLAN_QWEN_PRUNE.md](plans/PLAN_QWEN_PRUNE.md). The section numbers below (§) refer to that plan.

## 1. Why

The owner wants two models in memory at once: DeepSeek for the deep work of the stock-research bot, and a fast model
for coding, web search and tool calls. Qwen3.6-35B-A3B decodes at 99 tok/s with its MTP draft, against 35 tok/s for
DeepSeek. But with the draft it uses 32.3 GiB at a 256k context, and about 29 GiB is free beside the pruned DeepSeek.
The routed experts hold 77% of its weights, so a prune of the experts is the main lever. The goal was a context of
128k with a margin of at least 4 GiB.

## 2. The method

The method is the method of the DeepSeek prune ([PRUNE.md](PRUNE.md)): record the experts that the traffic uses,
rank them, remove the cold ones, and gate the cut against the full model. Three parts are simpler for Qwen:

- Each expert is a set of plain tensors, so [qwen_prune.py](qwen_prune.py) copies the kept ones with no new
  quantization.
- vLLM 0.33 returns the routed experts with a flag, so the capture needs no server patch.
- One expert count for all layers needs no loader patch.

**The data** ([qwen_data.py](qwen_data.py), §3): 99 segments, 2.44M tokens, from the owner's Claude Code transcripts,
the bot's recorded sessions and the owner's repositories. The data stays local and out of git, because the
transcripts can hold secrets. The held-out set is whole sessions and one repository that the calibration never sees.

**The gates** (§4), against the full model in the same window:

| Gate | Measure | Limit |
|---|---|---|
| G1 | The NLL of 60 held-out turns that the full model wrote | At most +3% |
| G2 | The tool names and the decisions of the same turns | Tool names at least 98%; decisions down by at most 3 |
| G3 | A tool-calling suite of 85 tasks | At most 3 tasks fewer |
| G5 | The decode speed with the draft | Report |
| G6 | The peak memory at 128k with the draft | A margin of at least 4 GiB beside DeepSeek |

## 3. The results of the gates (§7.1, §7.2)

| Point | G1 NLL | G2, G3 | Target experts removed |
|---|---|---|---|
| K 208 | +4.2% | no change | 3.17 GiB |
| K 192 | +8.0% | no change | 4.22 GiB |
| K 176 | +11.1% | no change | 5.27 GiB |

Every pruned point fails G1. The full model repeats within 0.1% between the two windows, so the cost is real and not
noise. G1 holds only above about K 224, and that leaves a margin of about 1 GiB. At the peak with the draft at 128k,
K 208 uses 26.51 GiB, against 29.9 GiB for the full model. At the same share of the routing,
Qwen loses about twice as much as DeepSeek. Its experts are small (1.69 MiB each), and each token uses 8 of 256.

## 4. The decision: K 208, with a cost on G1

G1 measures how far a pruned model moves from the full model. It does not measure whether the owner sees a loss. So
two more tests ran on K 208 (§7.5, §7.6, [qwen_facts.py](qwen_facts.py)):

- **80 rare facts of SimpleQA, with no search.** K 208 answers 10 right and 46 wrong. The full model answers 15 right
  and 43 wrong. Both differences are inside the noise of 80 questions, but the right answers fall by a third. Rare
  facts are where a prune loses first.
- **A blind judge on 30 of the owner's held-out turns.** DeepSeek as the judge finds K 208 better on 8 turns and worse
  on 10 (p = 0.81): no loss that shows.

The owner chose K 208 on 2026-11-14. Two more changes then gave more margin with no change to the output:

- **NVFP4 draft experts** ([qwen_draft_quant.py](qwen_draft_quant.py), §7.7 to §7.9). The 256 experts of the MTP draft
  go from BF16 to NVFP4. The target model checks every draft token, so its output does not change. The peak falls by
  1.09 GiB to 25.42 GiB, with the same speed (101.8 tok/s) and acceptance (2.31 tokens per step).
- **A patch for vLLM 0.33** ([qwen_mtp_patch/](qwen_mtp_patch/sitecustomize.py), §7.1). Without it, vLLM builds the
  draft with the expert count of the pruned target, and the boot fails. The draft keeps its 256 experts.

## 5. Both servers at once (§8.1 to §8.3)

[pair_run.sh](pair_run.sh) boots Qwen beside the running DeepSeek, with no stop of DeepSeek, and runs load tests.
[pair_watch.sh](pair_watch.sh) checks the memory every second. Below 2 GiB available, it stops Qwen, and never
DeepSeek.

| Test (text only, 2026-11-14) | Lowest available | Result |
|---|---|---|
| Qwen: a prompt of 119,745 tokens, while DeepSeek decodes | 8,220 MB | Qwen 52.7 s (45 s alone) |
| DeepSeek: a prompt of 127,837 tokens, while Qwen decodes | 8,477 MB | DeepSeek 245 s |
| Bot decode test, each model alone | 7,517 MB | Qwen 99.1 tok/s, DeepSeek 36.9 tok/s |
| Bot decode test, both at once | 7,459 MB | Qwen 46.8 tok/s, DeepSeek 19.3 tok/s |

- **Swap carries part of the margin.** One hour after the boot, at least 4.6 GiB of the two servers' idle host pages
  was in swap. Without the 16 GiB swap file, the lowest available memory would be about 0 to 2.7 GiB. The setup works
  because of the swap, and the watch is the guard against a hard reset.
- **The two models share the memory bandwidth.** Each model alone keeps its speed. When both decode at once, each runs
  at about half speed. A router that sends one request at a time keeps the full speed.
- **The DeepSeek context cap is not used.** [cap_switch.sh](cap_switch.sh) cuts the DeepSeek context to 160k for
  about 0.35 GiB. That is too small a gain for the loss of the 256k context.

## 6. Vision (§8.4 to §8.9)

The rank counted text tokens only, so the experts that image tokens use were never counted. The vision tower itself
is unchanged by the prune. [qwen_vision.py](qwen_vision.py) renders screenshots of code, charts, UI screens and
diagrams with known answers. DeepSeek grades each answer.

- **Quality:** K 208 answers 51 of 51 questions, as the full model does: 36 normal questions and 15 hard ones (dense
  full-HD code, a table of 16 rows, a chart with no value labels). Both models reach the ceiling, so the test rules
  out a gross loss and cannot show a small one.
- **Memory:** vision costs 2.36 GiB at the peak, with two settings: `--mm-processor-cache-gb 0` and at most
  2560 × 1440 pixels per image. With the default cache of 4 GiB in each of two processes, the server can grow by up to
  8 GiB as images arrive.
- **The live test with vision**, beside DeepSeek: the lowest available memory was 5,827 MB, the lowest free swap was
  6.6 GiB, and the hard set stayed 15 of 15. Vision stays on since 2026-11-14 17:33.

## 7. Open items

- **The 24-hour gate** of normal use with vision ends on 2026-11-15 at 17:33. It passes if the watch never fires, no
  process is killed for memory, and the free swap stays above about 4 GiB.
- **The margin is thin.** A durable fix is on the DeepSeek side: a larger cold cut or a narrower draft (§8.7). None of
  it is started.
- **Rare facts.** The SimpleQA result suggests that K 208 knows fewer rare facts. A question about a rare fact needs web
  search.
