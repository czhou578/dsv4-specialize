# Plans

These plans moved here from the bot's repo, `czhou578/stock-research-agent`, on 2026-11-08. Their text did not
change, so the section numbers that the scripts cite are still correct.

| Plan | What it covers |
|---|---|
| `PLAN_OVERNIGHT_MASK_GRID.md` | The mask-grid night of 2026-11-03, and the safety rules of the run scripts (§2) |
| `PLAN_OVERNIGHT_TOPK.md` | The top-k night of 2026-11-04 |
| `PLAN_FINAL_REVIEW.md` | The final review of bot K 160 at top-k 5: the LLM judge and the blind review |
| `PLAN_PHYSICAL_PRUNE.md` | Phase 3: the prune tool, the test boot, the bot-rank draft and the deploy |
| `PLAN_BETTER_RANK.md` | The weighted rank and the capture patch. K 144 fails with every rank, so K 160 is the floor |
| `PLAN_COLD_EXPERTS.md` | Do the cold experts matter: can the experts that the bot's traffic rarely uses go, with no cost on the replay (2026-11-11: done; two windows: both cold sets (486 experts, 4.29 GiB) have no measurable cost; next, the loader-patch plan) |
| `PLAN_LOADER_PATCH.md` | A different expert count in each layer: the runtime patch that a prune of the cold experts needs (one plan per count, one shared scratch), the prune of MD, and its test window (2026-11-11: the work before the window is done; the MD prune equals the mask in the kernel check; the window waits) |
| `PLAN_FEWER_LAYERS.md` | Fewer layers (depth pruning): rank the layers by Block Influence, confirm them with the replay gate, heal the cut with a linear map (2026-11-10: plan written; nothing built) |
| `PLAN_FEWER_EXPERTS.md` | Fewer experts per token: a top-k 6 runtime with a drop rule and an anchored gain (2026-11-09 to 2026-11-10: no faster point; the anchored top-5 is a little better on the replay only; no deploy) |
| `../serve-tune/PLAN_SERVE_TUNE.md` | A faster decode and less memory from the serving settings: draft depth, KV pin, prefill chunk, FP8 head. It moved to `serve-tune/` with its tools, results and write-up (2026-11-10: done; the KV pin of 7.25 GiB is live; the 128-expert draft passes but is skipped for now; the 24-hour metrics and the KV tier study are blocked) |

The plans give paths as they were in the bot's repo. Read them as follows:

- `PLAN_MODEL_SPECIALIZE.md` (the method), `PLAN_E2E_RUNNER.md` (the full runs), `PLAN_MODEL_ROUTER.md`,
  `MODEL_SWAP.md` and `MODEL_PRUNE.md` stay in the bot's repo.
- `model_eval/` stays in the bot's repo, except three files: `model_eval/capture.py`, `rank_data.py` and
  `rank_report.py` are now `capture.py`, `rank_data.py` and `rank_report.py` in this repo.
- `model_eval/results/` is now `results/` in this repo. The bot's folder keeps the full-run rows of the stock model
  (`e2e/`) and a copy of the stock baselines (`baseline-k216*`, `speed-*-k216*`).
- `model_eval/data/` (the recorded bot sessions, with `rank/`) stays in the bot's eval worktree and out of git.
- The step scripts of `~/.hermes-e2e/` and their logs are now in `history/hermes-e2e/`. The logs of the run scripts
  are in `logs/`.
