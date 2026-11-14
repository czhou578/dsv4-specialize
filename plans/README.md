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
| `PLAN_LOADER_PATCH.md` | A different expert count in each layer: the runtime patch that a prune of the cold experts needs (one plan per count, one shared scratch), the prune of MD, and its test window (2026-11-11: done; the MD prune loads at 69.79 GiB, frees 4.18 GiB, and passes the replay and the bot speed; next, the deploy chain) |
| `PLAN_MD_REVIEW.md` | The deploy review of the MD prune: the full runs against the live model, the LLM judge and your blind review (2026-11-12: done; the MD prune passes, and a deploy is possible) |
| `PLAN_MD_DEPLOY.md` | The deploy of the MD prune on the live server: the new compose file, the script with its rollback, and the time per model call before and after (2026-11-12: done; the MD prune is live since 18:00, after 2 min 53 s of downtime; it takes 9.9% more time per model call, from longer answers, and you kept it; reading C is due 2026-11-13) |
| `PLAN_COLD_CUT2.md` | The second cold window: S15 and S20, two supersets of MD (5.56 and 8.01 GiB), two replays per point on the mean against the R0 of each half (2026-11-13: done; S15 +0.82% and S20 +1.49% against R0, and both fail the baseline gate; MD is the limit of the cold cut; next, the fewer-layers plan) |
| `PLAN_FEWER_LAYERS.md` | Fewer layers (depth pruning): rank the layers by Block Influence, confirm them with the replay gate, heal the cut with a linear map (2026-11-13: done; zero layers: every whole layer costs +2.15% or more on the screen, the best FFN half (F24) costs +1.06% on the replay and +0.42% with its heal map, and the model with no skip already sits on the gate) |
| `PLAN_SECOND_MODEL_FIT.md` | The fit of a second model (Qwen 27B NVFP4, 256k context) beside DeepSeek: measure its overhead alone, then choose the DeepSeek savings with no output change (context cap, narrower draft, prefill budget), then run both (2026-11-13: plan written; it does not fit today: Qwen needs 28.4 GiB plus its overhead, and about 29 GiB is available) |
| `PLAN_FEWER_EXPERTS.md` | Fewer experts per token: a top-k 6 runtime with a drop rule and an anchored gain (2026-11-09 to 2026-11-10: no faster point; the anchored top-5 is a little better on the replay only; no deploy) |
| `PLAN_HEAD_VOCAB.md` | A smaller vocabulary for the head: a token set for the draft only (no output change), or for the target and the draft (2026-11-11: plan written; nothing built) |
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
