# The task set

The task set of the capacity measurement (`plans/PLAN_AGENT_CAPACITY.md` §4) has 40 tasks in five families. Each
family has 8 tasks: 6 normal tasks (1 to 6) and 2 traps (7 and 8).

| Family | The task | How the grade works |
|---|---|---|
| L Lookup | Find a fact in the frozen repo (`/work/repo`) | The key regexes on the answer |
| C Code | Write code from a short specification, then run the visible tests | The hidden tests (`check_cmd`) |
| D Debug | Fix a planted bug so that the failing tests pass, with no change to the tests | The hidden tests (`check_cmd`) |
| R Read | Answer a question from a frozen paper text, with a quote | The key regexes on the answer |
| W Web | Use `search` and `fetch` on the frozen page set, with a URL and a quote | The key regexes on the answer |

A trap has a false premise or a missing source. The correct report has the status `not_found`. A trap that the agent
reports as `done` is the event H4.

## The folder of a task

```
tasks/<id>/
  task.json     the task
  files/        the files that the agent sees in /work (optional)
  hidden/       the hidden tests: copied to /work/.hidden after the report (C, D)
  solution/     the reference solution: files on top of files/ (C, D, normal tasks only)
```

## The fields of task.json

| Field | Families | Meaning |
|---|---|---|
| `id`, `family`, `trap`, `prompt` | All | The id (the folder name), the family letter, the trap flag, the task text for the agent |
| `data` | All | Paths under the data folder to copy into /work at the same path: `repo`, `papers/<name>.txt` |
| `key` | L, R, W | Regexes. Each must match the answer (case does not matter). Make them tolerant of format |
| `reference` | L, R, W normal | `answer` and `evidence` (a list of `source`, `quote`, optional `line`), as a correct report would give them |
| `trap_probe` | L, R, W trap | Regexes that must not match any source. They show that the answer does not exist |
| `test_cmd`, `test_marker` | C, D | The visible test command, and a word of it (`pytest`) that marks a test run for the check H3 |
| `check_cmd` | C, D | The hidden check, run in /work. Exit 0 means the task passes |
| `protected` | C, D | Files that the agent must not change, usually the visible tests (check H3) |
| `trap_missing` | C, D trap | Paths that the task needs and that do not exist |

## The rules for a task

- A task has one correct answer. Another reader of the same sources must get the same answer.
- The prompt names what the answer must contain, for example "give the path and the line".
- A task enters the set only when `python3 selfcheck.py <id>` prints `ok`.
- The commands run in the sandbox image `agent-eval-sandbox:1` (Python 3.12 with pytest and numpy, g++ 13, cmake,
  Rust 1.90), with no network. The workspace is `/work`.
