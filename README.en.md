# codegen-robustness-bench · LLM Code-Generation Robustness Benchmark

[中文](README.md) | English

Using "the same functional specification under code obfuscation of different strength" to test whether an LLM writes C functions from understanding or from memorization.

This document describes the project: its purpose, workflow, usage, credibility measures and limitations, all stated plainly. Section 9 gives experiment results (2026-10-28) for three DeepSeek models on two kinds of task, spec generation and bug repair; for interpretation see [`docs/实验结果.md`](docs/实验结果.md). Integration-check results for the pipeline itself (prepare through verify, plus mutate for the bug-repair task) on the 9 real corpora are in [`docs/实验记录.md`](docs/实验记录.md).
Interface contracts and design details are in [`docs/设计.md`](docs/设计.md) (v0.4; its Section 10 is the "implementation deviation record") and, for the bug-repair task, in the appendix
[`docs/设计-缺陷修复任务.md`](docs/设计-缺陷修复任务.md).

**Note:** the documents under `docs/` are written in Chinese only; the links to them in this README point to the Chinese originals.

## 1. Purpose

Existing code-generation benchmarks (HumanEval, MBPP, and function-level benchmarks taken directly from GitHub) are prone to training-data leakage: the function under test and the project it lives in
are likely to appear in the model's training corpus, so the model generates code from memory rather than from understanding, and the score does not reflect real engineering ability.

What this platform does:

1. Extract target functions and their dependency context from several open-source C projects;
2. Apply three kinds of verifiable code obfuscation to the **whole project**: symbol renaming (L1), structural rewriting (L2) and semantic rewriting (L3); every obfuscated copy must still pass all of the project's own tests;
3. Have the model regenerate the target function under "the same functional specification, different obfuscation levels";
4. Put the generated code back into the matching obfuscated copy, then compile it and run the project's own tests in a Docker sandbox;
5. Compute how the pass rate changes across levels, attribute the errors of failed samples, and measure the similarity between the generated code and the original function (a trace of memorization).

Questions it answers:

1. How much the compile rate and the functional-test pass rate of the same model differ between no obfuscation (L0) and each obfuscation level (L1–L3);
2. Which kinds of failure occur (syntax, missing dependency, type, logic, crash, timeout, constraint violation), and how they shift as obfuscation gets stronger;
3. How different models compare (any OpenAI-compatible endpoint can be plugged in);
4. Whether the similarity between the L0 output and the original function drops noticeably at L1.

What it does not do: it does not evaluate multi-file feature additions or languages other than C, and does not fine-tune models.

**Two parallel tasks.** What is described above is the **spec-generation task** (generate): the model's input is only the function signature, the dependency declarations and the functional specification, with no function body.
The L2 / L3 transformations happen inside the function body, which the model never sees, so at those two levels the model input is word-for-word identical to L1, and the levels matter only for the project copy used on the evaluation side (Section 8, item 7).
To make all three kinds of obfuscation act on the code the model actually reads, there is also a **bug-repair task** (repair, Section 3.4): a defect that the project's tests can catch is injected into the target function in each level's copy,
the model reads that (obfuscated) function and returns a corrected one. The two tasks are reported side by side: the generation task answers "does writing a function depend on memorization?", and the repair task answers "does understanding and fixing a function depend on its surface form?".

## 2. Pipeline overview

```
prepare    fetch the corpus into corpus/src/<name>; confirm in the sandbox that build and tests work
coverage   compile with --coverage in the sandbox and run the tests; gcov gives each function's execution count
targets    select target functions: not in a test file, not main, statements ≥ 3, execution count > 0, stub-kill passed
obfuscate  generate four copies L0 / L1 / L2 / L3 of the whole project (test files included), record the mapping and statistics, then verify
verify     oracle gate: run a complete evaluate on each target using the obfuscated original function text; it must pass, otherwise the target is voided
mutate     (bug-repair task only) inject one defect into the target function of each level's copy; a problem is set only if the sandbox confirms "it compiles, and the project tests fail reliably"
check-mutants  (bug-repair task only) replay the mutants chosen by mutate in the sandbox as a re-check; each must make the project tests fail reliably
describe   use an LLM to generate a functional specification "containing no project identifiers" for each target (generated once and shared by all four levels; the repair task does not need it)
generate   set problems per (target, level, model, sample), call the model, cache the replies; samples whose call failed are recorded separately and never silently dropped (--task selects the task type)
evaluate   post-process → direct verdict on constraint violations → put back into the obfuscated copy → sandbox compile → run project tests → attribution → similarity (--task selects the task type)
report     aggregate metric tables over model × level × corpus, the attribution distribution and similarity; output md / csv / figure (both tasks side by side)
```

Every step is idempotent: if the output already exists and the input hash has not changed, the step is skipped; all randomness is determined by `--seed`.
When problems are set (the step before `generate` / `evaluate`), the upstream outputs are cross-checked once more: targets whose copies, oracle gate or functional specification no longer match the current state are voided and counted,
so stale data is never used just because some intermediate stage was skipped.

## 3. Quick start

### 3.1 Environment

* Python 3.10 (the `requires-python` in `pyproject.toml`; older versions get a clear message when `bench` is imported).
  Dependencies: `pip install clang==14.0 pytest pyflakes` (only `clang==14.0` is needed at runtime; `pyproject.toml` declares the Python version, the dependencies and the pytest configuration,
  and the project is not packaged for installation: run it directly from the project root). The libclang library is expected at
  `/usr/lib/llvm-14/lib/libclang.so.1` by default; if it is elsewhere, point to it with the environment variable `LIBCLANG_PATH`;
* Docker, with the image `gcc:13` available locally (`docker pull gcc:13`). Builds and tests of the corpus repositories, as well as the C code generated by models,
  are executed only inside the sandbox (`--network none --cap-drop ALL --pids-limit --memory --cpus`, non-root user); the host never runs make directly;
* No other third-party dependencies are needed: HTTP uses the standard library's urllib, and figures are written directly as SVG with the standard library;
* The commands below are all run from the project root (`python3 -m bench` has to be able to import the `bench` package).

### 3.2 Run the offline self-test first

No API key and no network are needed. It runs the whole pipeline on `tests/fixtures/mini`, a small purpose-written project, with MockLLM (both the generation task and the bug-repair task are run;
measured at 5 to 7 minutes on a 16-core / 15 GB development machine, with the elapsed time printed at the end as "用时 N s" ("elapsed N s"); this varies with the machine and the concurrency and is not a guaranteed figure):

```bash
python3 -m bench e2e --backend mock
```

It runs L0–L3 × the three MockLLM modes oracle / broken / wrong, and asserts that: oracle gets 100% pass; broken gets 100% compile_syntax;
for wrong (a zero-value stub that returns a zero value or has an empty function body) not a single sample passes, and each sample's attribution equals exactly the kill mode recorded by stub-kill
(mostly test_fail; for a few targets the stub makes later test code read uninitialized state and crash, which is attributed to runtime_crash, a fact stub-kill has already recorded);
verify and the oracle gate pass for every copy; and rerunning every stage must skip all of them. Output goes to `tmp/e2e/`; `results/` is not touched.

The bug-repair task runs mutate → generate → evaluate → check-mutants in the same e2e run. MockLLM has three more modes, used only for the repair task, and the run asserts that:
oracle gets 100% pass and restored (restored verbatim); for unchanged (returns the mutant as is) not a single sample passes, the stage is test, `unchanged` is true, and the share whose attribution agrees with the mutant's `kill_mode`
is at least 95%; reformat (the original function with different formatting and added comments) gets 100% pass with `restored` true and `unchanged` false, which checks that the normalized comparison ignores only formatting and comments;
for touched (the mutant plus one harmless statement) not a single sample passes and both `unchanged` and `restored` are false, which checks that "changed something elsewhere, defect still there" is not counted as changing nothing;
broken gets 100% compile_syntax; for wrong not a single sample passes and its attribution equals the stub-kill record; every (target, level) whose mutant was killed had a problem set, with a problem-setting rate of at least 90%;
and in the replay re-check (`check-mutants --oracle`) all mutants are still killed and none passes, while all comment-stripped original functions pass.

### 3.3 Run on the real corpora

```bash
export DEEPSEEK_API_KEY=...            # read from the environment only; never written to any file, log or result

python3 -m bench prepare   --corpus cjson sds
python3 -m bench coverage  --corpus cjson sds
python3 -m bench targets   --corpus cjson sds
python3 -m bench obfuscate --corpus cjson sds
python3 -m bench verify    --corpus cjson sds
python3 -m bench describe  --corpus cjson sds --model deepseek-v4-pro
python3 -m bench generate  --corpus cjson sds --model deepseek-flash --levels L0 L1 L2 L3
python3 -m bench evaluate  --corpus cjson sds --model deepseek-flash --tag pilot
python3 -m bench report    --tag pilot
```

* `--corpus` accepts several values; if omitted, all 9 corpora under `corpus/specs/` are used. `--seed` defaults to 20261027; `--jobs` is the concurrency of sandbox and LLM calls.
* Before the real run, do `python3 -m bench generate --dry-run ...` first: it only estimates how many requests still have to be sent (minus replies already on hand), how many distinct requests there are after deduplication,
  and roughly how many prompt tokens that is; it does not construct a model and writes no files. The L1–L3 prompts are word-for-word identical (see Section 8) and share one cache key, so with all four levels enabled the real number of requests is about half the number of tasks.
* `describe` and `generate` need an API key only when there is actually work to do: if their outputs are already up to date, a purely offline resume or re-check does not fail for lack of a key.
* Model names: the DeepSeek official Change Log announces that `deepseek-chat` / `deepseek-reasoner`, the names used in the design document, are deprecated from 2026-07-24;
  whether they can still be called is determined by the current official documentation. The defaults and presets now use the `PRESETS` in `bench/llm.py` (`deepseek-flash`, `deepseek-flash-thinking`,
  `deepseek-v4-pro`, `deepseek-v4-pro-thinking`); the old names are kept only as legacy presets and give a warning when used.
  `mock-oracle` / `mock-broken` / `mock-wrong` / `mock-empty` are fake models for the offline self-test (`mock-unchanged`, `mock-reformat` and `mock-touched` are used only by the bug-repair task:
  they return the mutant as is, return the original function reformatted with comments added, and return the mutant plus one harmless statement, respectively); only `--unsafe-debug` allows them to be written into the real workspace.
  Any other OpenAI-compatible endpoint can be added as one more entry in `PRESETS`.
* Multiple samples: `--samples N` requires `--temperature > 0`; sample i uses i as a seed in the cache key, so every sample is requested and cached separately.
  The DeepSeek presets do not send the seed to the API (the documentation has no such parameter), so the differences between samples come from temperature sampling.
  `--no-cache` does not read the LLM cache (it still writes to it): `--force` only invalidates existing records, and a cache hit still reuses the old reply unchanged, so to resample you need to pass both.
* Corpus source: if the environment variable `CORPUS_MIRROR` is set and holds a matching local clone, `prepare` copies it directly without going online; if it is not set, `prepare` clones from the spec's url + commit
  (it never guesses any local directory). When git goes online and a proxy is configured in the environment, it probes first: a reachable proxy is used as usual, and the proxy is bypassed temporarily only if none of them can be reached
  (`CORPUS_KEEP_PROXY=1` forces it to be kept, `CORPUS_NO_PROXY=1` forces it to be cleared); the client that calls the model uses the proxy from the environment by default, and `BENCH_LLM_NO_PROXY=1` turns that off.
* An evaluation interrupted midway (Ctrl-C, process killed) can simply be resumed: outputs already on disk are reused according to their input hash, and a half-written line left in a result file is recognized and counted.

### 3.4 Bug-repair task

```bash
python3 -m bench mutate        --corpus cjson sds                  # set problems: inject defects into the target function of each level's copy (verify output must exist first)
python3 -m bench check-mutants --corpus cjson sds --oracle         # replay re-check: every mutant is still killed, every comment-stripped original function passes
python3 -m bench generate  --corpus cjson sds --model deepseek-flash --task repair
python3 -m bench evaluate  --corpus cjson sds --model deepseek-flash --task repair --tag pilot
python3 -m bench report    --tag pilot                             # default is --task all: generation and repair tables side by side
```

* **The problem.** The model gets the dependency declarations and the **complete function with one defect injected** in that level's copy (comments stripped); there is no functional specification, no test output, and no hint about where the defect is.
  It has to return the corrected complete function (signature unchanged, no new `#include`). Judging is exactly the same as for the generation task (put back → syntax check → build → project tests → attribution), and pass = all project tests pass.
  The prompt template is in `bench/prompt.py` (`REPAIR_PROMPT_VERSION` / `REPAIR_PROMPT_DIGEST`, kept separate from the generation task's `PROMPT_VERSION` / `PROMPT_DIGEST`).
* **Mutation operators.** Eight: ROR (relational operator replacement), NEG (condition negation), LOR (logical operator replacement), OBO (off-by-one), DGS (delete guard), AOR (arithmetic operator replacement),
  RVR (return value replacement), DST (delete statement). Sites are found on the target function body via the AST and rewritten by byte offset; after rewriting, the whole file must re-parse with libclang with zero new errors.
  Sites inside macro arguments, sites generated by macros and sites on lines containing preprocessor directives are never touched.
* **Selection and kill check.** For each (target, level), all sites are shuffled into a fixed order (determined by `--seed` and the target; **the level is not part of the seed**), and the mutants are taken in turn as candidates
  and put through a complete evaluate: only a mutant that compiles and fails in the **test stage** (`test_fail` / `runtime_crash` / `runtime_timeout`, recorded as `kill_mode`) counts as killed and gets a problem set;
  a mutant for which all tests still pass is an equivalent mutant, and one that fails to compile does not count as killed. At most `--max-tries` (default 12) sites are tried; if none is killed after that, the (target, level) is given up and counted.
  Only (target, level) pairs that pass the oracle gate are processed.
* **Re-check.** A candidate judged killed is evaluated once more, and a problem is set only if it fails the second time too (recorded as `kill_recheck`); a candidate that passes the second time (the outcome of a crash / timeout / OOM varies with load,
  and so does an uninitialized variable left behind by a deleted assignment) is counted in `unstable_count`, no problem is set, and the next site is tried. The re-check runs at the same concurrency, so it filters out only some of the sporadic cases;
  to confirm once more on a quiet machine, use `check-mutants` (`--jobs 1`).
* **Across levels: the same defect.** L1 changes only identifiers, so its site table corresponds to the L0 table one to one, by index. L0 therefore chooses first, and L1–L3 first try **the site at their own level that corresponds to the one L0 chose**,
  then prefer sites of the same operator (`same_operator_as_L0`). The correspondence at L1 is exact; L2 / L3 rewrite the function body, so they align by token sequence (all identifiers are treated as one and the same token,
  and a site matches only if its own token and one token on each side all match). A site whose tokens were rewritten (`x == 0` turned into `!x`, `a < b` into `b > a`, the condition after an `if` / `else` swap)
  cannot be matched, and the choice falls back to reusing the operator only. The selected mutant is recorded with `same_site_as_L0`; the ±1 of OBO is also decided by the index of the L0 site, so the same defect is changed in the same direction at every level.
  **The "same operator at all four levels" subset is always identical to "problem set at all four levels", so it is not a control** (reusing the operator guarantees this); the **comparable subset** in the report consists of targets
  that have the same site at all four levels and no pruned entries in the prompt context (the respective counts are computed from the data files by `python3 -m bench.experiment_record` and written in `docs/实验记录.md`).
* **Comment stripping.** Symbol renaming changes only identifiers, so in the function-body comments at L1 and above the project's original names are still there, and putting them into the prompt unchanged would leak names that are supposed to be hidden;
  the mutants of all four levels are therefore comment-stripped before the kill check. For a function that has comments, the comment-stripped original function is first put back into the copy and the project tests are run once more;
  if they fail, the (target, level) is given up (no search, and it is not used as a reference for L1–L3). Original names inside string literals cannot be removed (changing them would change behavior): a problem at L1 or above
  whose target function's own original name appears in a literal is voided when problems are set (`name_in_literal`), and the rest can only be tallied and reported as they are.
* **Prompt context.** The dependency declarations were originally taken from the original function, so calls, macros and globals that the defect deleted would still be in there, which amounts to writing a clue to the answer into the prompt.
  Problems are therefore re-pruned against the function with the defect injected when they are set (`prompt.prune_context`; only prototypes, macros and globals are pruned, type definitions are not),
  and the pruned names are recorded in the task record's `context_orphans`. Repairing such a problem may need names that are not in the prompt, so its difficulty comes from an information gap rather than from obfuscation;
  it is left out of the comparable subset, and its pass rate is reported separately.
* **Unreliable problems.** If an `unchanged` candidate (the model changed nothing) nevertheless passes the project tests, the mutant is not reliably killed by the tests: `evaluate` names it and fails the stage,
  and `report` drops the whole problem (all models, all samples) and names it at the top.
* **Result records.** Each record carries these extra fields: `task` (`repair`), `mutation` (operator, site index, original text, what it was changed into), `kill_mode` (how the mutant died), `stub_kill_mode` (how the target was killed by the zero-value stub),
  `unchanged` (the candidate equals the mutant after normalization, i.e. the model changed nothing), `restored` (the candidate equals that level's original function after normalization, i.e. restored verbatim), `same_operator_as_L0`, `same_site_as_L0`,
  `context_orphans`. Normalization ignores only formatting and comments; identifiers are compared as they are. In the repair task `restored` and similarity are **not memorization metrics** (the prompt already contains the original function with a difference in only one place,
  and any correct fix of a single-point defect is word for word equal to the original function); look at how the pass rate changes between levels.
* **Replay re-check `check-mutants`.** The mutants chosen by mutate are taken as candidates and evaluated again (equivalent to evaluating `mock-unchanged`, but without going through generate / evaluate and without writing evaluation results):
  each must fail in the test stage, and whether the kill mode changed is also recorded (either outcome counts as killed; it only affects the stratification label); with `--oracle`, the comment-stripped original function is additionally put back into the copy, and each must pass.
  The result is written to `results/mutants-<corpus>.check.json`, and the command returns non-zero as soon as one mutant's replay passes the tests. Before problems are set and before evaluation, the mutate output is checked for staleness, and a mismatch is an immediate error.
* **`--task`.** `generate` / `evaluate` / `report` all take `--task {generate,repair,all}`, default `all`: the generation task is processed as usual, and the repair task is processed along with it only if mutate already has output
  (otherwise a notice is printed and it is skipped, so that someone who only wants the generation task is not blocked); `--task repair` fails with an error telling you to run mutate first if mutate has not been run.
  In `report`, if one task type has no result file, only the other is produced. For the generation task, `--tag` and the model name must not start with `repair-` (they would collide with files such as the repair task's `results-repair-<tag>.jsonl`).
* **Limitations.** See the last few items of Section 8: the mutants are artificially injected single-point defects; no test output is given; for targets killed by a crash / timeout the tests may not check the returned content; the prompt context has been pruned;
  similarity and the restored rate in the repair task are not memorization metrics; how conspicuous a NEG defect is varies with the level; and project vocabulary inside string literals cannot be removed.

## 4. Command reference

`python3 -m bench <cmd>`. All commands accept `--corpus`, `--seed`, `--jobs`, `--force` (ignore existing outputs), `--workspace DIR`
(put results, corpus copies, obfuscated copies and temporary directories all into another directory), `--image` / `--cpus` / `--memory` (sandbox parameters), `--unsafe-debug` and `-v` / `--verbose`.

Values of `--task`: `generate` (the spec-generation task), `repair` (the bug-repair task; run `mutate` first), `all` (the default; handles both, and the repair task is skipped with a notice if mutate has no output yet).
Values of `--tag` and `--model` must not start with `repair-` (they would collide with the repair task's data files, and are rejected as a usage error).

Exit codes: 0 success; 1 a stage failed (including evaluate / report finding missing samples or bad lines) or an e2e assertion does not hold; 2 usage error or missing prerequisite
(missing upstream output, no API key, docker unavailable, etc.); 130 interrupted by Ctrl-C (outputs already on disk can be resumed).
Unexpected runtime failures (sandbox unavailable, libclang parse failure, etc.) return 1, and the full stack trace is appended to `results/bench-error.log` in the workspace.

`--unsafe-debug` is the only switch that "relaxes the credibility measures" (`targets` may skip stub-kill, allow mock descriptions, and let through targets that never went through stub-kill);
it is for debugging only, its outputs must not be used for a formal report, and the start of the report says so.

| Command | What it does | Main options | Outputs |
|---|---|---|---|
| `prepare` | Fetch the sources and confirm in the sandbox that build and tests work | `--src DIR` local sources, `--no-build-check` | `corpus/src/<name>`, `results/prepare-<name>.json` |
| `coverage` | Build with `--coverage`; gcov gives per-function execution counts | | `results/coverage-<name>.json` |
| `targets` | Select targets: static filtering, execution counts, sampling by seed, stub-kill | `--max-per-corpus`, `--min-statements`, `--min-lines`, `--max-lines`, `--no-exec-filter`, `--no-stub-kill` (must be combined with `--unsafe-debug`), `--lazy` | `results/targets-<name>.jsonl`, `.summary.json` |
| `obfuscate` | Generate the obfuscated copies and verify | `--level` / `--levels L0 L1 L2 L3`, `--strict-stringify auto/on/off` | `results/obf-<name>-<level>.json`, `tmp/obf/<name>/<level>/` |
| `verify` | oracle gate: run a complete evaluate per target with the obfuscated original function | `--level` / `--levels` | Written back into the `oracle_gate` of `obf-*.json` |
| `mutate` | Problem setting for the bug-repair task: inject defects into the target function of each level's copy and run the kill check; a mutant judged killed is re-checked once more (verify must have run first) | `--level` / `--levels`, `--max-tries` (default 12) | `results/mutants-<name>-<level>.jsonl`, `results/mutants-<name>.summary.json` |
| `check-mutants` | Replay re-check of the mutants chosen by mutate: each must still fail in the test stage; also records whether the kill mode changed | `--level` / `--levels`, `--oracle` (additionally puts the comment-stripped original function back into the copy; each must pass) | `results/mutants-<name>.check.json` |
| `describe` | Generate an identifier-free functional specification for each target | `--model` | `results/descriptions.jsonl` |
| `generate` | Set problems and call the model | `--model` (several allowed), `--levels` / `--level`, `--samples`, `--temperature`, `--max-tokens`, `--dry-run`, `--no-cache`, `--task` | `results/tasks.jsonl`, `results/generations-<model>.jsonl`, `results/generation-failures-<model>.jsonl` (for the repair task see Section 6) |
| `evaluate` | Evaluate replies: post-processing, replacement, compilation, tests, attribution, similarity | `--model` (several allowed), `--levels` / `--level`, `--tag`, `--allow-missing`, `--task` | `results/results-<tag>.jsonl`, `results/completeness-<tag>.json` (for the repair task see Section 6) |
| `report` | Aggregate the metrics (both task types side by side) | `--tag`, `--task`, `--format md csv svg`, `--allow-stale`, `--allow-missing` | `results/summary-<tag>.md` / `.csv`, `results/fig-<tag>.svg` |
| `e2e` | End-to-end self-test of the whole pipeline on the mini fixture | `--backend mock`, `--keep`, `--levels` | `tmp/e2e/` |

### Environment variables

| Variable | Purpose |
|---|---|
| `BENCH_ROOT` | Project root (by default derived from the location of `bench/store.py`) |
| `BENCH_RESULTS_DIR` / `BENCH_CORPUS_SRC` / `BENCH_OBF_DIR` / `BENCH_WORK_DIR` | Results directory, corpus directory, obfuscated-copies directory and temporary work directory (defaults: `results/`, `corpus/src/`, `tmp/obf/` and `tmp/work/` respectively) |
| `BENCH_SPEC_DIR` | Spec directory (default `corpus/specs/`) |
| `LIBCLANG_PATH` | The libclang library file |
| `BENCH_CLANG_INCLUDE` | When libclang cannot find the compiler's built-in headers (stddef.h etc.), explicitly specifies the built-in header directories (separated by `os.pathsep`) |
| `CORPUS_MIRROR` | Directory holding local clone mirrors; `prepare` copies from it first; if not set, no mirror is used |
| `CORPUS_GIT_TIMEOUT` | Timeout in seconds for git commands that go online (default 180) |
| `CORPUS_KEEP_PROXY` / `CORPUS_NO_PROXY` | When git goes online, force keeping / force clearing the proxy from the environment (if neither is set, it probes: the proxy is bypassed only if it cannot be reached) |
| `BENCH_LLM_NO_PROXY` | Do not use the proxy from the environment when calling the model API |

Model API keys are read only from environment variables (`DEEPSEEK_API_KEY`; see the `api_key_env` of each preset in `PRESETS`) and are never written to any file, log, cache or result.
Variables used only by tests and CI are listed in Section 11.

## 5. Obfuscation levels

Each level includes all the transformations of the levels below it. All three kinds of transformation are determined by `--seed`, so the same seed gives the same result (the rename table of symbol renaming also depends on the host's system headers; see item 9 of Section 8);
every copy must pass the project's own tests (verify).

| Level | Transformation | Notes |
|---|---|---|
| L0 | Verbatim copy | Control group |
| L1 | Symbol renaming: the spellings of functions, variables, parameters, fields, types, enum constants, labels and macros declared inside the project are all renamed to names that reveal nothing about their meaning: `f_<n>` `t_<n>` `v_<n>` `M_<n>` `m_<n>` | Token-level replacement: only identifier tokens change, and string literals and comments are left alone. Spellings that appear in system headers, `main`, keywords, third-party frameworks (`spec.vendored`), macros specified with `-D` in the build command (`spec.keep_names`) and the like are never renamed. When a stringifying macro is encountered, non-strict verification is tried first and strict is the fallback if it does not pass; the record states which one was used, and in non-strict mode how many spellings were still renamed after being stringified |
| L2 | Structural rewriting: `for` rewritten as `while`, `if/else` branches swapped, compound assignments expanded, ternary `return` rewritten as `if/else`, multi-declarations split | About 50% of the available sites in each function body; statements generated by macros and positions inside macro arguments are left alone; after the rewrite the file must re-parse with zero errors |
| L3 | Semantic rewriting: `a < b` rewritten as `b > a`, De Morgan, `!x` and `x == 0` swapped, `a + b` reordered by commutativity, an opaque statement inserted at the start of the function body | Applied only to simple operands without side effects and only to integer and arithmetic types; likewise the file must re-parse with zero errors |

## 6. Data files

JSON / JSONL, all carrying `schema_version`. Stage outputs are rewritten atomically as a whole; those that can be resumed from a checkpoint (generations, results) are appended line by line.
Every output carries `inputs` (the content hash of each input) and `input_hash`, on which the idempotency check and the validation in `report` both rely; a digest of the evaluation-related code
(`bench/provenance.py` hashes the content of the source files per stage), the sandbox image ID and the resource limits are included as well, so if the image is replaced or the evaluation code is changed, old results are judged stale.

| File | Contents |
|---|---|
| `results/prepare-<corpus>.json` | Source-tree hash, results of building and testing in the sandbox |
| `results/coverage-<corpus>.json` | Execution count of each function |
| `results/targets-<corpus>.jsonl` | Selected targets: all FunctionTarget fields + `exec_count` + `stub_killed` + `stub_kill_mode` (test_fail / runtime_crash / runtime_timeout) |
| `results/targets-<corpus>.summary.json` | Number extracted, number of candidates, the count and target_id for each exclusion reason (including the number of stub-kill survivors and each target's stub-kill verdict) |
| `results/obf-<corpus>-<level>.json` | `rename_map`, `stats`, verify results, oracle gate results (which targets were checked and which passed), the copy's source-tree hash, a digest of the host's libclang environment |
| `results/descriptions.jsonl` | `target_id`, functional specification, generating model, number of leak retries, `failed`, allow-list hits, phrase warnings, a digest of the identifier-leak checker |
| `results/tasks.jsonl` | All Task fields (the complete inputs of the prompt) and the message hash |
| `results/generations-<model>.jsonl` | `task_id`, `sample_idx`, reply text, token usage, whether the cache was hit, latency, temperature / max_tokens / seed, model fingerprint, input hash |
| `results/generation-failures-<model>.jsonl` | Generation calls that still failed after retries were used up (task_id, sample_idx, reason); if a call later succeeds, it is deleted from here |
| `results/results-<tag>.jsonl` | EvalResult + `category` + `detail` + `similarity` + reply conformance issues + input hash |
| `results/completeness-<tag>.json` | For each (model, level, corpus), how many samples there should be, how many were evaluated and how many are missing (generation failed / not generated / reply stale) |
| `results/mutants-<corpus>-<level>.jsonl` | The mutants chosen by mutate, one line per (target, level) that had a problem set: `target_id`, `level`, `operator`, `site_index`, `original_text`, `mutated_text`, `mutant_code` (the complete mutated function, comments stripped), `kill_mode`, `kill_recheck` (the attribution at re-check), `tried_sites`, `equivalent_count`, `compile_fail_count`, `parse_fail_count`, `unstable_count`, `same_operator_as_L0`, `same_site_as_L0`, `anchor` (where the L0 site ended up at this level), `context_orphans` (names in the prompt context made superfluous by the defect), input hash |
| `results/mutants-<corpus>.summary.json` | Summary of mutate: the input hash of each level, number of problems set, number given up and the reasons, distribution of operators and kill modes, the share of `same_operator_as_L0` / `same_site_as_L0`, accounting (re-checks included), sandbox time taken by each kind of mutant, elapsed time, and the number of targets with problems set at all four levels / the same operator at all four levels / the same site at all four levels |
| `results/mutants-<corpus>.check.json` | Replay results of `check-mutants`: how many were replayed, how many are still killed, whether the kill mode changed, whether any replay passed (a failure), and whether the comment-stripped original functions passed |
| `results/tasks-repair.jsonl`, `results/generations-repair-<model>.jsonl`, `results/generation-failures-repair-<model>.jsonl` | The tasks, replies and failed calls of the bug-repair task: same structure as the corresponding generation-task files, with the extra `task` and `mutation` fields |
| `results/results-repair-<tag>.jsonl`, `results/completeness-repair-<tag>.json` | Evaluation results and completeness records of the bug-repair task, with the extra `task`, `mutation`, `kill_mode`, `unchanged`, `restored`, `same_operator_as_L0`, `same_site_as_L0`, `context_orphans` |
| `results/usage.jsonl` | Token usage of each real API call (cache hits are not recorded) |
| `results/cache/` | LLM reply cache, keyed by sha256(model, endpoint, messages, temperature, max_tokens, seed); ignored by `.gitignore` |
| `results/summary-<tag>.md` / `.csv`, `fig-<tag>.svg` | Output of `report` |
| `results/bench-error.log` | Full stack traces of unexpected failures (appended) |

## 7. Credibility measures

| Measure | How | Self-test |
|---|---|---|
| oracle gate | Each obfuscated copy is first verified as a whole; then a complete evaluate is run on every target with the obfuscated original function text, and it must pass. A copy or target that does not pass is voided and counted. The result records which targets were checked, and they are re-checked one by one when problems are set; a gate that was run against a different target set / copy is always reported as an error | `tests/test_e2e_mock.py`, `tests/test_integrity.py` |
| stub-kill | After the target function is replaced by a stub that "returns a zero value / has an empty function body", the project tests must fail, otherwise the target is not selected. The number of survivors is written into the report; the kill mode (test_fail / runtime_crash / runtime_timeout) is recorded per target, and the report stratifies pass rates by kill mode | `tests/test_targets.py` |
| Hard constraints | The prompt forbids calling exit / abort / system and the like, forbids network access and inline assembly, and forbids defining macros. `postprocess` scans the candidate code with libclang, and these as well as constructs that would bypass the pass criterion (constructor attributes, `#undef` / `#pragma`) are recorded straight away as `policy_violation` and never enter the sandbox. Besides the exit code, the pass criterion of each corpus also requires a completion marker in the test output (a summary line or a fixed line, with the number of occurrences pinned down where necessary), so an early `exit(0)` does not pass | `tests/test_postprocess.py`, `tests/test_corpus.py`, `tests/test_integrity.py` |
| MockLLM three-mode e2e | oracle 100% pass, broken 100% compile_syntax, wrong does not pass and its attribution agrees with stub-kill. The bug-repair task additionally has an unchanged mode (returns the mutant as is; none may pass and `unchanged` must be true), a reformat mode (the original function reformatted with comments added; it must pass and be `restored`) and a touched mode (the mutant plus one harmless statement; it must not pass, and `unchanged` and `restored` must both be false); the oracle mode must also be `restored`; plus the `check-mutants` replay | `tests/test_e2e_mock.py` |
| Mutant kill check | Each mutant of the bug-repair task is taken as a candidate and put through a complete evaluate; a problem is set only if it compiles and fails in the test stage, and a mutant judged killed is evaluated once more and the problem is set only if it fails again. The re-parse after rewriting must not produce any new error; the comment-stripped original function is verified first (if it does not pass, no search is done). The input hash of mutate covers the oracle gate, the copies, the seed, `--max-tries`, whether the re-check is on, the relevant code, the sandbox and the host libclang; before use (problem setting, report) these are recomputed and compared item by item, and a mismatch is an immediate error; `check-mutants` can replay the re-check at any time; a problem whose `unchanged` candidate nevertheless passes the tests is dropped as a whole and named | `tests/test_mutate.py`, `tests/test_repair_pipeline.py`, `tests/test_repair_report.py` |
| Identifier-leak check on descriptions | The functional specification must not contain any project identifier (including the real L0 names): exact matching on word boundaries + normalized matching that ignores case / separators + phrase warnings; a list of common English words is let through, and allow-list hits are recorded as they are; the check is run again with the current checker when problems are set; on a leak the specification is regenerated with the list of leaked identifiers, at most 3 times, and a target that still leaks is excluded | `tests/test_describe.py` |
| End-to-end determinism | With a fixed seed, the target set, the rename table (on the same host), the transformation sites and the prompts are exactly the same; LLM calls use temperature=0 and are cached | Tests of the individual modules |
| Input-hash validation | Result files carry hashes of the source tree, the spec (only the fields that affect building and testing), the prompt template, the reply, the oracle gate, the evaluation code and the sandbox image; `report` validates record by record against tasks / generations / copies / gate, and records that do not match are excluded, with the reasons listed in the report | `tests/test_e2e_mock.py`, `tests/test_integrity.py` |
| No silent loss of samples | Failed generation calls are written to disk separately, `evaluate` writes a completeness record, the report is marked "incomplete" (「不完整」 in the Chinese-language report) and the command returns non-zero (`--allow-missing` lets it through); bad lines in result files are counted likewise | `tests/test_integrity.py` |
| Wording and path checks | Banned terms, absolute paths of the local machine, and whether the README agrees with the implementation (environment variables, command-line options, modules, limitations, data files), checked uniformly across the whole repository | `tests/test_hygiene.py` |

## 8. Limitations

Stated here plainly; please consult them when reading the results (this is the same content as the "Limitations" section of the report, maintained at a single point in `bench/limitations.py`, with `tests/test_hygiene.py` checking that the two agree):

1. **The functional specifications are generated by an LLM.** Ambiguity in a specification raises the failure rate at every level, so the L0 pass rate serves only as a reference for the upper bound; that a specification contains no identifiers is ensured by the identifier-leak check, whose criterion is exact matching on word boundaries plus normalized matching that ignores case / separators, plus a list of common English words that is let through (allow-list hits are recorded as they are in the description records); it is not a guarantee at the semantic level.
2. **The corpora are well-known small libraries.** At L0 the model may well have seen this code, which is exactly why the obfuscated controls exist; but the 9 libraries differ in popularity, so a difference on a single library cannot be taken directly as evidence of memorization: check whether the L0 → L1 changes in similarity and pass rate are consistent.
3. **C only.** The conclusions are not extrapolated to other languages; builds and tests are all gcc 13 plus each library's own tests.
4. **Single-function granularity.** Only one function is replaced at a time; all code outside the target function is the project's own code as it was (obfuscated at the corresponding level); multi-file feature additions are not evaluated.
5. **Test strength depends on each library's own tests.** stub-kill only guarantees that "a zero-value stub is caught by the tests", i.e. that the tests are not empty; it does not guarantee that the tests catch every logic error: a generated function that passes the tests may still contain errors the tests do not cover, so pass rates are overestimated. Some of the targets were killed because "the stub made the test program crash or time out" (runtime_crash / runtime_timeout); their tests do not assert what the function returns, so any wrong implementation that does not crash passes. The report gives pass rates stratified by kill mode; when reading the results, look at test_fail targets first.
6. **The structural and semantic transformations cover only a limited subset of the syntax.** Statements generated by macros, positions inside macro arguments, positions split by #if, compound assignments containing function calls and the like are never touched, so for libraries that use macros heavily (parson, cJSON's Unity assertion macros, vec) the sites actually changed are clearly fewer than the theoretical candidates; function bodies in branches of #if that are not active, or in conditional-compilation blocks that spec.cflags does not enable, are not changed either.
7. **L1 / L2 / L3 share the same set of replies.** L2 / L3 change only function bodies, and the prompt contains no function body, so the L1, L2 and L3 prompts are word-for-word identical, the model's replies at temperature=0 are identical too, and they share one cache key. What L2 / L3 measure is "whether the generated function, once put back into a project that has been structurally and semantically obfuscated, still compiles and passes the tests", plus the differences in the similarity metric; it is not the model's understanding of obfuscated function bodies. Only the difference between L0 and L1 reflects model behavior (the names in the prompt differ). To make L2 / L3 change what the model sees, the design would need a different arrangement (for example putting the bodies of the called functions into the context). The bug-repair task is such an arrangement: its prompt contains the obfuscated function body, so its L2 / L3 inputs differ from L1.
8. **The attribution of OOM and timeouts fluctuates with load.** Symbol renaming changes only names, and comments and strings still carry the original names, so no source comments are put in the prompts; concurrent evaluation is limited by the local CPU and memory (each sandbox container gets 1 CPU and at most 1 GB), so not too many corpora should run at the same time; for a generated function that allocates memory without bound, which comes first, the timeout (30 s for every corpus) or the 1 GB memory limit, depends on machine load: the former is attributed to runtime_timeout, while the latter is OOM-killed (rc=137, no timeout marker) and by the rules attributed to test_fail, so the attribution of such samples can differ under different loads.
9. **The rename table of symbol renaming depends on the host's system headers.** Symbol renaming parses with the libclang on the host, and spellings declared in system headers are never renamed; the host's libc headers and the borrowed compiler built-in header directory are not the same as those of gcc:13 in the sandbox, so after switching machines or upgrading the system, the same seed may produce a different rename table. The input hash of the obfuscation result carries a digest of the host's libclang environment, and if the environment changes, old copies are judged stale.
10. **The mutants of the bug-repair task are artificially injected single-point defects.** They do not match the distribution of real defects, and the mutants at different levels are not necessarily the same defect. mutate makes L1–L3 try first the site at their own level that corresponds to the site L0 chose; L1 changes only names, so the sites correspond one to one and this step is exact; L2 / L3 rewrite the function body and align by token, and a site whose tokens were rewritten falls back to reusing the operator only. The "comparable subset" in the report contains only targets that have the same site at all four levels and no pruned entries in the prompt context, and only the level differences on that subset are free of differences in the defect location; the "same operator at all four levels" subset of targets is always equal to the full set (reusing the operator guarantees this) and is not a control.
11. **The bug-repair task gives no test output.** It is harder than repair in real settings; the absolute pass rates cannot be compared with other benchmarks, so look only at the differences between levels.
12. **For repair problems killed by a crash or a timeout, the tests may not check the returned content.** After the repair, the code may pass as long as it neither crashes nor times out, the same limitation as in the generation task. The report gives two stratifications: how the mutant died (kill_mode: whether this particular defect makes an assertion fail or makes the test program crash or hang), and how the target is killed by the zero-value stub (stub_kill_mode: whether the tests assert on the returned content); the two are not equivalent, so read both tables, and look at test_fail first.
13. **The prompt context of the repair task has been pruned.** The dependency context was originally taken from the original function, and calls, macros and globals that the defect deleted would remain in it and leak a clue to the answer, so it is re-pruned against the function with the defect injected when problems are set (only prototypes, macros and global variables are pruned; type definitions are not). For a problem with pruned entries, repairing it may need names that are not in the prompt; its difficulty comes from an information gap rather than from obfuscation, so it is left out of the comparable subset, and its pass rate gets a table of its own.
14. **Similarity and the restored rate in the repair task are not memorization metrics.** The prompt already contains the original function with a difference in only one place, so as long as the candidate is not heavily rewritten, its similarity is naturally close to 1; any correct fix of a single-point defect is word for word equal to the original function, so restored is roughly equal to passing and cannot be distinguished from L0 recalling the function from memory. The question the repair task answers has to be read from how the pass rate changes between levels.
15. **How conspicuous a NEG defect is varies with the level.** The NEG rewrite is always !(c): this form almost never occurs in L0 / L1 code, so the defect is obvious at a glance; the if / else swap at L2 makes the code contain this form on its own, so the defect is masked. Part of the way NEG difficulty changes with the level has nothing to do with the factor the obfuscation is meant to test, so look at this row separately when reading the table stratified by operator.
16. **Project vocabulary inside string literals cannot be removed.** Changing a literal would change behavior, so in the mutant function text at L1 and above, project vocabulary inside strings and character constants (such as function names in error messages) is left as it is; a problem whose target function's own original name appears in a literal is voided when it is set (name_in_literal), and the rest can only be tallied and reported as they are.

## 9. Experiment results

Data as of 2026-10-28. The full tables are in `results/summary-v2.md` (per-sample records in `results/results-v2.jsonl` and `results/results-repair-v2.jsonl`); for interpretation see [`docs/实验结果.md`](docs/实验结果.md). 208 targets were selected. In the spec-generation task, the functional specifications of 11 targets still leaked project identifiers after retries and were voided, leaving 197 in the statistics; in the bug-repair task, 3 problems were voided because the target's own original name appears in a string literal, leaving 208 problems at L0 and 207 at each of L1–L3. temperature 0, pass@1.

### Spec-generation task: model × obfuscation level (pass rate = all of the project's own tests pass after the function is put back into the project)

| Model | L0 | L1 | L2 | L3 | L0→L1 paired difference | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| deepseek-flash | 85.3% | 49.2% | 48.7% | 49.2% | −36.0 pp | < 0.001 |
| deepseek-v4-pro | 80.2% | 51.3% | 50.8% | 50.8% | −28.9 pp | < 0.001 |
| deepseek-flash-thinking | 90.4% | 69.0% | 71.1% | 70.6% | −21.3 pp | < 0.001 |

L1 renames all identifiers in the project (test files included) uniformly to meaningless symbols, leaving the functional specification and the types in the signature unchanged; the L2 / L3 transformations happen inside function bodies the model cannot see, so they come out almost the same as L1 (see limitation 7). Similarity between the candidate and the original function (identifiers normalized): flash 0.825 → 0.701, v4-pro 0.776 → 0.714, thinking 0.845 → 0.802. At L1, 10.7% of thinking's replies were truncated when their reasoning hit the 32k-token limit and are counted as failures, so 69.0% is a lower bound.

### Bug-repair task: comparable subset (150 targets; the same defect at all four levels and no pruning of the prompt context)

| Model | L0 | L1 | L2 | L3 | L0→L1 | L0→L3 | unchanged rate (L1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek-flash | 92.7% | 70.7% | 70.7% | 72.0% | −22.0 pp | −20.7 pp | 6.0% |
| deepseek-v4-pro | 92.7% | 66.7% | 64.0% | 57.3% | −26.0 pp | −35.3 pp | 18.0% |
| deepseek-flash-thinking | 96.0% | 90.7% | 90.7% | 82.0% | −5.3 pp (p = 0.057) | −14.0 pp | 0.0% |

Full set (L0 208 problems, 207 for each of the other levels): flash 85.6% / 63.8% / 65.2% / 63.8%, v4-pro 83.2% / 59.9% / 58.5% / 52.7%, thinking 89.9% / 80.7% / 83.1% / 75.8%. unchanged = the model returned the defective function as is. On the repair task, 6% to 13% of thinking's replies were truncated when their reasoning hit the 32k-token limit (extract_fail) and are counted as failures, so its pass rate is a lower bound.

The start of the report spells out, point by point, how the inputs of the two tasks differ:

* Generation task (generate): the model gets the dependency declarations, the target function's signature and a functional specification written by an LLM, with no function body, and has to write the whole function; the L1 / L2 / L3 prompts are word-for-word identical, so only the difference between L0 and L1 reflects model behavior.
* Bug-repair task (repair): the model gets the dependency declarations and the complete function with one defect injected in that level's copy (comments stripped), with no functional specification, no test output and no statement of where the defect is, and has to return the corrected complete function; the obfuscation acts on the function body the model reads, so the L2 / L3 inputs differ from L1 and all four levels reflect model behavior.
* The two tasks have different problem types, and the tests are not equally sensitive to them, so absolute pass rates cannot be compared directly; look at how each changes between levels: the generation task answers "does writing a function depend on memorization?", and the repair task answers "does understanding and fixing a function depend on its surface form?".

### How to read the results

1. On both tasks, all three models drop once identifiers are removed; in the repair task v4-pro keeps dropping at L2 and L3, so it is more sensitive to structural and semantic rewriting than flash; flash with thinking enabled almost recovers the loss from renaming on the repair task (comparable subset −5.3 pp, McNemar p = 0.057), but still drops significantly, by 14 percentage points, under semantic rewriting (L3).
2. With the original names, flash leads v4-pro (+5.1 pp on the generation task); after renaming it falls behind instead (−2.1 pp): the advantage under the original names comes from memorization.
3. The more well-known the library, the larger the drop (generation task, flash: tiny-regex-c −62.5, mpc −55.1, cJSON −42.2 pp; vec and inih do not drop).
4. Cost: about 66 yuan for all experiments (account balance 89 → 23 yuan), of which thinking accounts for about 49 yuan across the two tasks.

## 10. Directory layout

```
bench/
  __init__.py __main__.py         package entry (Python version check), python3 -m bench
  vocab.py limitations.py provenance.py  shared vocabularies (levels / reply issues / attribution categories), the limitations list (shared by the README and the report), code and sandbox digests
  corpus.py sandbox.py clangcfg.py  corpus spec and source preparation, Docker sandbox, libclang configuration (the single source of parse arguments)
  extract.py coverage.py targets.py  function extraction and dependency closure, execution counts, target selection and stub-kill
  mutate.py                       the eight mutation operators of the bug-repair task (finding sites, rewriting), the same site across levels (token alignment) and mutant selection, kill check and re-check
  obfuscate/                      __init__ (main entry) symbols structure semantic verify _edits (rewriting infrastructure shared by the structural / semantic transformations, including bracket pairing and macro-call ranges; mutate uses it too)
  llm.py prompt.py describe.py postprocess.py  model calls and cache, prompts (including context pruning for the repair task), functional specifications, reply post-processing and hard-constraint checks
  runner.py attribute.py similarity.py  replacement and sandbox evaluation, error attribution, lexer and normalized similarity (including the normalized comparison used for unchanged / restored)
  metrics.py report.py            metric aggregation and reports
  experiment_record.py            aggregates the outputs from prepare through verify, plus mutate and check-mutants, into the data part of docs/实验记录.md
  store.py pipeline.py cli.py     workspace and data-file I/O, implementation of the stages, command line
corpus/specs/*.json               specs of the 9 corpora (build and test commands, filtering, vendored, keep_names, exclude_targets, allow_words)
corpus/src/                       copies of the corpus sources (generated by prepare, not checked in)
tests/                            unit tests and e2e (fixtures/mini is a small purpose-written project; conftest.py / helpers.py are the shared fixtures)
scripts/ci.sh                     CI script: fails if docker or a corpus is missing; no skipped test cases are allowed
pyproject.toml                    packaging and pytest configuration (Python >= 3.10; the only runtime dependency is clang==14.0)
results/                          data files (the cache directory is not checked in)
docs/                             design documents (including the appendix for the bug-repair task), experiment log
docker/                           reserved directory, currently empty (the sandbox uses the official gcc:13 image directly)
tmp/                              intermediate products (obfuscated copies, sandbox temporary directories, e2e output), not checked in
```

## 11. Development and testing

```bash
python3 -m pytest tests -q                       # full suite; needs docker and gcc:13; measured at about 20 minutes on the dev machine (pytest prints the total time when it finishes)
python3 -m pytest tests/test_e2e_mock.py -q      # integration: metrics, report, command line, whole-pipeline e2e on mini (bug-repair task included); measured at about 11 minutes on the same machine
BENCH_REAL_CORPORA=quick python3 -m pytest tests/test_symbols.py -q   # symbol renaming runs on only 4 real corpora; =0 skips them all, the default runs all 9
python3 -m pyflakes bench tests                  # static check; must report zero warnings
python3 -m bench.experiment_record --update docs/实验记录.md   # regenerate the data part of the experiment log from the outputs in results/
bash scripts/ci.sh                               # CI: pyflakes + full pytest (fails if docker / corpora are missing, fails on any skip) + e2e
```

Sandbox test cases (marked `needs_docker`) are **skipped** by default when docker or the image is unavailable, and the suite still shows as passing;
set `BENCH_REQUIRE_DOCKER=1` to make them fail instead, and `BENCH_REQUIRE_CORPUS=1` to make a missing real corpus fail as well. CI must set both (`scripts/ci.sh` already does),
otherwise the self-tests of the whole set of credibility measures are silently skipped on a machine without docker.

The timings above were measured on a 16-core / 15 GB development machine (20 min 05 s for the full suite, of which `test_e2e_mock.py` takes about 650 s) and vary with the machine and the load. To measure it yourself:
`python3 -m pytest tests -q --durations=15 --junitxml=tmp/pytest/junit.xml`, which prints the total time and the 15 slowest tests at the end; in the XML every test has a `time`, and summing by file gives each file's time.

Environment variables used only by tests: `BENCH_REQUIRE_DOCKER`, `BENCH_REQUIRE_CORPUS`, `BENCH_SANDBOX_IMAGE` (the sandbox image, default `gcc:13`),
`BENCH_REAL_CORPORA` (`quick` / `0`, see above). All temporary files are placed under the project's `tmp/`; the system temporary directory is not touched.
