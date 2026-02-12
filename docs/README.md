# NeoBERT Documentation

Core docs for training, evaluating, exporting, and maintaining NeoBERT.
This directory is the canonical home for behavior semantics.

## Essential Guides

- [Training](training.md) - runtime behavior and launch patterns
- [Configuration](configuration.md) - config schema and defaults
- [Evaluation](evaluation.md) - GLUE/MTEB behavior and caveats
- [Export](export.md) - HF export behavior and constraints

## Technical Reference

- [Architecture](architecture.md) - model internals and backend behavior
- [Testing](testing.md) - test execution and authoring conventions
- [Troubleshooting](troubleshooting.md) - common runtime/perf failures and fixes
- [Dev Notes](dev.md) - current status and engineering backlog

## Source-of-Truth Map

| Concept                                | Canonical doc                            | Notes                                                   |
| -------------------------------------- | ---------------------------------------- | ------------------------------------------------------- |
| Training runtime behavior              | [training.md](training.md)               | Script READMEs should link here for behavior semantics. |
| Field-level config defaults and schema | [configuration.md](configuration.md)     | Avoid duplicating defaults elsewhere.                   |
| Evaluation behavior/caveats            | [evaluation.md](evaluation.md)           | Includes current known limitations.                     |
| Export behavior/constraints            | [export.md](export.md)                   | Script README is command-focused only.                  |
| Failure diagnosis                      | [troubleshooting.md](troubleshooting.md) | Keep fixes centralized here.                            |
| Test process and authoring             | [testing.md](testing.md)                 | `tests/README.md` stays lightweight.                    |
| Architecture/runtime internals         | [architecture.md](architecture.md)       | Keep code-path details here, not in directory READMEs. |
| Engineering status/backlog             | [dev.md](dev.md)                         | Working notes, not user-facing runtime docs.            |

Directory-level READMEs (`configs/`, `scripts/`, `tests/`, `jobs/`) should stay
index/usage-focused and link back to these canonical docs instead of
re-declaring field semantics.

Non-canonical markdown:

- `outputs/**/README.md` files are generated model cards for exported artifacts.
  Treat them as run outputs, not repository source-of-truth docs.
- `CLAUDE.md` is assistant/tooling guidance and should defer to docs in this
  directory for runtime semantics.
- `.pytest_cache/README.md` is tool-generated metadata.

## Directory READMEs

- [Scripts](../scripts/README.md)
- [Evaluation scripts](../scripts/evaluation/README.md)
- [GLUE helpers](../scripts/evaluation/glue/README.md)
- [Export scripts](../scripts/export-hf/README.md)
- [Configs](../configs/README.md)
- [Jobs](../jobs/README.md)
- [Tests](../tests/README.md)
- [Test configs](../tests/configs/README.md)

## Project Links

- [Main README](../README.md)
