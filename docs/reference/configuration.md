# Configuration

The complete, copyable field reference is [config_reference.yaml](config_reference.yaml). It documents every accepted key inline with its type, default, valid values, constraints, task scope, interactions, and relevant warnings. Start there; unknown keys are rejected.

Runnable maintained recipes live under [configs](../../configs/README.md). Tiny files under [tests/configs](../../tests/configs/README.md) are test fixtures, not tuning recommendations.

## Loading and overrides

Entrypoints accept a YAML path followed by dot-path overrides in any of these forms:

```text
trainer.max_steps=2000
--trainer.max_steps=2000
--trainer.max_steps 2000
```

Dot-path override values use the current target field's type. Booleans accept `true/false`, `1/0`, `yes/no`, or `on/off`; fields annotated as optional accept `null`, `none`, or `~` only when their current value is unset; lists and mappings use YAML syntax. String-backed fields keep these tokens as literal text, so follow the per-field clearing instructions in the YAML reference. Unknown paths and invalid values fail with the full field path.

In Python, `ConfigLoader.load(path, overrides=...)` accepts either the same list of tokens or a nested mapping. A mapping is merged before validation; a token list is applied to the hydrated config and then revalidated.

## YAML variables

The [YAML configuration reference](config_reference.yaml) defines the supported variable forms, failure behavior, and sequence-length interactions inline. Keep `model.max_position_embeddings` explicit because it is an architecture decision rather than a reusable interpolation value.
