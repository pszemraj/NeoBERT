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

Override values use the target field's type. Booleans accept `true/false`, `1/0`, `yes/no`, or `on/off`; optional values accept `null`, `none`, or `~`; lists and mappings use YAML syntax. Unknown paths and invalid values fail with the full field path.

In Python, `ConfigLoader.load(path, overrides=...)` accepts either the same list of tokens or a nested mapping. A mapping is merged before validation; a token list is applied to the hydrated config and then revalidated.

## YAML variables

An optional top-level `variables` mapping can remove repetition. An exact `$variables.path` value preserves its YAML type, while `{$variables.path}` or `${variables.path}` interpolates inside a string. Nested references are supported, cycles fail, unknown exact references fail, and unresolved inline tokens warn.

```yaml
variables:
  seq_len: 1024
  tag: pretrain-1024

dataset:
  max_seq_length: $variables.seq_len
tokenizer:
  max_length: $variables.seq_len
datacollator:
  max_length: $variables.seq_len
wandb:
  name: "neobert-{$variables.tag}"
```

Keep `model.max_position_embeddings` explicit: it is an architecture decision and must be at least the active training sequence length.
