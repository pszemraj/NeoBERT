"""Regression tests for GLUE config generation helpers."""

import importlib.util
import sys
from pathlib import Path


def _load_builder_module():
    """Load ``build_glue_configs.py`` directly from its file path."""
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "evaluation" / "glue" / "build_glue_configs.py"
    spec = importlib.util.spec_from_file_location("build_glue_configs", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_wandb_section_enables_wandb_explicitly() -> None:
    """Ensure generated GLUE configs opt in to W&B explicitly."""
    builder = _load_builder_module()
    section = builder.build_wandb_section(
        project="proj",
        run_prefix="run",
        task="cola",
        checkpoint_step="100",
    )

    assert section["enabled"] is True
    assert section["name"] == "run-cola-100"


def test_build_trainer_section_does_not_emit_legacy_report_to() -> None:
    """Ensure generator does not produce deprecated trainer.report_to."""
    builder = _load_builder_module()
    trainer = builder.build_trainer_section(
        base_output_dir=Path("outputs/glue/test"),
        task_name="cola",
        overrides={},
    )

    assert "report_to" not in trainer
