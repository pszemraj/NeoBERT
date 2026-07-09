#!/usr/bin/env python3
"""Run a full or quick GLUE suite from a directory of task configs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from neobert.config import ConfigLoader
from neobert.glue.tasks import GLUE_TASK_SPECS

QUICK_TASKS = ("rte", "mrpc", "stsb", "cola")
RUN_GLUE = Path(__file__).resolve().parent.parent / "run_glue.py"


def _parser() -> argparse.ArgumentParser:
    """Build the suite CLI parser.

    :return argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config_dir",
        type=Path,
        nargs="?",
        default=Path("configs/glue"),
        help="Directory containing <task>.yaml configs",
    )
    parser.add_argument(
        "--suite",
        choices=("all", "quick"),
        default="all",
        help=(
            "Task suite: 'all' runs every task and reports failures after the suite; "
            "'quick' runs small tasks and stops at the first failure"
        ),
    )
    parser.add_argument(
        "--model-name-or-path",
        help="Optional Hub model/path override forwarded to every task",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Log directory (default: logs/<config-directory-name>)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs and print commands without training",
    )
    return parser


def _stream_command(command: Sequence[str], log_path: Path) -> int:
    """Run a command while mirroring combined output to a log file.

    :param Sequence[str] command: Command and arguments to execute.
    :param Path log_path: Destination log file.
    :return int: Process exit code.
    """
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if process.stdout is None:  # pragma: no cover - guaranteed by stdout=PIPE
            raise RuntimeError("Failed to capture GLUE subprocess output")
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def run_suite(args: argparse.Namespace) -> int:
    """Run the selected GLUE suite.

    The full suite deliberately continues after task failures so one launch can
    expose every failing task, then returns a nonzero status. The quick suite is
    a smoke test and deliberately fails fast.

    :param argparse.Namespace args: Parsed CLI arguments.
    :return int: Zero on success, otherwise one.
    """
    tasks = tuple(GLUE_TASK_SPECS) if args.suite == "all" else QUICK_TASKS
    log_dir = args.log_dir or Path("logs") / args.config_dir.name
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)

    passed: list[str] = []
    failed: list[str] = []
    outputs: dict[str, str] = {}
    logs: dict[str, str] = {}

    print(f"Running {args.suite} GLUE suite from {args.config_dir}", flush=True)
    for task in tasks:
        config_path = args.config_dir / f"{task}.yaml"
        log_path = log_dir / f"{task}.log"
        logs[task] = str(log_path)

        if not config_path.is_file():
            print(f"FAIL {task}: config not found at {config_path}", file=sys.stderr)
            failed.append(task)
            outputs[task] = "(config missing)"
            if args.suite == "quick":
                break
            continue

        try:
            config = ConfigLoader.load(config_path)
        except (OSError, TypeError, ValueError) as exc:
            print(f"FAIL {task}: invalid config: {exc}", file=sys.stderr)
            failed.append(task)
            outputs[task] = "(invalid config)"
            if args.suite == "quick":
                break
            continue

        outputs[task] = str(config.trainer.output_dir)
        command = [sys.executable, str(RUN_GLUE), str(config_path)]
        if args.model_name_or_path:
            command.extend(("--model_name_or_path", args.model_name_or_path))

        if args.dry_run:
            print(f"DRY-RUN {task}: {' '.join(command)}")
            passed.append(task)
            continue

        print(f"Running {task} (log: {log_path})")
        return_code = _stream_command(command, log_path)
        if return_code == 0:
            print(f"PASS {task}")
            passed.append(task)
            continue

        print(f"FAIL {task}: exited with status {return_code}", file=sys.stderr)
        failed.append(task)
        if args.suite == "quick":
            break

    print(f"Passed ({len(passed)}): {', '.join(passed) or '(none)'}")
    print(f"Failed ({len(failed)}): {', '.join(failed) or '(none)'}")
    for task in (*passed, *failed):
        print(f"{task}: output={outputs[task]} log={logs[task]}")
    return int(bool(failed))


def main() -> None:
    """Run the GLUE suite CLI."""
    raise SystemExit(run_suite(_parser().parse_args()))


if __name__ == "__main__":
    main()
