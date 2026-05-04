"""Diverse OS task generation for larger long-horizon benchmark sets."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .os_env import OSEnvironment, OSTask


@dataclass(frozen=True)
class DiverseOSTaskSpec:
    """An OS task plus the family label used for reporting."""

    family: str
    task: OSTask

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "id": self.task.id,
            "horizon": self.task.horizon,
            "family": self.family,
            "description": self.task.description,
            "setup_cmds": self.task.setup_cmds,
            "success_check": self.task.success_check,
        }


TaskFactory = Callable[[int, int, int], DiverseOSTaskSpec]


def generate_diverse_os_task_specs(
    horizons: list[int] | None = None,
    n_per_horizon: int = 100,
) -> list[DiverseOSTaskSpec]:
    """Generate deterministic, unique OS task specs without creating sandboxes."""
    if horizons is None:
        horizons = list(range(1, 51))
    if n_per_horizon <= 0:
        raise ValueError("n_per_horizon must be positive")

    specs: list[DiverseOSTaskSpec] = []
    factories = _diverse_factories()
    task_id = 0
    for horizon in horizons:
        for index in range(n_per_horizon):
            factory = factories[index % len(factories)]
            specs.append(factory(task_id, horizon, index))
            task_id += 1
    return specs


def generate_diverse_os_tasks(
    horizons: list[int] | None = None,
    n_per_horizon: int = 100,
) -> list[dict[str, Any]]:
    """Generate diverse OS tasks with live sandbox environments attached."""
    task_dicts: list[dict[str, Any]] = []
    for spec in generate_diverse_os_task_specs(horizons, n_per_horizon):
        env = OSEnvironment(spec.task)
        spec.task.env = env
        task_dict = spec.task.to_dict()
        task_dict["env"] = env
        task_dict["family"] = spec.family
        task_dicts.append(task_dict)
    return task_dicts


def generate_diverse_os_task_manifest(
    horizons: list[int] | None = None,
    n_per_horizon: int = 100,
) -> list[dict[str, Any]]:
    """Generate serializable metadata for the diverse OS task set."""
    return [
        spec.to_manifest_row()
        for spec in generate_diverse_os_task_specs(horizons, n_per_horizon)
    ]


def write_diverse_os_task_manifest(
    path: str | Path,
    horizons: list[int] | None = None,
    n_per_horizon: int = 100,
) -> Path:
    """Write the diverse task manifest as JSONL or CSV based on file suffix."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_diverse_os_task_manifest(horizons, n_per_horizon)

    if output_path.suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return output_path

    if output_path.suffix == ".csv":
        import pandas as pd

        pd.DataFrame(rows).to_csv(output_path, index=False)
        return output_path

    raise ValueError("Manifest path must end in .jsonl or .csv")


def _diverse_factories() -> list[TaskFactory]:
    return [
        _task_find_file_variant,
        _task_create_executable_variant,
        _task_count_lines_variant,
        _task_directory_structure_variant,
        _task_move_files_variant,
        _task_search_replace_variant,
        _task_configure_file_variant,
        _task_multi_file_review_variant,
        _task_script_debug_variant,
        _task_git_workflow_variant,
        _task_full_debug_variant,
        _task_pipeline_variant,
    ]


def _spec(family: str, task_id: int, task: OSTask) -> DiverseOSTaskSpec:
    task.id = f"diverse_{family}_{task_id:05d}"
    return DiverseOSTaskSpec(family=family, task=task)


def _q(value: str) -> str:
    return shlex.quote(value)


def _python_cmd(code: str) -> str:
    return "python3 -c " + shlex.quote(code)


def _write_text_cmd(path: str, content: str) -> str:
    code = (
        "from pathlib import Path\n"
        f"p = Path({path!r})\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        f"p.write_text({content!r})\n"
    )
    return _python_cmd(code)


def _parent(path: str) -> str:
    parent = Path(path).parent.as_posix()
    return parent if parent != "." else "."


def _task_find_file_variant(task_id: int, horizon: int, index: int) -> DiverseOSTaskSpec:
    filename = f"secret_h{horizon:02d}_{index:03d}.txt"
    nested_path = f"vault/h{horizon:02d}/case_{index:03d}/{filename}"
    output = f"answers/found_h{horizon:02d}_{index:03d}.txt"
    answer = f"answer-{horizon:02d}-{index:03d}"
    task = OSTask(
        id="",
        description=(
            f"Find the file named '{filename}' somewhere under the current directory, "
            f"read its content, and write the exact content to '{output}'."
        ),
        horizon=horizon,
        setup_cmds=[_write_text_cmd(nested_path, answer)],
        success_check=f"test \"$(cat {_q(output)} 2>/dev/null)\" = {_q(answer)}",
    )
    return _spec("os_find", task_id, task)


def _task_create_executable_variant(task_id: int, horizon: int, index: int) -> DiverseOSTaskSpec:
    script = f"scripts/run_h{horizon:02d}_{index:03d}.sh"
    output = f"logs/run_h{horizon:02d}_{index:03d}.out"
    message = f"hello-{horizon:02d}-{index:03d}"
    task = OSTask(
        id="",
        description=(
            f"Create an executable script '{script}' that prints '{message}', "
            f"run it, and save stdout to '{output}'."
        ),
        horizon=horizon,
        setup_cmds=[],
        success_check=(
            f"test -x {_q(script)} && "
            f"test \"$(cat {_q(output)} 2>/dev/null)\" = {_q(message)}"
        ),
    )
    return _spec("os_perms", task_id, task)


def _task_count_lines_variant(task_id: int, horizon: int, index: int) -> DiverseOSTaskSpec:
    source = f"inputs/lines_h{horizon:02d}_{index:03d}.txt"
    output = f"answers/line_count_h{horizon:02d}_{index:03d}.txt"
    n_lines = 5 + ((horizon * 13 + index) % 41)
    content = "\n".join(f"line-{i}" for i in range(n_lines)) + "\n"
    task = OSTask(
        id="",
        description=(
            f"Count the lines in '{source}' and write the integer total to '{output}'."
        ),
        horizon=horizon,
        setup_cmds=[_write_text_cmd(source, content)],
        success_check=(
            f"test \"$(cat {_q(output)} 2>/dev/null | tr -d '[:space:]')\" = "
            f"\"{n_lines}\""
        ),
    )
    return _spec("os_lines", task_id, task)


def _task_directory_structure_variant(task_id: int, horizon: int, index: int) -> DiverseOSTaskSpec:
    base = f"workspace_h{horizon:02d}_{index:03d}"
    dirs = [
        f"{base}/src",
        f"{base}/tests",
        f"{base}/docs",
        f"{base}/config/env_{index % 4}",
    ]
    checks = " && ".join(
        f"test -d {_q(d)} && test -f {_q(f'{d}/.gitkeep')}" for d in dirs
    )
    quoted_dirs = ", ".join(f"'{d}'" for d in dirs)
    task = OSTask(
        id="",
        description=(
            f"Create these directories: {quoted_dirs}. "
            "In each directory, create a '.gitkeep' placeholder file."
        ),
        horizon=horizon,
        setup_cmds=[],
        success_check=checks,
    )
    return _spec("os_dirs", task_id, task)


def _task_move_files_variant(task_id: int, horizon: int, index: int) -> DiverseOSTaskSpec:
    source = f"inbox_h{horizon:02d}_{index:03d}"
    archive = f"archive_h{horizon:02d}_{index:03d}"
    manifest = f"{archive}/manifest_h{horizon:02d}_{index:03d}.txt"
    file_count = 2 + (index % 5)
    setup = [f"mkdir -p {_q(source)} {_q(archive)}"]
    setup.extend(
        _write_text_cmd(f"{source}/note_{n}.txt", f"content-{horizon}-{index}-{n}\n")
        for n in range(1, file_count + 1)
    )
    task = OSTask(
        id="",
        description=(
            f"Move all .txt files from '{source}' to '{archive}' and write a "
            f"manifest to '{manifest}' listing the moved file names."
        ),
        horizon=horizon,
        setup_cmds=setup,
        success_check=(
            f"test $(find {_q(source)} -maxdepth 1 -name '*.txt' | wc -l) -eq 0 && "
            f"test $(find {_q(archive)} -maxdepth 1 -name 'note_*.txt' | wc -l) -eq {file_count} && "
            f"test -f {_q(manifest)}"
        ),
    )
    return _spec("os_move", task_id, task)


def _task_search_replace_variant(task_id: int, horizon: int, index: int) -> DiverseOSTaskSpec:
    config = f"configs/config_h{horizon:02d}_{index:03d}.ini"
    old = f"FEATURE_{horizon}_{index}=off"
    new = f"FEATURE_{horizon}_{index}=on"
    content = f"[settings]\n{old}\nLOG=true\n{old}\n"
    task = OSTask(
        id="",
        description=(
            f"In file '{config}', replace every '{old}' with '{new}' and save the file."
        ),
        horizon=horizon,
        setup_cmds=[_write_text_cmd(config, content)],
        success_check=(
            f"grep -q {_q(new)} {_q(config)} && ! grep -q {_q(old)} {_q(config)}"
        ),
    )
    return _spec("os_sed", task_id, task)


def _task_configure_file_variant(task_id: int, horizon: int, index: int) -> DiverseOSTaskSpec:
    config = f"apps/app_h{horizon:02d}_{index:03d}.conf"
    backup = f"{config}.bak"
    host = f"127.0.{horizon % 255}.{(index % 250) + 1}"
    port = str(8000 + ((horizon * 17 + index) % 1000))
    task = OSTask(
        id="",
        description=(
            f"Read '{config}', update the 'host' field to '{host}' and the "
            f"'port' field to '{port}', then create a backup at '{backup}'."
        ),
        horizon=horizon,
        setup_cmds=[_write_text_cmd(config, "host=localhost\nport=3000\ndebug=true\n")],
        success_check=(
            f"grep -q {_q(f'host={host}')} {_q(config)} && "
            f"grep -q {_q(f'port={port}')} {_q(config)} && "
            f"test -f {_q(backup)}"
        ),
    )
    return _spec("os_cfg", task_id, task)


def _task_multi_file_review_variant(task_id: int, horizon: int, index: int) -> DiverseOSTaskSpec:
    src = f"src_h{horizon:02d}_{index:03d}"
    files = [f"{src}/module_{name}.py" for name in ("alpha", "beta", "gamma")]
    marker = f"# reviewed h{horizon:02d} task{index:03d}"
    review = f"{src}/REVIEW_h{horizon:02d}_{index:03d}.md"
    setup = [_write_text_cmd(path, f"def {Path(path).stem}():\n    return True\n") for path in files]
    checks = " && ".join(
        f"head -1 {_q(path)} | grep -Fxq {_q(marker)}" for path in files
    )
    quoted_files = ", ".join(f"'{path}'" for path in files)
    task = OSTask(
        id="",
        description=(
            f"Prepend the line '{marker}' to each of these files: {quoted_files}. "
            f"Then create '{review}' listing their basenames."
        ),
        horizon=horizon,
        setup_cmds=setup,
        success_check=f"{checks} && test -f {_q(review)}",
    )
    return _spec("os_multi", task_id, task)


def _task_script_debug_variant(task_id: int, horizon: int, index: int) -> DiverseOSTaskSpec:
    script = f"debug/compute_h{horizon:02d}_{index:03d}.py"
    fix = f"debug/fix_h{horizon:02d}_{index:03d}.txt"
    content = """def compute(x):
    return x * 2

result = compute("bad")
print("OK" if result == 10 else "FAIL")
"""
    task = OSTask(
        id="",
        description=(
            f"Fix script '{script}' so it prints 'OK', then write a fix summary to '{fix}'."
        ),
        horizon=horizon,
        setup_cmds=[_write_text_cmd(script, content)],
        success_check=f"python3 {_q(script)} | grep -q OK && test -f {_q(fix)}",
    )
    return _spec("os_debug", task_id, task)


def _task_git_workflow_variant(task_id: int, horizon: int, index: int) -> DiverseOSTaskSpec:
    readme = f"README_h{horizon:02d}_{index:03d}.md"
    title = f"# Project H{horizon:02d} Task {index:03d}"
    branch = f"feature_h{horizon:02d}_{index:03d}"
    feature = f"feature_h{horizon:02d}_{index:03d}.py"
    task = OSTask(
        id="",
        description=(
            f"Initialize a git repository, create file '{readme}' with content '{title}', "
            f"commit it, create branch '{branch}', add file '{feature}' with "
            f"'print({index + 1})', commit it, and merge back to main."
        ),
        horizon=horizon,
        setup_cmds=[
            "git init -b main .",
            "git config user.email 'test@test.com'",
            "git config user.name 'Test'",
        ],
        success_check=(
            "git log --oneline | wc -l | grep -qE '[2-9]|[0-9]{2,}' && "
            f"test -f {_q(readme)} && test -f {_q(feature)}"
        ),
    )
    return _spec("os_git", task_id, task)


def _task_full_debug_variant(task_id: int, horizon: int, index: int) -> DiverseOSTaskSpec:
    report = f"reports/test_report_h{horizon:02d}_{index:03d}.txt"
    task = OSTask(
        id="",
        description=(
            "Run 'python3 -m pytest tests/' to verify the small project test suite, "
            f"ensure all tests pass, and write the pytest summary to '{report}'."
        ),
        horizon=horizon,
        setup_cmds=[
            "mkdir -p tests src",
            _write_text_cmd(
                "src/main.py",
                "def add(a, b):\n    return a + b\n\n"
                "def multiply(a, b):\n    return a * b\n",
            ),
            _write_text_cmd(
                "tests/test_main.py",
                "import sys, os\n"
                "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))\n"
                "from main import add, multiply\n\n"
                "def test_add():\n    assert add(2, 3) == 5\n\n"
                "def test_multiply():\n    assert multiply(3, 4) == 12\n",
            ),
            "touch tests/__init__.py",
        ],
        success_check=(
            "python3 -m pytest tests/ -q 2>&1 | grep -q 'passed' && "
            f"test -f {_q(report)}"
        ),
    )
    return _spec("os_fulldbg", task_id, task)


def _task_pipeline_variant(task_id: int, horizon: int, index: int) -> DiverseOSTaskSpec:
    base = f"data_h{horizon:02d}_{index:03d}"
    source = f"{base}/raw/input.csv"
    output = f"{base}/processed/output.csv"
    script = f"process_h{horizon:02d}_{index:03d}.py"
    pipeline = f"pipeline_h{horizon:02d}_{index:03d}.sh"
    task = OSTask(
        id="",
        description=(
            f"Set up a data pipeline: create '{source}' with five numeric rows, "
            f"write script '{script}' that doubles the values into '{output}', "
            f"run it, and create '{pipeline}' that reruns the pipeline."
        ),
        horizon=horizon,
        setup_cmds=[f"mkdir -p {_q(f'{base}/raw')} {_q(f'{base}/processed')}"],
        success_check=(
            f"test -f {_q(source)} && test -f {_q(output)} && "
            f"test -f {_q(script)} && test -f {_q(pipeline)}"
        ),
    )
    return _spec("os_pipeline", task_id, task)
