"""
AgentBench OS environment simulator using Python's subprocess and tempfile.
Tasks are graded against ground-truth state captured at creation time.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OSTask:
    id: str
    description: str
    horizon: int
    setup_cmds: List[str]          # Commands run in sandbox before agent starts
    success_check: str             # Shell command that exits 0 on success
    env: "OSEnvironment" = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "horizon": self.horizon,
            "env": self.env,
        }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class OSEnvironment:
    """
    Sandboxed OS environment backed by a temporary directory.
    Supports: bash, read_file, write_file, check_output.
    """

    def __init__(self, task: OSTask):
        self.task = task
        self._tmpdir = tempfile.mkdtemp(prefix="agent_os_")
        self._sandbox = Path(self._tmpdir)
        self._success: Optional[bool] = None
        self._setup()

    # ------------------------------------------------------------------

    def step(self, action_str: str) -> str:
        """Parse and execute an agent action string; return observation."""
        action_str = action_str.strip()
        # Strip leading "Action: "
        if action_str.lower().startswith("action:"):
            action_str = action_str[7:].strip()

        if action_str.startswith("finish("):
            self._success = self._run_success_check()
            return "Task declared complete."

        # Match tool(args)
        m = re.match(r"(\w+)\((.*)\)$", action_str, re.DOTALL)
        if not m:
            return "Error: Could not parse action. Expected format: tool_name(arguments)"

        tool = m.group(1).strip()
        args = m.group(2).strip()

        if tool in ("bash", "check_output"):
            return self._bash(args)
        elif tool == "read_file":
            return self._read_file(args)
        elif tool == "write_file":
            return self._write_file(args)
        else:
            return f"Error: Unknown tool '{tool}' in OS environment."

    def is_success(self) -> bool:
        if self._success is None:
            self._success = self._run_success_check()
        return self._success

    def cleanup(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()

    # ------------------------------------------------------------------
    # Tool implementations (sandboxed)
    # ------------------------------------------------------------------

    def _bash(self, cmd: str) -> str:
        cmd = cmd.strip().strip('"').strip("'")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self._sandbox),
                env={**os.environ, "HOME": self._tmpdir, "SANDBOX": self._tmpdir},
            )
            out = (result.stdout + result.stderr).strip()
            return out[:3000] if out else "(command completed with no output)"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out."
        except Exception as exc:
            return f"Error: {exc}"

    def _read_file(self, path: str) -> str:
        path = path.strip().strip('"').strip("'")
        full_path = self._resolve(path)
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            return content[:3000]
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except Exception as exc:
            return f"Error: {exc}"

    def _write_file(self, args: str) -> str:
        parts = args.split(",", 1)
        if len(parts) < 2:
            return "Error: write_file requires (path, content)"
        path = parts[0].strip().strip('"').strip("'")
        content = parts[1].strip().strip('"').strip("'")
        full_path = self._resolve(path)
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            return f"Written {len(content)} bytes to {path}"
        except Exception as exc:
            return f"Error: {exc}"

    # ------------------------------------------------------------------

    def _setup(self) -> None:
        for cmd in self.task.setup_cmds:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(self._sandbox),
                env={**os.environ, "HOME": self._tmpdir, "SANDBOX": self._tmpdir},
            )
            if result.returncode != 0:
                logger.warning("Setup cmd failed: %s\n%s", cmd, result.stderr)

    def _run_success_check(self) -> bool:
        try:
            result = subprocess.run(
                self.task.success_check,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self._sandbox),
                env={**os.environ, "HOME": self._tmpdir, "SANDBOX": self._tmpdir},
            )
            return result.returncode == 0
        except Exception as exc:
            logger.warning("Success check failed: %s", exc)
            return False

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            # Redirect absolute paths into sandbox
            rel = str(p).lstrip("/")
            return self._sandbox / rel
        return self._sandbox / p


# ---------------------------------------------------------------------------
# Task generator
# ---------------------------------------------------------------------------

def generate_os_tasks(
    horizons: List[int] = None,
    n_per_horizon: int = 25,
) -> List[Dict[str, Any]]:
    """Generate OS tasks at multiple horizons; return list of task dicts."""
    if horizons is None:
        horizons = [5, 10, 15, 20]

    all_tasks = []
    task_id = 0

    for horizon in horizons:
        templates = _get_templates_for_horizon(horizon)
        for i in range(n_per_horizon):
            tmpl = templates[i % len(templates)]
            suffix = f"_{i}"
            task_data = tmpl(task_id, suffix)
            task_data.horizon = horizon
            env = OSEnvironment(task_data)
            task_dict = task_data.to_dict()
            task_dict["env"] = env
            all_tasks.append(task_dict)
            task_id += 1

    return all_tasks


def _get_templates_for_horizon(horizon: int):
    if horizon <= 5:
        return [_task_find_file, _task_check_permissions, _task_count_lines]
    elif horizon <= 10:
        return [_task_directory_structure, _task_move_files, _task_search_replace]
    elif horizon <= 15:
        return [_task_configure_file, _task_multi_file_edit, _task_script_debug]
    else:
        return [_task_git_workflow, _task_full_debug, _task_pipeline_setup]


# ---------------------------------------------------------------------------
# Task templates (H=5)
# ---------------------------------------------------------------------------

def _task_find_file(task_id: int, suffix: str) -> OSTask:
    filename = f"secret{suffix}.txt"
    content = f"The answer is 42{suffix}"
    return OSTask(
        id=f"os_find_{task_id}",
        description=(
            f"Find the file named '{filename}' in the current directory, "
            f"read its content, and output the answer it contains."
        ),
        horizon=5,
        setup_cmds=[
            f"mkdir -p data",
            f"echo '{content}' > data/{filename}",
        ],
        success_check=f"test -f data/{filename}",
    )


def _task_check_permissions(task_id: int, suffix: str) -> OSTask:
    return OSTask(
        id=f"os_perms_{task_id}",
        description=(
            "Create a file named 'run.sh' with the content '#!/bin/bash\\necho hello', "
            "make it executable, and verify it runs correctly."
        ),
        horizon=5,
        setup_cmds=[],
        success_check="test -x run.sh && bash run.sh | grep -q hello",
    )


def _task_count_lines(task_id: int, suffix: str) -> OSTask:
    n_lines = 10 + task_id % 20
    return OSTask(
        id=f"os_lines_{task_id}",
        description=(
            f"Count the number of lines in 'data.txt' and write the result "
            f"as a single integer to 'result.txt'."
        ),
        horizon=5,
        setup_cmds=[
            f"python3 -c \"open('data.txt','w').write('\\n'.join(['line']*{n_lines}))\"",
        ],
        success_check=f"test \"$(cat result.txt | tr -d '[:space:]')\" = \"{n_lines}\"",
    )


# ---------------------------------------------------------------------------
# Task templates (H=10)
# ---------------------------------------------------------------------------

def _task_directory_structure(task_id: int, suffix: str) -> OSTask:
    return OSTask(
        id=f"os_dirs_{task_id}",
        description=(
            "Create the directory structure: project/src/, project/tests/, project/docs/. "
            "In each directory create a placeholder file (e.g., .gitkeep). "
            "Then verify the structure exists."
        ),
        horizon=10,
        setup_cmds=[],
        success_check=(
            "test -d project/src && test -d project/tests && test -d project/docs"
        ),
    )


def _task_move_files(task_id: int, suffix: str) -> OSTask:
    return OSTask(
        id=f"os_move_{task_id}",
        description=(
            "Move all .txt files from the 'inbox/' directory to 'archive/', "
            "then create a file 'archive/manifest.txt' listing the moved files."
        ),
        horizon=10,
        setup_cmds=[
            "mkdir -p inbox archive",
            "for i in 1 2 3; do echo \"content$i\" > inbox/file$i.txt; done",
        ],
        success_check=(
            "test $(ls inbox/*.txt 2>/dev/null | wc -l) -eq 0 && "
            "test $(ls archive/*.txt | wc -l) -ge 3 && "
            "test -f archive/manifest.txt"
        ),
    )


def _task_search_replace(task_id: int, suffix: str) -> OSTask:
    return OSTask(
        id=f"os_sed_{task_id}",
        description=(
            "In the file 'config.ini', replace all occurrences of 'DEBUG=false' "
            "with 'DEBUG=true' and save the result."
        ),
        horizon=10,
        setup_cmds=[
            "printf '[settings]\\nDEBUG=false\\nLOG=false\\nDEBUG=false\\n' > config.ini",
        ],
        success_check="grep -q 'DEBUG=true' config.ini && ! grep -q 'DEBUG=false' config.ini",
    )


# ---------------------------------------------------------------------------
# Task templates (H=15)
# ---------------------------------------------------------------------------

def _task_configure_file(task_id: int, suffix: str) -> OSTask:
    return OSTask(
        id=f"os_cfg_{task_id}",
        description=(
            "Read 'app.conf', update the 'host' field to '127.0.0.1' and "
            "'port' to '8080', save the updated config, and create a backup at 'app.conf.bak'."
        ),
        horizon=15,
        setup_cmds=[
            "printf 'host=localhost\\nport=3000\\ndebug=true\\n' > app.conf",
        ],
        success_check=(
            "grep -q 'host=127.0.0.1' app.conf && "
            "grep -q 'port=8080' app.conf && "
            "test -f app.conf.bak"
        ),
    )


def _task_multi_file_edit(task_id: int, suffix: str) -> OSTask:
    return OSTask(
        id=f"os_multi_{task_id}",
        description=(
            "There are three Python files in 'src/': module_a.py, module_b.py, module_c.py. "
            "Add the line '# reviewed' as the first line of each file, "
            "then create 'src/REVIEW.md' listing all reviewed files."
        ),
        horizon=15,
        setup_cmds=[
            "mkdir -p src",
            "for m in a b c; do echo \"def func_$m(): pass\" > src/module_$m.py; done",
        ],
        success_check=(
            "head -1 src/module_a.py | grep -q '# reviewed' && "
            "head -1 src/module_b.py | grep -q '# reviewed' && "
            "head -1 src/module_c.py | grep -q '# reviewed' && "
            "test -f src/REVIEW.md"
        ),
    )


def _task_script_debug(task_id: int, suffix: str) -> OSTask:
    return OSTask(
        id=f"os_debug_{task_id}",
        description=(
            "The script 'compute.py' has a bug causing it to fail. "
            "Identify the bug, fix it, run the script to verify it outputs 'OK', "
            "and write the fix description to 'fix.txt'."
        ),
        horizon=15,
        setup_cmds=[
            textwrap.dedent("""\
                cat > compute.py << 'EOF'
                def compute(x):
                    return x * 2

                result = compute("hello")  # Bug: should pass an int
                print("OK" if result == 10 else "FAIL")
                EOF
            """),
        ],
        success_check="python3 compute.py | grep -q OK && test -f fix.txt",
    )


# ---------------------------------------------------------------------------
# Task templates (H=20)
# ---------------------------------------------------------------------------

def _task_git_workflow(task_id: int, suffix: str) -> OSTask:
    return OSTask(
        id=f"os_git_{task_id}",
        description=(
            "Initialize a git repository, create a file 'README.md' with the content "
            "'# My Project', commit it with message 'Initial commit', "
            "create a branch 'feature', add a file 'feature.py' with 'print(1)', "
            "commit it, and merge back to main."
        ),
        horizon=20,
        setup_cmds=[
            "git init -b main .",
            "git config user.email 'test@test.com'",
            "git config user.name 'Test'",
        ],
        success_check=(
            "git log --oneline | wc -l | grep -qE '[2-9]|[0-9]{2,}' && "
            "test -f README.md && test -f feature.py"
        ),
    )


def _task_full_debug(task_id: int, suffix: str) -> OSTask:
    return OSTask(
        id=f"os_fulldbg_{task_id}",
        description=(
            "The project has a broken test suite. Run 'python3 -m pytest tests/' to "
            "see failures, identify and fix each failing test in 'tests/test_main.py', "
            "ensure all tests pass, and write a summary to 'test_report.txt'."
        ),
        horizon=20,
        setup_cmds=[
            "mkdir -p tests src",
            textwrap.dedent("""\
                cat > src/main.py << 'EOF'
                def add(a, b):
                    return a + b

                def multiply(a, b):
                    return a * b
                EOF
            """),
            textwrap.dedent("""\
                cat > tests/test_main.py << 'EOF'
                import sys, os
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
                from main import add, multiply

                def test_add():
                    assert add(2, 3) == 5

                def test_multiply():
                    assert multiply(3, 4) == 12

                def test_add_negative():
                    assert add(-1, 1) == 0
                EOF
            """),
            "touch tests/__init__.py",
        ],
        success_check=(
            "python3 -m pytest tests/ -q 2>&1 | grep -q 'passed' && test -f test_report.txt"
        ),
    )


def _task_pipeline_setup(task_id: int, suffix: str) -> OSTask:
    return OSTask(
        id=f"os_pipeline_{task_id}",
        description=(
            "Set up a data processing pipeline: "
            "1) Create 'data/raw/input.csv' with 5 rows of dummy data. "
            "2) Write a Python script 'process.py' that reads input.csv, "
            "doubles numeric values, and writes to 'data/processed/output.csv'. "
            "3) Run the script and verify the output exists. "
            "4) Create 'pipeline.sh' that runs the full pipeline."
        ),
        horizon=20,
        setup_cmds=["mkdir -p data/raw data/processed"],
        success_check=(
            "test -f data/raw/input.csv && "
            "test -f data/processed/output.csv && "
            "test -f process.py && "
            "test -f pipeline.sh"
        ),
    )
