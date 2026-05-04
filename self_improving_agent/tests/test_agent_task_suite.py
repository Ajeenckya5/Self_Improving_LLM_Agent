"""Integration-style tests for agent execution on representative tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ..agent.strategy_agent import StrategyAgent
from ..environments.os_env import OSEnvironment, OSTask


@dataclass(frozen=True)
class AgentTaskCase:
    """A deterministic task plus the LLM actions needed to solve it."""

    task: OSTask
    responses: list[str]


class ScriptedLLMClient:
    """Small deterministic LLM stub that returns one ReAct response per call."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        self.calls.append(messages)
        if not self._responses:
            return "Thought: The target state should now be complete.\nAction: finish(done)"
        return self._responses.pop(0)


def _minimal_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "model": {
            "primary": "gpt-4",
            "analyzer": "gpt-4",
            "embedding": "all-MiniLM-L6-v2",
        },
        "agent": {"max_steps": 8, "temperature": 0.0, "max_tokens": 300},
        "memory": {
            "top_k": 3,
            "similarity_threshold": 0.65,
            "db_path": str(tmp_path / "strategy_memory.db"),
        },
        "logging": {"llm_calls_log": str(tmp_path / "llm_calls.jsonl")},
    }


def _finish() -> str:
    return "Thought: The verification passed, so I can finish.\nAction: finish(done)"


def _agent_task_cases() -> list[AgentTaskCase]:
    return [
        AgentTaskCase(
            task=OSTask(
                id="agent_write_file",
                description="Create hello.txt with exactly the text world.",
                horizon=3,
                setup_cmds=[],
                success_check='test "$(cat hello.txt)" = "world"',
            ),
            responses=[
                "Thought: I need to create the target file.\nAction: write_file(hello.txt, world)",
                "Thought: I should verify the file content.\nAction: check_output(cat hello.txt)",
                _finish(),
            ],
        ),
        AgentTaskCase(
            task=OSTask(
                id="agent_count_lines",
                description="Count lines in data.txt and write the integer result to result.txt.",
                horizon=5,
                setup_cmds=["printf 'alpha\nbeta\ngamma\ndelta\n' > data.txt"],
                success_check='test "$(cat result.txt | tr -d \'[:space:]\')" = "4"',
            ),
            responses=[
                "Thought: I can use wc and normalize whitespace into the result file.\n"
                "Action: bash(wc -l < data.txt | tr -d ' ' > result.txt)",
                "Thought: I should inspect the generated answer.\nAction: check_output(cat result.txt)",
                _finish(),
            ],
        ),
        AgentTaskCase(
            task=OSTask(
                id="agent_replace_config",
                description="Replace every DEBUG=false entry in config.ini with DEBUG=true.",
                horizon=5,
                setup_cmds=["printf 'DEBUG=false\nLOG=false\nDEBUG=false\n' > config.ini"],
                success_check="grep -q 'DEBUG=true' config.ini && ! grep -q 'DEBUG=false' config.ini",
            ),
            responses=[
                "Thought: I need to rewrite the config in place.\n"
                "Action: bash(python3 -c \"from pathlib import Path; p=Path('config.ini'); "
                "p.write_text(p.read_text().replace('DEBUG=false','DEBUG=true'))\" && printf updated)",
                "Thought: I should verify there are no stale false flags.\n"
                "Action: check_output(grep DEBUG config.ini)",
                _finish(),
            ],
        ),
        AgentTaskCase(
            task=OSTask(
                id="agent_project_structure",
                description="Create project/src, project/tests, and project/docs with placeholder files.",
                horizon=6,
                setup_cmds=[],
                success_check="test -d project/src && test -d project/tests && test -d project/docs",
            ),
            responses=[
                "Thought: I need to create all required directories and placeholders.\n"
                "Action: bash(mkdir -p project/src project/tests project/docs && "
                "touch project/src/.gitkeep project/tests/.gitkeep project/docs/.gitkeep)",
                "Thought: I should list the resulting structure.\nAction: check_output(find project -maxdepth 2 -type d | sort)",
                _finish(),
            ],
        ),
        AgentTaskCase(
            task=OSTask(
                id="agent_backup_clean_logs",
                description="Copy all log files into backup and remove the original log files.",
                horizon=6,
                setup_cmds=[
                    "printf 'app log' > app.log",
                    "printf 'error log' > error.log",
                    "printf 'keep me' > notes.txt",
                ],
                success_check=(
                    "test -f backup/app.log && test -f backup/error.log && "
                    "! test -f app.log && ! test -f error.log && test -f notes.txt"
                ),
            ),
            responses=[
                "Thought: I need to preserve log copies before deleting originals.\n"
                "Action: bash(mkdir -p backup && cp *.log backup/ && rm *.log)",
                "Thought: I should confirm only backup copies remain.\nAction: check_output(find . -maxdepth 2 -type f | sort)",
                _finish(),
            ],
        ),
        AgentTaskCase(
            task=OSTask(
                id="agent_create_sqlite_users",
                description="Create task.db with a users table and insert Alice and Bob.",
                horizon=8,
                setup_cmds=[],
                success_check=(
                    "python3 -c \"import sqlite3; rows=sqlite3.connect('task.db')"
                    ".execute('SELECT id,name,email FROM users ORDER BY id').fetchall(); "
                    "assert rows == [(1,'Alice','alice@example.com'), (2,'Bob','bob@example.com')]\""
                ),
            ),
            responses=[
                "Thought: I can create the SQLite table and seed both required users.\n"
                "Action: bash(python3 -c \"import sqlite3; c=sqlite3.connect('task.db'); "
                "c.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)'); "
                "c.executemany('INSERT INTO users VALUES (?,?,?)', "
                "[(1,'Alice','alice@example.com'),(2,'Bob','bob@example.com')]); "
                "c.commit(); c.close()\" && printf seeded)",
                "Thought: I should confirm both rows were inserted.\n"
                "Action: check_output(python3 -c \"import sqlite3; "
                "print(sqlite3.connect('task.db').execute('SELECT COUNT(*) FROM users').fetchone()[0])\" "
                "&& printf checked)",
                _finish(),
            ],
        ),
    ]


@pytest.mark.parametrize("case", _agent_task_cases(), ids=lambda case: case.task.id)
def test_strategy_agent_solves_scripted_sandbox_tasks(
    case: AgentTaskCase,
    tmp_path: Path,
) -> None:
    llm_client = ScriptedLLMClient(case.responses)
    agent = StrategyAgent(
        config=_minimal_config(tmp_path),
        llm_client=llm_client,
        retriever=None,
    )

    with OSEnvironment(case.task) as env:
        case.task.env = env
        success, trace = agent.run(
            case.task.to_dict(),
            strategies=[
                {
                    "strategy_text": "Make the state change, inspect the result, then finish only after verification.",
                    "tags": ["verification"],
                }
            ],
        )

    assert success is True, trace.to_text()
    assert trace.final_success is True
    assert trace.total_steps == len(case.responses)
    assert len(trace.steps) == len(case.responses)
    assert trace.steps[-1].action.lower().startswith("action: finish")
    assert all(not step.observation.lower().startswith("error") for step in trace.steps[:-1]), trace.to_text()
    assert len(llm_client.calls) == len(case.responses)
