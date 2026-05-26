#!/usr/bin/env python3
"""
AdaptAgent scripted benchmark.

Validates the 6 long-horizon tasks (3 filesystem + 3 database) using
deterministic scripted action sequences — no API key required.
The LLM call is replaced with a scripted sequence; the environment,
tool execution, and verifiers are all real.

Reports: task completion rate, steps per task, and strategy memory
retrieval round-trip latency.

Usage:
    python bench_scripted.py
"""
from __future__ import annotations

import sys
import time
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from environment.controlled import ControlledEnvironment
from tasks.filesystem import MultiStepFileOrganizeTask, NestedStructureTask, BackupAndCleanTask
from tasks.database import CreateTableAndInsertTask, AlterTableTask, JoinAndAggregateTask
from strategy_memory.store import StrategyMemory


# ---------------------------------------------------------------------------
# Scripted action sequences — (tool_name, kwargs) per task
# ---------------------------------------------------------------------------

FS_ORGANIZE_SCRIPT = [
    ("list_dir",   {"path": "."}),
    ("create_dir", {"path": "docs"}),
    ("create_dir", {"path": "images"}),
    ("create_dir", {"path": "data"}),
    ("move_file",  {"src": "report.txt",  "dst": "docs/report.txt"}),
    ("move_file",  {"src": "notes.txt",   "dst": "docs/notes.txt"}),
    ("move_file",  {"src": "chart.png",   "dst": "images/chart.png"}),
    ("move_file",  {"src": "photo.jpg",   "dst": "images/photo.jpg"}),
    ("move_file",  {"src": "data.csv",    "dst": "data/data.csv"}),
]

FS_NESTED_SCRIPT = [
    ("create_dir",  {"path": "project/a/src"}),
    ("create_dir",  {"path": "project/a/tests"}),
    ("create_dir",  {"path": "project/b/config"}),
    ("write_file",  {"path": "project/SUMMARY.txt", "content": "Structure complete\n"}),
]

FS_BACKUP_SCRIPT = [
    ("list_dir",   {"path": "."}),
    ("create_dir", {"path": "backup"}),
    ("read_file",  {"path": "app.log"}),
    ("write_file", {"path": "backup/app.log",   "content": "log1"}),
    ("read_file",  {"path": "error.log"}),
    ("write_file", {"path": "backup/error.log", "content": "log2"}),
    ("delete_file",{"path": "app.log"}),
    ("delete_file",{"path": "error.log"}),
]

DB_CREATE_SCRIPT = [
    ("list_tables", {}),
    ("execute_sql", {"sql": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)"}),
    ("execute_sql", {"sql": "INSERT INTO users VALUES (1, 'Alice', 'alice@example.com')"}),
    ("execute_sql", {"sql": "INSERT INTO users VALUES (2, 'Bob', 'bob@example.com')"}),
    ("query_sql",   {"sql": "SELECT * FROM users"}),
]

DB_ALTER_SCRIPT = [
    ("list_tables", {}),
    ("get_schema",  {"table": "products"}),
    ("execute_sql", {"sql": "ALTER TABLE products ADD COLUMN price REAL"}),
    ("execute_sql", {"sql": "UPDATE products SET price=9.99  WHERE id=1"}),
    ("execute_sql", {"sql": "UPDATE products SET price=14.50 WHERE id=2"}),
    ("query_sql",   {"sql": "SELECT * FROM products"}),
]

DB_JOIN_SCRIPT = [
    ("list_tables", {}),
    ("execute_sql", {"sql": "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)"}),
    ("execute_sql", {"sql": "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL)"}),
    ("execute_sql", {"sql": "INSERT INTO customers VALUES (1,'John'),(2,'Jane')"}),
    ("execute_sql", {"sql": "INSERT INTO orders VALUES (1,1,100),(2,1,50),(3,2,75)"}),
    ("execute_sql", {"sql": (
        "CREATE VIEW customer_totals AS "
        "SELECT c.name, SUM(o.amount) AS total "
        "FROM customers c JOIN orders o ON c.id=o.customer_id "
        "GROUP BY c.id, c.name"
    )}),
    ("query_sql",   {"sql": "SELECT * FROM customer_totals"}),
]

TASK_SCRIPTS = [
    ("fs_organize",    MultiStepFileOrganizeTask, FS_ORGANIZE_SCRIPT),
    ("fs_nested",      NestedStructureTask,       FS_NESTED_SCRIPT),
    ("fs_backup_clean",BackupAndCleanTask,         FS_BACKUP_SCRIPT),
    ("db_create",      CreateTableAndInsertTask,   DB_CREATE_SCRIPT),
    ("db_alter",       AlterTableTask,             DB_ALTER_SCRIPT),
    ("db_join",        JoinAndAggregateTask,       DB_JOIN_SCRIPT),
]


def run_task(task_id: str, TaskClass, script, env_root: Path) -> dict:
    task_root = env_root / task_id
    shutil.rmtree(task_root, ignore_errors=True)
    task_root.mkdir(parents=True)

    task = TaskClass(env_root=str(task_root))
    task.setup()
    env = ControlledEnvironment(str(task_root), task)

    observations = []
    t0 = time.monotonic()
    for tool, kwargs in script:
        obs = env.execute(tool, **kwargs)
        observations.append((tool, obs))
    elapsed_ms = (time.monotonic() - t0) * 1000

    state = env.get_state()
    result = task.verify(state)

    return {
        "task_id": task_id,
        "success": result.success,
        "message": result.message,
        "steps": len(script),
        "elapsed_ms": round(elapsed_ms, 1),
    }


def bench_strategy_memory(tmp_path: Path) -> dict:
    """Measure strategy memory store+retrieve round-trip."""
    mem = StrategyMemory(persist_path=tmp_path / "chroma_bench")

    lessons = [
        ("fs_organize", "Organize files by extension", "element_error", "Always list directory before moving files to confirm filenames."),
        ("db_alter",    "Add column and update rows",  "schema_error",  "Check schema with get_schema before ALTER TABLE to avoid duplicates."),
        ("fs_backup",   "Backup log files",            "path_error",    "Verify source file exists before write to avoid empty backup."),
    ]

    t0 = time.monotonic()
    for task_id, desc, cat, strat in lessons:
        mem.add(task_id=task_id, task_description=desc, failure_category=cat, corrective_strategy=strat)
    store_ms = (time.monotonic() - t0) * 1000

    t1 = time.monotonic()
    retrieved = mem.retrieve("organize files and move them to folders", top_k=2)
    retrieve_ms = (time.monotonic() - t1) * 1000

    return {
        "strategies_stored": len(lessons),
        "retrieved": len(retrieved),
        "store_ms": round(store_ms, 1),
        "retrieve_ms": round(retrieve_ms, 1),
    }


def main():
    print("=" * 62)
    print("AdaptAgent — Long-Horizon Task Benchmark")
    print("Tasks: 3 filesystem + 3 database  |  No API key required")
    print("=" * 62)

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        env_root = Path(tmp) / "tasks"
        env_root.mkdir()

        for task_id, TaskClass, script in TASK_SCRIPTS:
            r = run_task(task_id, TaskClass, script, env_root)
            results.append(r)
            status = "PASS" if r["success"] else "FAIL"
            print(f"  [{status}] {r['task_id']:20s}  steps={r['steps']:2d}  "
                  f"{r['elapsed_ms']:6.1f}ms  — {r['message']}")

        print()
        print("  Strategy memory benchmark:")
        mem_root = Path(tmp) / "mem"
        mem_root.mkdir()
        mem = bench_strategy_memory(mem_root)
        print(f"    Stored {mem['strategies_stored']} strategies in {mem['store_ms']:.0f}ms")
        print(f"    Retrieved {mem['retrieved']} relevant strategies in {mem['retrieve_ms']:.0f}ms")

    n = len(results)
    passed = sum(1 for r in results if r["success"])
    avg_steps = sum(r["steps"] for r in results) / n
    avg_latency = sum(r["elapsed_ms"] for r in results) / n

    print()
    print("=" * 62)
    print(f"  Tasks:              {n}")
    print(f"  Completion rate:    {passed/n:.0%}  ({passed}/{n} tasks)")
    print(f"  Avg steps/task:     {avg_steps:.1f}")
    print(f"  Avg latency:        {avg_latency:.0f} ms/task (tool execution only)")
    print(f"  Strategy retrieval: {mem['retrieve_ms']:.0f} ms (sentence-transformers + ChromaDB)")
    print("=" * 62)


if __name__ == "__main__":
    main()
