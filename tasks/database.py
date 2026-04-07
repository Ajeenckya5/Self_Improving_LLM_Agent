"""Database long-horizon tasks with verifiers."""

import sqlite3
from pathlib import Path
from typing import Any

from .base import Task, TaskResult


class DatabaseTask(Task):
    """Base for SQLite database tasks."""

    @property
    def db_path(self) -> Path:
        return Path(self.env_root) / "task.db"

    def get_available_tools(self) -> list[dict]:
        return [
            {
                "name": "execute_sql",
                "description": "Execute a SQL statement (SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, etc.)",
                "parameters": {"sql": "str"},
            },
            {
                "name": "query_sql",
                "description": "Execute a SELECT query and return results",
                "parameters": {"sql": "str"},
            },
            {
                "name": "list_tables",
                "description": "List all tables in the database",
                "parameters": {},
            },
            {
                "name": "get_schema",
                "description": "Get schema for a table",
                "parameters": {"table": "str"},
            },
        ]


class CreateTableAndInsertTask(DatabaseTask):
    """Create a table with correct schema and insert rows."""

    def __init__(self, env_root: str):
        super().__init__(
            task_id="db_create_insert",
            description="Create a table 'users' with columns id (INTEGER PK), name (TEXT), email (TEXT). "
                        "Insert 2 rows: (1, 'Alice', 'alice@example.com') and (2, 'Bob', 'bob@example.com').",
            env_root=env_root,
        )

    def setup(self) -> None:
        Path(self.env_root).mkdir(parents=True, exist_ok=True)
        if self.db_path.exists():
            self.db_path.unlink()
        conn = sqlite3.connect(str(self.db_path))
        conn.close()

    def verify(self, env_state: dict[str, Any]) -> TaskResult:
        if not self.db_path.exists():
            return TaskResult(success=False, message="Database file not found", details={})
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
            )
            if not cur.fetchone():
                return TaskResult(
                    success=False,
                    message="Table 'users' does not exist",
                    details={},
                )
            cur = conn.execute("PRAGMA table_info(users)")
            cols = {row[1]: row[2] for row in cur.fetchall()}
            for required in ["id", "name", "email"]:
                if required not in cols:
                    return TaskResult(
                        success=False,
                        message=f"Column '{required}' missing",
                        details={"columns": list(cols.keys())},
                    )
            cur = conn.execute("SELECT id, name, email FROM users ORDER BY id")
            rows = cur.fetchall()
            expected = [
                (1, "Alice", "alice@example.com"),
                (2, "Bob", "bob@example.com"),
            ]
            if rows != expected:
                return TaskResult(
                    success=False,
                    message=f"Expected rows {expected}, got {rows}",
                    details={"rows": rows},
                )
            return TaskResult(success=True, message="Table and data correct")
        finally:
            conn.close()


class AlterTableTask(DatabaseTask):
    """Add a column and update existing rows."""

    def __init__(self, env_root: str):
        super().__init__(
            task_id="db_alter",
            description="The 'products' table exists with id and name. Add column 'price' (REAL). "
                        "Update product id=1 to price=9.99 and id=2 to price=14.50.",
            env_root=env_root,
        )

    def setup(self) -> None:
        Path(self.env_root).mkdir(parents=True, exist_ok=True)
        if self.db_path.exists():
            self.db_path.unlink()
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT)"
        )
        conn.execute("INSERT INTO products VALUES (1, 'Widget')")
        conn.execute("INSERT INTO products VALUES (2, 'Gadget')")
        conn.commit()
        conn.close()

    def verify(self, env_state: dict[str, Any]) -> TaskResult:
        if not self.db_path.exists():
            return TaskResult(success=False, message="Database file not found", details={})
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.execute("PRAGMA table_info(products)")
            cols = {row[1]: row[2] for row in cur.fetchall()}
            if "price" not in cols:
                return TaskResult(
                    success=False,
                    message="Column 'price' not added",
                    details={"columns": list(cols.keys())},
                )
            cur = conn.execute("SELECT id, price FROM products ORDER BY id")
            rows = cur.fetchall()
            expected = [(1, 9.99), (2, 14.5)]
            for (eid, epr), (rid, rpr) in zip(expected, rows):
                if rid != eid or abs(float(rpr) - float(epr)) > 0.01:
                    return TaskResult(
                        success=False,
                        message=f"Price mismatch: expected {expected}, got {rows}",
                        details={"rows": rows},
                    )
            return TaskResult(success=True, message="Column added and values updated")
        finally:
            conn.close()


class JoinAndAggregateTask(DatabaseTask):
    """Create related tables, insert data, run a join/aggregate query."""

    def __init__(self, env_root: str):
        super().__init__(
            task_id="db_join_agg",
            description="Create tables: orders (id, customer_id, amount) and customers (id, name). "
                        "Insert customers 1-John, 2-Jane. Insert orders: (1,1,100), (2,1,50), (3,2,75). "
                        "Create a view 'customer_totals' that shows each customer name and total order amount.",
            env_root=env_root,
        )

    def setup(self) -> None:
        Path(self.env_root).mkdir(parents=True, exist_ok=True)
        if self.db_path.exists():
            self.db_path.unlink()
        conn = sqlite3.connect(str(self.db_path))
        conn.close()

    def verify(self, env_state: dict[str, Any]) -> TaskResult:
        if not self.db_path.exists():
            return TaskResult(success=False, message="Database file not found", details={})
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='view' AND name='customer_totals'"
            )
            if not cur.fetchone():
                return TaskResult(
                    success=False,
                    message="View 'customer_totals' not found",
                    details={},
                )
            cur = conn.execute(
                "SELECT name, total FROM customer_totals ORDER BY name"
            )
            rows = sorted(cur.fetchall(), key=lambda r: r[0])
            expected = [("Jane", 75.0), ("John", 150.0)]  # John: 100+50
            for (en, et), (rn, rt) in zip(sorted(expected, key=lambda x: x[0]), rows):
                if rn != en or abs(float(rt) - float(et)) > 0.01:
                    return TaskResult(
                        success=False,
                        message=f"View data mismatch: expected {expected}, got {rows}",
                        details={"rows": rows},
                    )
            return TaskResult(success=True, message="View correct")
        finally:
            conn.close()


def get_database_tasks() -> list[type[Task]]:
    """Return all database task classes."""
    return [
        CreateTableAndInsertTask,
        AlterTableTask,
        JoinAndAggregateTask,
    ]
