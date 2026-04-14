"""Controlled execution environment for structured filesystem/database tasks."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ..tasks.base import Task


class ControlledEnvironment:
    """
    Sandboxed environment for executing agent tools.
    All file-system paths are constrained to env_root.
    Supports filesystem and SQLite database operations.
    """

    def __init__(self, env_root: str, task: Task):
        self.env_root = Path(env_root).resolve()
        self.task = task
        self.env_root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        """Resolve path and ensure it stays within env_root."""
        p = (self.env_root / path).resolve()
        if not str(p).startswith(str(self.env_root)):
            raise PermissionError(f"Path '{path}' escapes sandbox")
        return p

    def execute(self, tool_name: str, **kwargs) -> str:
        """Execute a tool and return an observation string."""
        try:
            if tool_name == "list_dir":
                return self._list_dir(kwargs.get("path", "."))
            if tool_name == "read_file":
                return self._read_file(kwargs.get("path", ""))
            if tool_name == "write_file":
                return self._write_file(kwargs.get("path", ""), kwargs.get("content", ""))
            if tool_name == "create_dir":
                return self._create_dir(kwargs.get("path", ""))
            if tool_name == "delete_file":
                return self._delete_file(kwargs.get("path", ""))
            if tool_name == "move_file":
                return self._move_file(kwargs.get("src", ""), kwargs.get("dst", ""))
            if tool_name == "execute_sql":
                return self._execute_sql(kwargs.get("sql", ""))
            if tool_name == "query_sql":
                return self._query_sql(kwargs.get("sql", ""))
            if tool_name == "list_tables":
                return self._list_tables()
            if tool_name == "get_schema":
                return self._get_schema(kwargs.get("table", ""))
            return f"Error: Unknown tool '{tool_name}'"
        except Exception as e:
            return f"Error: {e!s}"

    # ------------------------------------------------------------------
    # Filesystem tools
    # ------------------------------------------------------------------

    def _list_dir(self, path: str) -> str:
        p = self._resolve(path)
        if not p.exists():
            return f"Error: Path does not exist: {path}"
        if not p.is_dir():
            return f"Error: Not a directory: {path}"
        items = sorted(p.iterdir())
        names = [f"{x.name}/" if x.is_dir() else x.name for x in items]
        return "\n".join(names) if names else "(empty directory)"

    def _read_file(self, path: str) -> str:
        p = self._resolve(path)
        if not p.exists():
            return f"Error: File not found: {path}"
        if not p.is_file():
            return f"Error: Not a file: {path}"
        return p.read_text(errors="replace")

    def _write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {len(content)} bytes to {path}"

    def _create_dir(self, path: str) -> str:
        p = self._resolve(path)
        p.mkdir(parents=True, exist_ok=True)
        return f"Created directory: {path}"

    def _delete_file(self, path: str) -> str:
        p = self._resolve(path)
        if not p.exists():
            return f"Error: Path does not exist: {path}"
        if p.is_dir():
            return f"Error: Cannot delete directory with delete_file: {path}"
        p.unlink()
        return f"Deleted: {path}"

    def _move_file(self, src: str, dst: str) -> str:
        sp = self._resolve(src)
        dp = self._resolve(dst)
        if not sp.exists():
            return f"Error: Source not found: {src}"
        if sp.is_dir():
            return f"Error: Cannot move directory with move_file: {src}"
        dp.parent.mkdir(parents=True, exist_ok=True)
        sp.rename(dp)
        return f"Moved {src} → {dst}"

    # ------------------------------------------------------------------
    # Database tools
    # ------------------------------------------------------------------

    def _db_path(self) -> Path:
        return self.env_root / "task.db"

    def _execute_sql(self, sql: str) -> str:
        db = self._db_path()
        conn = sqlite3.connect(str(db))
        try:
            conn.execute(sql)
            conn.commit()
            return "SQL executed successfully"
        except sqlite3.Error as e:
            return f"SQL error: {e}"
        finally:
            conn.close()

    def _query_sql(self, sql: str) -> str:
        db = self._db_path()
        if not db.exists():
            return "Error: Database file does not exist."
        conn = sqlite3.connect(str(db))
        try:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            if not rows:
                return "(no rows)"
            cols = [d[0] for d in cur.description] if cur.description else []
            lines = [" | ".join(str(c) for c in cols)] if cols else []
            for r in rows:
                lines.append(" | ".join(str(x) for x in r))
            return "\n".join(lines)
        except sqlite3.Error as e:
            return f"Query error: {e}"
        finally:
            conn.close()

    def _list_tables(self) -> str:
        return self._query_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )

    def _get_schema(self, table: str) -> str:
        return self._query_sql(f"PRAGMA table_info({table})")

    # ------------------------------------------------------------------
    # State + cleanup
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Capture current environment state for task verification."""
        state: dict[str, Any] = {"env_root": str(self.env_root)}
        db = self._db_path()
        if db.exists():
            conn = sqlite3.connect(str(db))
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            )
            state["tables"] = [r[0] for r in cur.fetchall()]
            conn.close()
        return state
