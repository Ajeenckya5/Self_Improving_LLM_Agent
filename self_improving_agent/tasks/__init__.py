"""Long-horizon tasks with verifiers."""

from .base import Task, TaskResult
from .filesystem import get_filesystem_tasks
from .database import get_database_tasks


def get_all_tasks(env_root: str) -> list["Task"]:
    """Return instantiated tasks for experiments."""
    fs = [T(env_root) for T in get_filesystem_tasks()]
    db = [T(env_root) for T in get_database_tasks()]
    return fs + db
