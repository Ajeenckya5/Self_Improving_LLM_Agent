"""Base task interface with verifier."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TaskResult:
    """Result of task verification."""
    success: bool
    message: str
    details: Optional[dict[str, Any]] = None


class Task(ABC):
    """Abstract base for long-horizon tasks."""

    def __init__(self, task_id: str, description: str, env_root: str):
        self.task_id = task_id
        self.description = description
        self.env_root = env_root

    @abstractmethod
    def setup(self) -> None:
        """Prepare the environment for the task."""
        pass

    @abstractmethod
    def verify(self, env_state: dict[str, Any]) -> TaskResult:
        """
        Check whether the task succeeded.
        Returns TaskResult(success=..., message=...).
        """
        pass

    @abstractmethod
    def get_available_tools(self) -> list[dict]:
        """Return tool schemas the agent can use."""
        pass

    def teardown(self) -> None:
        """Clean up after task (optional)."""
        pass
