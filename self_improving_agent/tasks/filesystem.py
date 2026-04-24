"""File-system long-horizon tasks with verifiers."""

from pathlib import Path
from typing import Any

from .base import Task, TaskResult


class FilesystemTask(Task):
    """Base for file-system tasks."""

    def get_available_tools(self) -> list[dict]:
        return [
            {
                "name": "list_dir",
                "description": "List files and directories at the given path",
                "parameters": {"path": "str"},
            },
            {
                "name": "read_file",
                "description": "Read contents of a file",
                "parameters": {"path": "str"},
            },
            {
                "name": "write_file",
                "description": "Write content to a file (creates or overwrites)",
                "parameters": {"path": "str", "content": "str"},
            },
            {
                "name": "create_dir",
                "description": "Create a directory (and parents if needed)",
                "parameters": {"path": "str"},
            },
            {
                "name": "delete_file",
                "description": "Delete a file",
                "parameters": {"path": "str"},
            },
            {
                "name": "move_file",
                "description": "Move or rename a file",
                "parameters": {"src": "str", "dst": "str"},
            },
        ]


class MultiStepFileOrganizeTask(FilesystemTask):
    """Organize files: create dirs, move files by extension."""

    def __init__(self, env_root: str):
        super().__init__(
            task_id="fs_organize",
            description="Create folders 'docs', 'images', 'data'. Move all .txt files to docs/, "
                        ".png and .jpg to images/, .csv to data/. Verify each folder exists and contains correct files.",
            env_root=env_root,
        )
        self.expected_files: dict[str, list[str]] = {}

    def setup(self) -> None:
        root = Path(self.env_root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "report.txt").write_text("Report content")
        (root / "notes.txt").write_text("Notes")
        (root / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (root / "photo.jpg").write_text("fake jpg")
        (root / "data.csv").write_text("a,b,c\n1,2,3")
        self.expected_files = {
            "docs": ["report.txt", "notes.txt"],
            "images": ["chart.png", "photo.jpg"],
            "data": ["data.csv"],
        }

    def verify(self, env_state: dict[str, Any]) -> TaskResult:
        root = Path(self.env_root)
        for folder, files in self.expected_files.items():
            folder_path = root / folder
            if not folder_path.is_dir():
                return TaskResult(
                    success=False,
                    message=f"Folder '{folder}' does not exist",
                    details={"missing": folder},
                )
            for f in files:
                if not (folder_path / f).exists():
                    return TaskResult(
                        success=False,
                        message=f"File {f} not in {folder}/",
                        details={"folder": folder, "missing": f},
                    )
        return TaskResult(success=True, message="All files correctly organized")


class NestedStructureTask(FilesystemTask):
    """Create a specific nested directory structure and a summary file."""

    def __init__(self, env_root: str):
        super().__init__(
            task_id="fs_nested",
            description="Create project/a/src, project/a/tests, project/b/config. "
                        "Add a file project/SUMMARY.txt with the line 'Structure complete'.",
            env_root=env_root,
        )

    def setup(self) -> None:
        root = Path(self.env_root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "junk.txt").write_text("ignore me")

    def verify(self, env_state: dict[str, Any]) -> TaskResult:
        root = Path(self.env_root)
        required = [
            "project/a/src",
            "project/a/tests",
            "project/b/config",
            "project/SUMMARY.txt",
        ]
        for p in required:
            if not (root / p).exists():
                return TaskResult(
                    success=False,
                    message=f"Required path missing: {p}",
                    details={"missing": p},
                )
        content = (root / "project/SUMMARY.txt").read_text().strip()
        if "Structure complete" not in content:
            return TaskResult(
                success=False,
                message="SUMMARY.txt does not contain 'Structure complete'",
                details={"content": content},
            )
        return TaskResult(success=True, message="Nested structure and summary correct")


class BackupAndCleanTask(FilesystemTask):
    """Create backup, then delete originals."""

    def __init__(self, env_root: str):
        super().__init__(
            task_id="fs_backup_clean",
            description="Create a 'backup' folder. Copy all .log files into backup. "
                        "Then delete the original .log files (not from backup).",
            env_root=env_root,
        )

    def setup(self) -> None:
        root = Path(self.env_root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "app.log").write_text("log1")
        (root / "error.log").write_text("log2")

    def verify(self, env_state: dict[str, Any]) -> TaskResult:
        root = Path(self.env_root)
        backup = root / "backup"
        if not backup.is_dir():
            return TaskResult(success=False, message="backup folder not found", details={})
        for f in ["app.log", "error.log"]:
            if not (backup / f).exists():
                return TaskResult(
                    success=False,
                    message=f"{f} not in backup/",
                    details={"missing": f},
                )
            if (root / f).exists():
                return TaskResult(
                    success=False,
                    message=f"Original {f} should be deleted",
                    details={"file": f},
                )
        return TaskResult(success=True, message="Backup created and originals removed")


class DedupAndArchiveTask(FilesystemTask):
    """Move duplicate (_copy) files into archive, keep originals."""

    def __init__(self, env_root: str):
        super().__init__(
            task_id="fs_dedup_archive",
            description="Create folder 'archive'. Move all files whose name ends with '_copy.txt' "
                        "into archive/. Keep the original files (file_a.txt, file_b.txt, unique.txt) in root.",
            env_root=env_root,
        )

    def setup(self) -> None:
        root = Path(self.env_root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "file_a.txt").write_text("original a")
        (root / "file_a_copy.txt").write_text("copy of a")
        (root / "file_b.txt").write_text("original b")
        (root / "file_b_copy.txt").write_text("copy of b")
        (root / "unique.txt").write_text("unique content")

    def verify(self, env_state: dict[str, Any]) -> TaskResult:
        root = Path(self.env_root)
        archive = root / "archive"
        if not archive.is_dir():
            return TaskResult(success=False, message="archive/ folder not found", details={})
        for f in ["file_a_copy.txt", "file_b_copy.txt"]:
            if not (archive / f).exists():
                return TaskResult(success=False, message=f"{f} not in archive/", details={"missing": f})
        for f in ["file_a.txt", "file_b.txt", "unique.txt"]:
            if not (root / f).exists():
                return TaskResult(success=False, message=f"Original {f} removed from root", details={"file": f})
        return TaskResult(success=True, message="Duplicates archived, originals intact")


def get_filesystem_tasks() -> list[type[Task]]:
    """Return filesystem task classes."""
    return [
        MultiStepFileOrganizeTask,
        NestedStructureTask,
        BackupAndCleanTask,
        DedupAndArchiveTask,
    ]
