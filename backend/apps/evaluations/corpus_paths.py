"""Resolve corpus paths recorded before the repository directory rename."""
from pathlib import Path


def resolve_corpus_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() and path.parts and path.parts[0] == "data":
        current = repo_root / "knowledge_base" / Path(*path.parts[1:])
        if current.exists():
            return current
    return path if path.is_absolute() else repo_root / path
