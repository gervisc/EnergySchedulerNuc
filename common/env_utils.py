import os
from pathlib import Path


def load_env_file(env_path: Path, overwrite: bool = False) -> None:
    """Load key=value pairs from a local env file into os.environ."""
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if not key or not value:
                continue
            if overwrite or key not in os.environ:
                os.environ[key] = value


def load_repo_env_local(start_path: Path, overwrite: bool = False) -> None:
    """Load env.local from the repository root relative to a file path."""
    env_path = start_path.resolve().parents[1] / "env.local"
    load_env_file(env_path, overwrite=overwrite)
