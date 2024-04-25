from pathlib import Path


def set_path(script_path: Path, dir_path: str, file_path: str) -> Path:
    """Set output path."""
    output_dir = script_path / Path(dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    new_path = output_dir / file_path
    return new_path
