"""Audit an exact publication allowlist or export a fresh source-only checkout."""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "release",
    "data",
    "models",
    "artifacts",
    "private",
    ".cache",
    ".venv",
    ".public-release",
    ".git",
    ".aws",
    ".codex",
    ".agents",
    "__pycache__",
}
SECRET_PATTERNS = (
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"gh[pousr]_[A-Za-z0-9]{30,}",
    r"github_pat_[A-Za-z0-9_]{30,}",
    r"AKIA[0-9A-Z]{16}",
    r"https?://[^\s/@:]+:[^\s/@]+@",
    r"/ho" + r"me/[^/\s]+/",
)
FORBIDDEN_SUFFIXES = {
    ".pkl",
    ".pt",
    ".pth",
    ".npz",
    ".parquet",
    ".csv",
    ".jsonl",
    ".log",
    ".pem",
    ".key",
    ".zip",
    ".gz",
}


def validate_files(root, names):
    """Fail closed on private paths, symlinks, unexpected binaries or secret patterns."""
    root = Path(root).resolve()
    for name in names:
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or any(p in FORBIDDEN for p in relative.parts)
        ):
            raise ValueError(f"Private or unsafe publication path: {name}")
        if relative.name.startswith(".env") or relative.suffix in FORBIDDEN_SUFFIXES:
            raise ValueError(f"Private artifact type: {name}")
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing, linked or escaping file: {name}")
        if any(
            parent.is_symlink()
            for parent in path.parents
            if parent != root and root in parent.parents
        ):
            raise ValueError(f"Linked parent directory: {name}")
        if path.stat().st_size > 2_000_000:
            raise ValueError(f"Unexpected large file: {name}")
        if relative.suffix == ".png" and relative.parent == Path("samples"):
            if not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"Invalid screenshot: {name}")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Unexpected binary: {name}") from exc
        if any(re.search(pattern, content) for pattern in SECRET_PATTERNS):
            raise ValueError(f"Potential secret or machine-local path in {name}; inspect locally")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export", type=Path, help="Create a new source-only directory; never overwrite"
    )
    args = parser.parse_args()
    names = [
        line.strip()
        for line in (ROOT / "PUBLIC_FILES.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate publication paths")
    required = {
        "LICENSE",
        "README.md",
        "RELATED_WORK.md",
        "PUBLIC_FILES.txt",
        ".gitignore",
        "src/explanation/reading_explainer.py",
    }
    if not required <= set(names):
        raise ValueError("Missing required release files")
    validate_files(ROOT, names)
    if "MIT License" not in (ROOT / "LICENSE").read_text():
        raise ValueError("Expected actual MIT LICENSE file")
    if args.export:
        destination = args.export.resolve()
        if destination.exists():
            raise ValueError("Export directory already exists; use a fresh directory")
        destination.mkdir(parents=True)
        for name in names:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        validate_files(destination, names)
        print(
            f"Exported {len(names)} allowlisted source files; no data, annotations, models or artifacts."
        )
    else:
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        tracked = {name for name in result.stdout.split("\0") if name}
        if tracked != set(names):
            raise ValueError(f"Tracked files differ from allowlist: {sorted(tracked ^ set(names))}")
        validate_files(ROOT, tracked)
        print(
            f"PASS: {len(tracked)} tracked files match the publication allowlist; LICENSE present."
        )


if __name__ == "__main__":
    main()
