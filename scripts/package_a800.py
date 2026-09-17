"""Build a portable generation kit; no model weights, raw parquet, or Git metadata."""
import argparse
import json
import tarfile
from pathlib import Path

from bisight_rl.common import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/bisight-rl-a800.tar.gz"))
    args = parser.parse_args()
    root = Path.cwd()
    paths = []
    for relative in ("README.md", "PLAN.md", "DATA_PLAN.md", "pyproject.toml", "environment.yml", "requirements-local.lock.txt", "requirements-generation.txt", ".gitignore"):
        path = root / relative
        if path.exists():
            paths.append(path)
    for directory in ("src", "scripts", "tests", "configs", "prompts", "reports", "data/manifests", "data/images", "data/candidates", "data/reviews"):
        paths.extend(p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".pyc" != p.suffix and not any(part.endswith(".egg-info") for part in p.parts))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz", compresslevel=1) as tar:
        for path in sorted(paths):
            tar.add(path, arcname=str(Path("bisight-rl") / path.relative_to(root)), recursive=False)
    manifest = {"archive": str(args.output), "bytes": args.output.stat().st_size, "sha256": file_hash(args.output), "files": len(paths)}
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
