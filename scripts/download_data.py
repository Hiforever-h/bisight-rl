"""Download immutable public dataset files and pin model metadata (no weights)."""
import argparse
import json
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

from bisight_rl.common import file_hash, now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    manifest_path = args.data_root / "manifests/source.json"
    api = HfApi(token=False)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["dataset"] != cfg["dataset"] or manifest["model"] != cfg["model"]:
            raise ValueError("Existing manifest uses a different dataset/model; use a new data root")
    else:
        dataset = api.dataset_info(cfg["dataset"], revision=cfg["dataset_revision"])
        model = api.model_info(cfg["model"], revision=cfg["model_revision"])
        files = sorted(s.rfilename for s in dataset.siblings if s.rfilename.endswith(".parquet") or s.rfilename == "README.md")
        manifest = {
            "dataset": cfg["dataset"], "dataset_revision": dataset.sha,
            "model": cfg["model"], "model_revision": model.sha,
            "created_at": now(), "config_sha256": file_hash(args.config),
            "files": [{"name": name} for name in files], "status": "downloading",
        }
        write_json(manifest_path, manifest)
    raw = args.data_root / "raw" / manifest["dataset_revision"]
    for entry in manifest["files"]:
        target = raw / entry["name"]
        if target.exists() and entry.get("sha256") == file_hash(target):
            print("verified", entry["name"], flush=True)
            continue
        print("downloading", entry["name"], flush=True)
        downloaded = hf_hub_download(
            repo_id=manifest["dataset"], repo_type="dataset",
            revision=manifest["dataset_revision"], filename=entry["name"],
            local_dir=raw, token=False,
        )
        entry.update(sha256=file_hash(downloaded), bytes=Path(downloaded).stat().st_size)
        write_json(manifest_path, manifest)
    manifest["status"] = "downloaded"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
