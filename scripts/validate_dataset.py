"""Validate frozen candidate artifacts without loading GPU models."""
import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from bisight_rl.common import file_hash, read_jsonl, write_json
from bisight_rl.data import normalize_question, stratum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()
    root = args.data_root
    build = json.loads((root / "manifests/build.json").read_text())
    for relative, info in build["artifacts"].items():
        path = root / relative
        if file_hash(path) != info["sha256"]:
            raise ValueError(f"Artifact changed: {relative}")
        if len(read_jsonl(path)) != info["rows"]:
            raise ValueError(f"Row count changed: {relative}")
    candidates = read_jsonl(root / "candidates/train_candidates.jsonl")
    initial = read_jsonl(root / "candidates/train_initial_2000.jsonl")
    pilot = read_jsonl(root / "candidates/pilot_100.jsonl")
    dev = read_jsonl(root / "candidates/dev_full.jsonl")
    test = read_jsonl(root / "candidates/test_full.jsonl")
    candidates_by_id = {r["id"]: r for r in candidates}
    assert len(candidates_by_id) == len(candidates)
    assert len(initial) == build["target_size"] and len(pilot) == 100
    assert Counter(stratum(r) for r in initial) == build["target_quotas"]
    assert len({(r["image_id"], normalize_question(r["question"])) for r in candidates}) == len(candidates)
    assert all(r["split"] == "train" and not r["data_errors"] and r["canonical_answer"] in r["answers"] for r in candidates)
    assert all(candidates_by_id[r["id"]] == r for r in initial + pilot)
    eval_images = {r["image_id"] for r in dev + test}
    assert not eval_images.intersection(r["image_id"] for r in candidates)
    assert not {r["image_id"] for r in dev}.intersection(r["image_id"] for r in test)
    pairs = read_jsonl(root / "reviews/cross_split_pairs.jsonl")
    banned = {r["left_image_id"] for r in pairs if r["left_split"] == "train" and r["status"] in {"confirmed_pixel_duplicate", "unresolved_near_duplicate"}}
    assert not banned.intersection(r["image_id"] for r in candidates)
    checked = set()
    if args.verify_images:
        for row in candidates + dev + test:
            if row["image_id"] in checked or "image_path" not in row:
                continue
            path = root / row["image_path"]
            if file_hash(path) != row["image_file_sha256"]:
                raise ValueError(f"Image checksum mismatch: {path}")
            with Image.open(path) as image:
                image.verify()
            checked.add(row["image_id"])
    result = {"status": "passed", "candidate_questions": len(candidates), "initial_questions": len(initial),
              "pilot_questions": len(pilot), "verified_images": len(checked), "rationale_count": 0,
              "note": "Validates candidate data only; GPU generation, human review and training loader masks are pending"}
    write_json(Path("reports/candidate_validation.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
