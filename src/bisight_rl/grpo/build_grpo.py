"""Build frozen EasyR1 ChartQA train/dev artifacts without answer leakage."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from bisight_rl.common import digest, file_hash, now, read_jsonl, write_json, write_jsonl
from bisight_rl.grpo.reward import encode_ground_truth
from bisight_rl.quality import CONTROL
from bisight_rl.sft.common import ROLE_MARKERS, resolve_image, validate_compiled_dataset


def compile_rows(rows, data_root: Path, expected_split: str, verify_images: bool = True):
    compiled, seen = [], set()
    for line_number, row in enumerate(rows, 1):
        required = (
            "id",
            "split",
            "image_path",
            "image_file_sha256",
            "question",
            "answers",
            "source",
            "answer_kind",
        )
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Row {line_number} is missing fields: {missing}")
        sample_id = row["id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        seen.add(sample_id)
        if row["split"] != expected_split:
            raise ValueError(f"Expected {expected_split!r}, got {row['split']!r} for {sample_id}")
        if row.get("data_errors"):
            raise ValueError(f"GRPO source contains data errors: {sample_id}: {row['data_errors']}")
        question = row["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Empty question: {sample_id}")
        if CONTROL.search(question) or any(marker in question for marker in ROLE_MARKERS):
            raise ValueError(f"Question contains response/role control tokens: {sample_id}")
        references = row["answers"]
        ground_truth = encode_ground_truth(
            sample_id,
            references,
            source=row.get("source"),
            answer_kind=row.get("answer_kind"),
        )
        image_path = resolve_image(data_root, row["image_path"])
        if not image_path.is_file():
            raise ValueError(f"Missing image for {sample_id}: {image_path}")
        if verify_images:
            if file_hash(image_path) != row["image_file_sha256"]:
                raise ValueError(f"Image hash mismatch: {sample_id}")
            with Image.open(image_path) as image:
                image.verify()
        compiled.append(
            {
                "prompt_id": sample_id,
                "problem": question,
                "images": [row["image_path"]],
                "answer": ground_truth,
                "image_sha256": row["image_file_sha256"],
                "source": row.get("source"),
                "answer_kind": row.get("answer_kind"),
            }
        )
    return compiled


def _artifact(path: Path, rows: list[dict]) -> dict:
    return {
        "path": str(path),
        "sha256": file_hash(path),
        "rows": len(rows),
        "id_digest": digest([row["prompt_id"] for row in rows]),
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "answer_kind_counts": dict(Counter(row["answer_kind"] for row in rows)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, default=Path("data/rationales/full_master.jsonl"))
    parser.add_argument("--dev-source", type=Path, default=Path("data/candidates/dev_quick.jsonl"))
    parser.add_argument("--sft-data", type=Path, default=Path("data/processed/sft_drop50.jsonl"))
    parser.add_argument("--sft-manifest", type=Path, default=Path("data/processed/sft_drop50.manifest.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--prompt", type=Path, default=Path("prompts/adaptive.txt"))
    parser.add_argument("--train-output", type=Path, default=Path("data/processed/grpo_train.jsonl"))
    parser.add_argument("--dev-output", type=Path, default=Path("data/processed/grpo_dev_quick.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/grpo.manifest.json"))
    parser.add_argument("--expected-train", type=int, default=2000)
    parser.add_argument("--expected-dev", type=int, default=256)
    args = parser.parse_args()

    sft_rows, sft_manifest = validate_compiled_dataset(args.sft_data, args.sft_manifest, args.prompt)
    train_source = read_jsonl(args.train_source)
    dev_source = read_jsonl(args.dev_source)
    train_rows = compile_rows(train_source, args.data_root, "train")
    dev_rows = compile_rows(dev_source, args.data_root, "val")
    if len(train_rows) != args.expected_train or len(dev_rows) != args.expected_dev:
        raise ValueError(
            f"Unexpected GRPO sizes: train={len(train_rows)} (expected {args.expected_train}), "
            f"dev={len(dev_rows)} (expected {args.expected_dev})"
        )
    sft_ids = [row["id"] for row in sft_rows]
    grpo_ids = [row["prompt_id"] for row in train_rows]
    if grpo_ids != sft_ids:
        raise ValueError("GRPO train ids/order differ from the frozen SFT dataset")
    if set(grpo_ids) & {row["prompt_id"] for row in dev_rows}:
        raise ValueError("Train/dev prompt ids overlap")
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("Adaptive prompt is empty")

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.dev_output, dev_rows)
    manifest = {
        "status": "validated_grpo_dataset",
        "created_at": now(),
        "contract": {
            "train_source": str(args.train_source),
            "train_source_sha256": file_hash(args.train_source),
            "dev_source": str(args.dev_source),
            "dev_source_sha256": file_hash(args.dev_source),
            "sft_data": str(args.sft_data),
            "sft_data_sha256": file_hash(args.sft_data),
            "sft_manifest_sha256": file_hash(args.sft_manifest),
            "adaptive_prompt": str(args.prompt),
            "adaptive_prompt_sha256": file_hash(args.prompt),
            "dataset_revision": sft_manifest["contract"].get("dataset_revision"),
            "model": sft_manifest["contract"].get("model"),
            "model_revision": sft_manifest["contract"].get("model_revision"),
            "builder_sha256": file_hash(Path(__file__)),
            "reward_sha256": file_hash(Path(__file__).with_name("reward.py")),
        },
        "artifacts": {
            "train": _artifact(args.train_output, train_rows),
            "dev_quick": _artifact(args.dev_output, dev_rows),
        },
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
