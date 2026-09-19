"""Build a frozen, sanitized drop-50 SFT artifact from full_master.jsonl."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from bisight_rl.common import digest, file_hash, now, write_json, write_jsonl
from bisight_rl.quality import CONTROL, equivalent_for_supervision, parse_response
from bisight_rl.sft.common import ROLE_MARKERS, resolve_image


def canonical_assistant(rationale: str, answer: str) -> str:
    return f"<think>\n{rationale.strip()}\n</think>\n<answer>{answer.strip()}</answer>"


def compile_rows(rows, data_root: Path, verify_images: bool = True):
    compiled, seen = [], set()
    for line_number, row in enumerate(rows, 1):
        missing = [
            key
            for key in ("id", "split", "image_path", "image_file_sha256", "image_width", "image_height", "question", "canonical_answer", "rationale", "generation")
            if key not in row
        ]
        if missing:
            raise ValueError(f"Row {line_number} is missing fields: {missing}")
        sample_id = row["id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate sample ID: {sample_id}")
        seen.add(sample_id)
        if row["split"] != "train" or row.get("data_errors"):
            raise ValueError(f"SFT source must contain clean train rows: {sample_id}")
        question = row["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Empty question: {sample_id}")
        if CONTROL.search(question) or any(marker in question for marker in ROLE_MARKERS):
            raise ValueError(f"Question contains response/role control tokens: {sample_id}")
        generation = row["generation"]
        parsed = parse_response(generation.get("raw_response", ""))
        if parsed is None:
            raise ValueError(f"Invalid source response format: {sample_id}")
        if row["rationale"].strip() != parsed["rationale"].strip():
            raise ValueError(f"Top-level and response rationales disagree: {sample_id}")
        if not equivalent_for_supervision(row["canonical_answer"], parsed["answer"]):
            raise ValueError(f"Response answer differs from canonical answer: {sample_id}")
        assistant = canonical_assistant(row["rationale"], parsed["answer"])
        normalized = parse_response(assistant)
        if normalized is None or normalized["rationale"].strip() != row["rationale"].strip():
            raise ValueError(f"Canonical SFT response failed validation: {sample_id}")
        think_empty = not bool(row["rationale"].strip())
        expected_empty = f"<think>\n\n</think>\n<answer>{parsed['answer'].strip()}</answer>"
        if think_empty and assistant != expected_empty:
            raise ValueError(f"Empty-think whitespace changed: {sample_id}")

        image_path = resolve_image(data_root, row["image_path"])
        if not image_path.is_file():
            raise ValueError(f"Missing image: {sample_id}: {image_path}")
        if verify_images:
            if file_hash(image_path) != row["image_file_sha256"]:
                raise ValueError(f"Image hash mismatch: {sample_id}")
            with Image.open(image_path) as image:
                if image.size != (row["image_width"], row["image_height"]):
                    raise ValueError(f"Image dimensions changed: {sample_id}")
                image.verify()
        compiled.append(
            {
                "id": sample_id,
                "image_path": row["image_path"],
                "image_sha256": row["image_file_sha256"],
                "question": question,
                "assistant": assistant,
                "think_empty": think_empty,
                "source": row.get("source"),
                "answer_kind": row.get("answer_kind"),
            }
        )
    return compiled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/rationales/full_master.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--prompt", type=Path, default=Path("prompts/adaptive.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/sft_drop50.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/sft_drop50.manifest.json"))
    parser.add_argument("--expected-rows", type=int, default=2000)
    parser.add_argument("--expected-empty", type=int, default=1000)
    args = parser.parse_args()

    rows = []
    with args.source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {args.source}:{line_number}: {exc}") from exc
    compiled = compile_rows(rows, args.data_root, verify_images=True)
    empty = sum(row["think_empty"] for row in compiled)
    if len(compiled) != args.expected_rows or empty != args.expected_empty:
        raise ValueError(
            f"Unexpected SFT composition: rows={len(compiled)} (expected {args.expected_rows}), "
            f"empty={empty} (expected {args.expected_empty})"
        )
    if not args.prompt.read_text(encoding="utf-8").strip():
        raise ValueError("Adaptive prompt is empty")
    source_manifest_path = args.data_root / "manifests/source.json"
    source_manifest = json.loads(source_manifest_path.read_text()) if source_manifest_path.exists() else {}
    write_jsonl(args.output, compiled)
    manifest = {
        "status": "validated_sft_drop50",
        "created_at": now(),
        "contract": {
            "source_path": str(args.source),
            "source_sha256": file_hash(args.source),
            "adaptive_prompt_path": str(args.prompt),
            "adaptive_prompt_sha256": file_hash(args.prompt),
            "dataset_revision": source_manifest.get("dataset_revision"),
            "model": source_manifest.get("model"),
            "model_revision": source_manifest.get("model_revision"),
            "builder_sha256": file_hash(Path(__file__)),
            "sft_common_sha256": file_hash(Path(__file__).with_name("common.py")),
            "quality_module_sha256": file_hash(Path(__file__).resolve().parents[1] / "quality.py"),
            "stale_generation_metadata_ignored": [
                "generation.response_sha256",
                "generation.quality",
                "generation.raw_response_before_normalization",
            ],
        },
        "artifact": {
            "path": str(args.output),
            "sha256": file_hash(args.output),
            "rows": len(compiled),
            "empty_think": empty,
            "nonempty_think": len(compiled) - empty,
            "empty_id_digest": digest(sorted(row["id"] for row in compiled if row["think_empty"])),
            "source_counts": dict(Counter(row["source"] for row in compiled)),
            "answer_kind_counts": dict(Counter(row["answer_kind"] for row in compiled)),
        },
    }
    write_json(args.manifest, manifest)
    print(json.dumps({"output": str(args.output), **manifest["artifact"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
