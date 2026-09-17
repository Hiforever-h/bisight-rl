"""Normalize ChartQA, audit cross-split images, freeze ordered candidate pools."""
import argparse
import io
import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import imagehash
import numpy as np
import pyarrow.parquet as pq
import yaml
from PIL import Image, ImageOps

from bisight_rl.common import digest, file_hash, now, read_jsonl, write_json, write_jsonl
from bisight_rl.data import HammingTree, answer_kind, candidate_order, deduplicate_train, quotas, select_quota, stratum


def normalize(root, source):
    revision = source["dataset_revision"]
    cache = root / "normalized/manifest.json"
    identity = {"source_revision": revision, "normalization_version": 1,
                "Pillow": version("Pillow"), "ImageHash": version("ImageHash")}
    if cache.exists():
        previous = json.loads(cache.read_text())
        if previous["identity"] != identity:
            raise ValueError("Normalization identity changed; use a new data root")
        for split, info in previous["files"].items():
            if file_hash(root / info["path"]) != info["sha256"]:
                raise ValueError("Normalized data hash mismatch")
        return {s: read_jsonl(root / v["path"]) for s, v in previous["files"].items()}
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    all_rows, image_cache = {}, {}
    features = None
    files_info = {}
    for split in ("train", "val", "test"):
        rows = []
        files = sorted((e for e in source["files"] if e["name"].startswith(f"data/{split}-")), key=lambda e: e["name"])
        for entry in files:
            path = root / "raw" / revision / entry["name"]
            if file_hash(path) != entry["sha256"]:
                raise ValueError(f"Raw checksum mismatch: {path}")
            parquet = pq.ParquetFile(path)
            hf = json.loads(parquet.schema_arrow.metadata[b"huggingface"])
            current_features = hf["info"]["features"]
            if features is None:
                features = current_features
            if current_features != features:
                raise ValueError("Inconsistent features between shards")
            names = features["human_or_machine"]["names"]
            if names != ["human", "machine"]:
                raise ValueError(f"Unexpected class labels: {names}")
            for batch in parquet.iter_batches(batch_size=32):
                for original in batch.to_pylist():
                    index = len(rows)
                    errors = []
                    image_meta = {}
                    try:
                        raw_bytes = original["image"]["bytes"]
                        raw_hash = digest(raw_bytes)
                        if raw_hash not in image_cache:
                            with Image.open(io.BytesIO(raw_bytes)) as loaded:
                                im = ImageOps.exif_transpose(loaded).convert("RGBA")
                                white = Image.new("RGBA", im.size, "white")
                                im = Image.alpha_composite(white, im).convert("RGB")
                                pixel_hash = digest(f"RGB:{im.width}:{im.height}:".encode() + im.tobytes())
                                target = images / f"{pixel_hash}.png"
                                if not target.exists():
                                    im.save(target)
                                image_cache[raw_hash] = {
                                    "image_id": pixel_hash, "image_path": str(target.relative_to(root)),
                                    "image_file_sha256": file_hash(target), "image_original_sha256": raw_hash,
                                    "image_pixel_sha256": pixel_hash, "image_phash": str(imagehash.phash(im)),
                                    "image_width": im.width, "image_height": im.height,
                                }
                        image_meta = image_cache[raw_hash]
                    except (OSError, ValueError, TypeError, KeyError) as exc:
                        errors.append("unreadable_image:" + type(exc).__name__)
                        image_meta = {"image_id": f"invalid:{split}:{index}"}
                    question = original["query"]
                    answers = original["label"]
                    if not isinstance(question, str) or not question.strip():
                        errors.append("empty_question")
                    if not isinstance(answers, list) or not answers or not all(isinstance(a, str) and a.strip() for a in answers):
                        errors.append("invalid_answers")
                    elif len(answers) != 1:
                        errors.append("multiple_reference_semantics_need_review")
                    answer = answers[0] if isinstance(answers, list) and len(answers) == 1 and isinstance(answers[0], str) else None
                    row = {
                        "id": f"chartqa:{revision}:{split}:{index}", "split": split,
                        **image_meta, "question": question, "answers": answers,
                        "source": names[original["human_or_machine"]],
                        "answer_kind": answer_kind(answer) if answer else "unknown",
                        "answer_semantics": "single_reference" if answer else "unresolved",
                        "canonical_answer": answer, "data_errors": errors,
                        "provenance": {"dataset": source["dataset"], "revision": revision,
                                       "row_index": index, "shard": entry["name"],
                                       "original_image_path": original["image"].get("path")},
                    }
                    rows.append(row)
                    if len(rows) % 1000 == 0:
                        print(f"normalized {split}: {len(rows)}", flush=True)
        output = root / "normalized" / f"{split}.jsonl"
        write_jsonl(output, rows)
        files_info[split] = {"path": str(output.relative_to(root)), "sha256": file_hash(output), "count": len(rows)}
        all_rows[split] = rows
    write_json(cache, {"identity": identity, "features": features, "files": files_info})
    return all_rows


def audit(all_rows, cfg, root):
    radius = cfg["phash_distance"]
    thumbnails = {}
    def thumbnail(image_id):
        if image_id not in thumbnails:
            with Image.open(root / "images" / f"{image_id}.png") as image:
                size = cfg["near_pixel_size"]
                thumbnails[image_id] = np.asarray(image.resize((size, size), Image.Resampling.LANCZOS)).astype(np.int16)
        return thumbnails[image_id]
    images = {s: {r["image_id"]: r for r in rows if "image_phash" in r} for s, rows in all_rows.items()}
    exclusions = {"train": set(), "val": set()}
    pairs = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        tree = HammingTree()
        for image_id, row in images[right].items():
            tree.add(int(row["image_phash"], 16), image_id)
        for image_id, row in images[left].items():
            for distance, other in tree.query(int(row["image_phash"], 16), radius):
                exact = image_id == other
                if exact:
                    mae, changed = 0.0, 0.0
                else:
                    delta = np.abs(thumbnail(image_id) - thumbnail(other))
                    mae = float(delta.mean() / 255)
                    changed = float((delta.max(axis=-1) > cfg["near_channel_difference"]).mean())
                near = mae <= cfg["near_pixel_mae"] and changed <= cfg["near_changed_fraction"]
                status = "confirmed_pixel_duplicate" if exact else "unresolved_near_duplicate" if near else "phash_only_below_pixel_similarity_threshold"
                pairs.append({"left_split": left, "left_image_id": image_id,
                              "right_split": right, "right_image_id": other,
                              "distance": distance, "pixel_mae": mae, "changed_pixel_fraction": changed, "status": status})
                if exact or near:
                    exclusions[left].add(image_id)
    return pairs, exclusions


def stats(rows):
    return {"questions": len(rows), "charts": len({r["image_id"] for r in rows}),
            "strata": dict(sorted(Counter(stratum(r) for r in rows).items())),
            "source": dict(Counter(r["source"] for r in rows)),
            "max_questions_per_chart": max(Counter(r["image_id"] for r in rows).values(), default=0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--rebuild", action="store_true", help="Archive the previous build manifest before explicitly rebuilding candidates")
    args = parser.parse_args()
    root = args.data_root
    cfg = yaml.safe_load(args.config.read_text())
    source = json.loads((root / "manifests/source.json").read_text())
    if source["status"] != "downloaded":
        raise ValueError("Run download_data.py to completion first")
    build_path = root / "manifests/build.json"
    if build_path.exists():
        prior = json.loads(build_path.read_text())
        if not args.rebuild:
            identity_matches = (
                prior["config_sha256"] == file_hash(args.config)
                and prior["dataset_revision"] == source["dataset_revision"]
                and prior["preparation_script_sha256"] == file_hash(__file__)
                and prior["data_module_sha256"] == file_hash(Path(__file__).resolve().parents[1] / "src/bisight_rl/data.py")
            )
            if not identity_matches:
                raise ValueError("Frozen preparation code/config changed; use --rebuild to explicitly invalidate candidate/generation manifests")
            for relative, info in prior["artifacts"].items():
                if file_hash(root / relative) != info["sha256"]:
                    raise ValueError(f"Frozen artifact changed: {relative}")
            print("Existing frozen build verified; no files changed", flush=True)
            print(json.dumps(prior["summary"], ensure_ascii=False, indent=2))
            return
        write_json(root / "manifests/archive" / (file_hash(build_path) + ".json"), prior)
    all_rows = normalize(root, source)
    actual = {s: len(rows) for s, rows in all_rows.items()}
    if actual != cfg["expected_counts"]:
        raise ValueError(f"Unexpected split sizes: {actual}; inspect before changing expected_counts")
    print("auditing cross-split images", flush=True)
    pairs, exclusions = audit(all_rows, cfg, root)
    train, rejected = deduplicate_train(all_rows["train"], exclusions["train"])
    weights = Counter(stratum(r) for r in all_rows["train"])
    targets = quotas(weights, cfg["target_size"], Counter(stratum(r) for r in train))
    ordered = candidate_order(train, weights, cfg["seed"])
    for rank, row in enumerate(ordered):
        row["candidate_rank"] = rank
    initial = select_quota(ordered, targets)
    pilot_capacities = Counter(stratum(r) for r in initial)
    pilot_targets = {k: v + 1 for k, v in quotas(targets, cfg["pilot_size"] - len(pilot_capacities), {k: v - 1 for k, v in pilot_capacities.items()}).items()}
    pilot = select_quota(candidate_order(initial, targets, cfg["seed"] + 1), pilot_targets)
    dev = [r for r in all_rows["val"] if r["image_id"] not in exclusions["val"]]
    quick_pool = [r for r in dev if not r["data_errors"]]
    quick = candidate_order(quick_pool, Counter(stratum(r) for r in quick_pool), cfg["seed"] + 2)[:cfg["quick_size"]]
    if len(quick) != cfg["quick_size"]:
        raise ValueError("Insufficient dev_quick pool")
    outputs = {
        "candidates/train_candidates.jsonl": ordered,
        "candidates/train_initial_2000.jsonl": initial,
        "candidates/pilot_100.jsonl": pilot,
        "candidates/dev_full.jsonl": dev,
        "candidates/dev_quick.jsonl": quick,
        "candidates/test_full.jsonl": all_rows["test"],
        "reviews/cross_split_pairs.jsonl": pairs,
        "reviews/train_exclusions.jsonl": rejected,
        "reviews/data_errors.jsonl": [r for rows in all_rows.values() for r in rows if r["data_errors"]],
    }
    artifacts = {}
    for relative, rows in outputs.items():
        path = root / relative
        write_jsonl(path, rows)
        artifacts[relative] = {"sha256": file_hash(path), "rows": len(rows)}
    summary = {
        "raw": {s: stats(rows) for s, rows in all_rows.items()},
        "eligible_train": stats(train), "initial_2000": stats(initial), "pilot_100": stats(pilot),
        "dev_full": stats(dev), "dev_quick": stats(quick),
        "cross_split_pairs": dict(Counter(r["status"] for r in pairs)),
        "excluded_train_reasons": dict(Counter(r["reason"] for r in rejected)),
        "quarantined_image_counts": {s: len(ids) for s, ids in exclusions.items()},
        "target_quotas": targets,
    }
    build = {
        "status": "candidates_ready_not_rationale_master", "created_at": now(),
        "dataset_revision": source["dataset_revision"], "model_revision": source["model_revision"],
        "config_sha256": file_hash(args.config), "preparation_script_sha256": file_hash(__file__),
        "data_module_sha256": file_hash(Path(__file__).resolve().parents[1] / "src/bisight_rl/data.py"),
        "seed": cfg["seed"], "target_size": cfg["target_size"], "target_quotas": targets,
        "python": platform.python_version(),
        "dependencies": {p: version(p) for p in ("Pillow", "ImageHash", "numpy", "pyarrow")},
        "artifacts": artifacts, "summary": summary,
        "limitations": ["pHash + normalized pixel similarity candidates quarantined; thresholds do not prove all near duplicates are detected",
                        "No GPU generation or human rationale review performed",
                        "Latency subset awaits frozen processor input token counts"],
    }
    write_json(build_path, build)
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    (reports / "data_audit.md").write_text(
        "# ChartQA 数据审计\n\n状态：候选集已准备，尚无推理母版。\n\n"
        "近重复先由 pHash 召回，再以归一化像素差异复核；达到双阈值的未确认项保守隔离。未逐项人工判断所有近重复；阈值不能保证穷尽泄漏。原始 val/test 保留。\n\n"
        "```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
