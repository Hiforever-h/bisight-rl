"""Publish the 2,000-row master only after final human review passes."""
import argparse
import json
from collections import Counter
from pathlib import Path

from bisight_rl.common import digest, file_hash, now, read_jsonl, write_json, write_jsonl
from bisight_rl.data import stratum
from bisight_rl.quality import check_response
from review_rationales import review_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/rationales/v1"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/v1/rationale_master.jsonl"))
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    build = json.loads((args.data_root / "manifests/build.json").read_text())
    gate = json.loads((args.run_dir / "final_gate.json").read_text())
    reviews = json.loads((args.run_dir / "human_reviews.json").read_text())
    source = args.run_dir / "full_candidate_master.jsonl"
    if not gate["passed"] or gate["parent_sha256"] != file_hash(source) or gate["contract_sha256"] != digest(manifest["contract"]):
        raise ValueError("Final review gate is failed or stale")
    if gate["reviews_digest"] != review_digest(reviews, gate["review_keys"]):
        raise ValueError("Human review changed since the final gate")
    if manifest["contract"]["build_sha256"] != file_hash(args.data_root / "manifests/build.json"):
        raise ValueError("Data build changed")
    candidates_path = args.data_root / "candidates/train_candidates.jsonl"
    if file_hash(candidates_path) != build["artifacts"]["candidates/train_candidates.jsonl"]["sha256"]:
        raise ValueError("Candidate pool checksum changed")
    allowed = {r["id"]: r for r in read_jsonl(candidates_path)}
    rows = read_jsonl(source)
    if len(rows) != build["target_size"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Master must contain exactly the frozen number of unique questions")
    if dict(Counter(stratum(r) for r in rows)) != {k: v for k, v in build["target_quotas"].items() if v}:
        raise ValueError("Stratum quotas differ from frozen build")
    for row in rows:
        if row["id"] not in allowed or row["split"] != "train":
            raise ValueError("Master contains an unapproved candidate")
        for field in ("question", "answers", "canonical_answer", "image_id", "image_path", "image_file_sha256"):
            if row[field] != allowed[row["id"]][field]:
                raise ValueError(f"Candidate field changed: {field}")
        result = row["generation"]
        if result["contract_sha256"] != digest(manifest["contract"]) or result["response_sha256"] != digest([row["id"], result["attempt"], result["raw_response"]]):
            raise ValueError("Generation provenance mismatch")
        qc = check_response(result["raw_response"], row["canonical_answer"], result["finish_reason"])
        if not qc["auto_pass"] or qc["rationale"] != row["rationale"]:
            raise ValueError("Master contains a failed/altered explanation")
        review = reviews.get(result["response_sha256"])
        if review and review["decision"] != "pass":
            raise ValueError("Known rejected explanation remains in master")
        row["quality_status"] = "human_checked" if review else "auto_checked_in_sample_audited_release"
    write_jsonl(args.output, rows)
    write_json(args.output.with_suffix(".manifest.json"), {
        "status": "final_sample_audited_master", "created_at": now(), "rows": len(rows),
        "sha256": file_hash(args.output), "source_run_manifest": manifest,
        "human_checked_in_master": sum(r["quality_status"] == "human_checked" for r in rows),
        "final_review_gate": gate, "limitation": "Sample audit; not every explanation has been human verified",
    })
    print(f"Published {len(rows)} rows: {args.output}")


if __name__ == "__main__":
    main()
