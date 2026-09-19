"""Export an image/text review page and import explicitly completed human CSV reviews."""
import argparse
import csv
import html
import json
import os
from pathlib import Path

from bisight_rl.common import digest, file_hash, now, read_jsonl, write_json, write_jsonl
from bisight_rl.data import stratum
from generate_rationales import get_candidate, load_attempts


def review_digest(reviews, keys):
    return digest({k: {f: reviews[k][f] for f in ("decision", "hint_leak")} for k in keys})


def export_review(args):
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    cfg = manifest["contract"]["generation_config"]
    selection_path = args.run_dir / f"{args.stage}_review_selection.jsonl"
    if args.stage == "pilot":
        rows = read_jsonl(args.data_root / "candidates/pilot_100.jsonl")
        selected = []
        for row in rows:
            attempts = load_attempts(args.run_dir, row, cfg["max_attempts"])
            if not attempts:
                raise ValueError("Complete pilot generation before exporting the review")
            result = get_candidate(attempts, {}) or attempts[-1]
            selected.append({**row, "generation": result})
        parent_hash = file_hash(args.data_root / "candidates/pilot_100.jsonl")
    else:
        parent = args.run_dir / "full_candidate_master.jsonl"
        rows = read_jsonl(parent)
        build = json.loads((args.data_root / "manifests/build.json").read_text())
        if len(rows) != build["target_size"]:
            raise ValueError("Full candidate master must have the frozen target count before final review")
        # Final review covers every generated training candidate. Sampling is
        # appropriate for the pilot readiness check, but it cannot record known
        # failures in the remaining rows before training data is published.
        for row in rows:
            tokens = row["generation"]["output_tokens"]
            row["review_stratum"] = stratum(row) + (":short" if tokens < 256 else ":medium" if tokens < 768 else ":long") + (":retry" if row["generation"]["attempt"] else ":first")
        selected = rows
        parent_hash = file_hash(parent)
    selection_meta = args.run_dir / f"{args.stage}_review_selection.json"
    if selection_path.exists():
        old_meta = json.loads(selection_meta.read_text())
        if old_meta["parent_sha256"] != parent_hash:
            raise ValueError("Candidate master changed. Archive the old review selection/CSV/HTML/meta before re-exporting.")
        frozen = read_jsonl(selection_path)
        if args.stage == "final" and len(frozen) != len(rows):
            raise ValueError("Final review selection must contain every candidate. Archive the old sampled review artifacts before re-exporting.")
        selected = frozen
    else:
        write_jsonl(selection_path, selected)
        write_json(selection_meta, {"parent_sha256": parent_hash, "selection_sha256": file_hash(selection_path),
                                    "contract_sha256": digest(manifest["contract"]), "count": len(selected)})
    csv_path = args.run_dir / f"{args.stage}_review.csv"
    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "response_sha256", "decision", "hint_leak", "reviewer", "notes"])
            writer.writeheader()
            writer.writerows({"sample_id": r["id"], "response_sha256": r["generation"]["response_sha256"]} for r in selected)
    decisions = {}
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            decisions = {r["sample_id"]: r for r in csv.DictReader(f)}
    passed = sum(r.get("decision") == "pass" for r in decisions.values())
    rejected = sum(r.get("decision") == "reject" for r in decisions.values())
    parts = ['<!doctype html><meta charset="utf-8"><title>ChartQA human review</title><style>body{font:16px system-ui;margin:24px;max-width:1400px}article{border-top:2px solid #777;margin:28px 0;padding:18px 0}article.pass{border-color:#16803c}article.reject{border-color:#c62828}.decision{font-weight:700}.pass .decision{color:#16803c}.reject .decision{color:#c62828}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}code{overflow-wrap:anywhere}</style>',
             '<h1>ChartQA 人工审查</h1><p>对照图片核查读数、图例、单位、计算、结论与答案提示残留。在对应 CSV 中填写 decision=pass/reject、hint_leak=yes/no、reviewer、notes。自动通过不等于视觉正确；不确定项使用 reject 并说明原因。</p>']
    parts.append(f'<p>共 {len(selected)} 条；已通过 {passed} 条；已判退 {rejected} 条。</p>')
    for index, row in enumerate(selected, 1):
        result = row["generation"]
        review = decisions.get(row["id"], {})
        decision = review.get("decision", "pending")
        note = review.get("notes", "")
        relative = os.path.relpath((args.data_root / row["image_path"]).resolve(), args.run_dir.resolve())
        parts.append(f'<article class="{html.escape(decision)}"><h2>{index}. {html.escape(row["question"])}</h2><code>{html.escape(row["id"])}</code><p class="decision">Decision: {html.escape(decision)}; hint leak: {html.escape(review.get("hint_leak", "pending"))}</p><p>Review note: {html.escape(note)}</p><p>Reference: {html.escape(row["canonical_answer"])}</p><div class="grid"><img loading="lazy" src="{html.escape(relative, quote=True)}"><div><pre>{html.escape(result["raw_response"])}</pre><p>Automatic errors: {html.escape(str(result["quality"]["errors"]))}</p></div></div></article>')
    (args.run_dir / f"{args.stage}_review.html").write_text("\n".join(parts))
    print(f"Review {len(selected)} rows: {csv_path}")


def import_review(args):
    selection_path = args.run_dir / f"{args.stage}_review_selection.jsonl"
    selected = read_jsonl(selection_path)
    meta = json.loads((args.run_dir / f"{args.stage}_review_selection.json").read_text())
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    if file_hash(selection_path) != meta["selection_sha256"] or digest(manifest["contract"]) != meta["contract_sha256"]:
        raise ValueError("Review selection hash or run contract mismatch")
    parent = args.data_root / "candidates/pilot_100.jsonl" if args.stage == "pilot" else args.run_dir / "full_candidate_master.jsonl"
    if file_hash(parent) != meta["parent_sha256"]:
        raise ValueError("Review no longer describes the current candidate set")
    expected = {r["id"]: r for r in selected}
    with (args.run_dir / f"{args.stage}_review.csv").open(newline="") as f:
        entries = list(csv.DictReader(f))
    if len(entries) != len(expected) or {e["sample_id"] for e in entries} != set(expected):
        raise ValueError("Review CSV must contain each selected sample exactly once")
    reviews_path = args.run_dir / "human_reviews.json"
    reviews = json.loads(reviews_path.read_text()) if reviews_path.exists() else {}
    new_reviews = {}
    for entry in entries:
        result = expected[entry["sample_id"]]["generation"]
        if entry["response_sha256"] != result["response_sha256"]:
            raise ValueError("Review response hash mismatch")
        if entry["decision"] not in {"pass", "reject"} or entry["hint_leak"] not in {"yes", "no"} or not entry["reviewer"].strip():
            raise ValueError("Complete decision, hint_leak and reviewer for every row")
        if entry["decision"] == "pass" and (not result["quality"]["auto_pass"] or entry["hint_leak"] == "yes"):
            raise ValueError("A failed automatic check or hint leakage cannot be marked pass")
        if entry["decision"] == "reject" and not entry["notes"].strip():
            raise ValueError("Rejected rows need a reason")
        new_reviews[entry["response_sha256"]] = {**entry, "reviewed_at": now()}
    reviews.update(new_reviews)
    write_json(reviews_path, reviews)
    errors = sum(e["decision"] == "reject" for e in entries)
    leaks = sum(e["hint_leak"] == "yes" for e in entries)
    accepted = len(entries) - errors
    accepted_leaks = sum(e["decision"] == "pass" and e["hint_leak"] == "yes" for e in entries)
    if args.stage == "pilot":
        # The pilot is a diagnostic/readiness checkpoint. Dataset label defects and
        # bad generations are expected to occur; their reviewed response hashes are
        # excluded by generate_rationales.get_candidate during the full run. Do not
        # turn their rate into a requirement that every source example be correct.
        passed = len(entries) >= 100 and accepted > 0 and accepted_leaks == 0
        policy = "review_complete_known_rejects_excluded"
    else:
        # Publication remains strict: every known rejected response must first be
        # removed/replaced, and the refreshed final sample must contain no leak.
        passed = len(entries) >= 100 and errors == 0 and leaks == 0
        policy = "final_sample_has_no_known_error_or_leak"
    gate = {"passed": passed, "stage": args.stage, "policy": policy,
            "reviewed_count": len(entries), "accepted_count": accepted, "excluded_count": errors,
            "error_count": errors, "leak_count": leaks, "accepted_leak_count": accepted_leaks,
            "pass_rate": accepted / len(entries) if entries else 0.0,
            "contract_sha256": digest(manifest["contract"]), "review_keys": sorted(new_reviews),
            "reviews_digest": review_digest(reviews, new_reviews), "parent_sha256": meta["parent_sha256"],
            "selection_sha256": file_hash(selection_path), "created_at": now()}
    write_json(args.run_dir / f"{args.stage}_gate.json", gate)
    print(json.dumps(gate, indent=2))
    if not passed:
        raise SystemExit("Review recorded but gate failed. Repair/retry known errors and re-review; do not publish.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["pilot", "final"])
    parser.add_argument("--action", required=True, choices=["export", "import"])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/rationales/v3"))
    args = parser.parse_args()
    (export_review if args.action == "export" else import_review)(args)


if __name__ == "__main__":
    main()
