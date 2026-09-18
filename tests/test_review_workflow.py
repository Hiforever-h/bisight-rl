"""Exercise review gates and final publishing with synthetic (non-training) data."""
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from bisight_rl.common import digest, file_hash, write_json, write_jsonl
from bisight_rl.quality import check_response

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_no_publication_without_completed_human_review(tmp_path):
    root = tmp_path / "data"
    run = root / "rationales/v1"
    candidates = []
    for i in range(100):
        candidates.append({"id": f"test:{i}", "split": "train", "question": "What is the value?", "answers": ["7"],
                           "canonical_answer": "7", "image_id": f"image:{i}", "image_path": "images/example.png", "image_file_sha256": "placeholder",
                           "source": "human", "answer_kind": "numeric", "candidate_rank": i, "data_errors": []})
    candidate_path = root / "candidates/train_candidates.jsonl"
    write_jsonl(candidate_path, candidates)
    build_path = root / "manifests/build.json"
    write_json(build_path, {"target_size": 100, "target_quotas": {"human:numeric": 100},
                           "artifacts": {"candidates/train_candidates.jsonl": {"sha256": file_hash(candidate_path)}}})
    contract = {"build_sha256": file_hash(build_path), "generation_config": {"max_attempts": 3}}
    write_json(run / "manifest.json", {"contract": contract})
    rows = []
    for row in candidates:
        text = "<think>The bar label reads 7.</think><answer>7</answer>"
        generation = {"sample_id": row["id"], "attempt": 0, "raw_response": text, "finish_reason": "stop", "output_tokens": 20,
                      "contract_sha256": digest(contract), "response_sha256": digest([row["id"], 0, text]), "quality": check_response(text, "7", "stop")}
        rows.append({**row, "generation": generation, "rationale": generation["quality"]["rationale"]})
    write_jsonl(run / "full_candidate_master.jsonl", rows)
    def call(script, *args):
        return subprocess.run([sys.executable, str(SCRIPTS / script), "--data-root", str(root), "--run-dir", str(run), *args], cwd=tmp_path, capture_output=True, text=True)
    assert call("review_rationales.py", "--stage", "final", "--action", "export").returncode == 0
    assert call("review_rationales.py", "--stage", "final", "--action", "import").returncode != 0
    assert not (run / "final_gate.json").exists()
    csv_path = run / "final_review.csv"
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields, entries = reader.fieldnames, list(reader)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**e, "decision": "pass", "hint_leak": "no", "reviewer": "synthetic-test-fixture"} for e in entries)
    checked = call("review_rationales.py", "--stage", "final", "--action", "import")
    assert checked.returncode == 0, checked.stderr
    output = root / "processed/rationale_master.jsonl"
    published = call("finalize_master.py", "--output", str(output))
    assert published.returncode == 0, published.stderr
    assert len(output.read_text().splitlines()) == 100
    # Altering the reviewed master invalidates the gate.
    rows[0]["rationale"] = "Changed after review"
    write_jsonl(run / "full_candidate_master.jsonl", rows)
    assert call("finalize_master.py", "--output", str(output)).returncode != 0


def test_completed_pilot_can_proceed_with_recorded_rejects(tmp_path):
    root = tmp_path / "data"
    run = root / "rationales/v1"
    contract = {"generation_config": {"max_attempts": 3}}
    write_json(run / "manifest.json", {"contract": contract})
    candidates, selected, csv_rows = [], [], []
    for i in range(100):
        row = {"id": f"test:{i}", "split": "train", "question": "What is the value?", "answers": ["7"],
               "canonical_answer": "7", "image_id": f"image:{i}", "image_path": "images/example.png",
               "image_file_sha256": "placeholder", "source": "human", "answer_kind": "numeric",
               "candidate_rank": i, "data_errors": []}
        text = "<think>The bar label reads 7.</think><answer>7</answer>"
        generation = {"sample_id": row["id"], "attempt": 0, "raw_response": text, "finish_reason": "stop",
                      "output_tokens": 20, "contract_sha256": digest(contract),
                      "response_sha256": digest([row["id"], 0, text]), "quality": check_response(text, "7", "stop")}
        candidates.append(row)
        selected.append({**row, "generation": generation})
        rejected = i < 20
        csv_rows.append({"sample_id": row["id"], "response_sha256": generation["response_sha256"],
                         "decision": "reject" if rejected else "pass",
                         "hint_leak": "yes" if i == 0 else "no", "reviewer": "synthetic-test-fixture",
                         "notes": "Known bad response is excluded." if rejected else ""})
    parent = root / "candidates/pilot_100.jsonl"
    selection = run / "pilot_review_selection.jsonl"
    write_jsonl(parent, candidates)
    write_jsonl(selection, selected)
    write_json(run / "pilot_review_selection.json", {
        "parent_sha256": file_hash(parent), "selection_sha256": file_hash(selection),
        "contract_sha256": digest(contract), "count": 100,
    })
    with (run / "pilot_review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "response_sha256", "decision", "hint_leak", "reviewer", "notes"])
        writer.writeheader()
        writer.writerows(csv_rows)
    checked = subprocess.run(
        [sys.executable, str(SCRIPTS / "review_rationales.py"), "--data-root", str(root), "--run-dir", str(run),
         "--stage", "pilot", "--action", "import"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert checked.returncode == 0, checked.stderr
    gate = json.loads((run / "pilot_gate.json").read_text())
    assert gate["passed"] is True
    assert gate["accepted_count"] == 80
    assert gate["excluded_count"] == 20
    assert gate["leak_count"] == 1
    assert gate["accepted_leak_count"] == 0
