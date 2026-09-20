"""Strict additive format and ChartQA answer rewards for EasyR1."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from bisight_rl.quality import (
    answer_matches_any_reference,
    extract_unique_answer,
    parse_response,
)


REWARD_NAME = "chartqa_format_plus_answer"
REWARD_TYPE = "batch"
GROUND_TRUTH_SCHEMA = "bisight-chartqa-reward-v1"
_AUDIT_BATCH_COUNTERS: dict[str, int] = {}


def encode_ground_truth(
    sample_id: str,
    references: list[str],
    *,
    source: str | None = None,
    answer_kind: str | None = None,
) -> str:
    """Serialize references without exposing them to the model prompt."""
    payload = {
        "schema": GROUND_TRUTH_SCHEMA,
        "id": sample_id,
        "references": references,
        "source": source,
        "answer_kind": answer_kind,
    }
    validate_ground_truth(payload)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def decode_ground_truth(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("GRPO ground truth is not valid JSON") from exc
    elif isinstance(value, dict):
        payload = value
    else:
        raise TypeError(f"GRPO ground truth must be a JSON string or mapping, got {type(value).__name__}")
    validate_ground_truth(payload)
    return payload


def validate_ground_truth(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or payload.get("schema") != GROUND_TRUTH_SCHEMA:
        raise ValueError(f"Expected ground-truth schema {GROUND_TRUTH_SCHEMA!r}")
    if not isinstance(payload.get("id"), str) or not payload["id"].strip():
        raise ValueError("Ground truth must contain a non-empty sample id")
    references = payload.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError(f"Ground truth for {payload['id']} has no references")
    if any(not isinstance(item, str) or not item.strip() for item in references):
        raise ValueError(f"Ground truth for {payload['id']} has an invalid reference")


def score_response(response: str, ground_truth: str | dict[str, Any]) -> dict[str, Any]:
    """Score the two components independently; diagnostics never affect reward."""
    if not isinstance(response, str):
        raise TypeError("Model response must be a string")
    payload = decode_ground_truth(ground_truth)
    parsed = parse_response(response)
    answer = extract_unique_answer(response)
    format_score = float(parsed is not None)
    accuracy_score = float(
        answer is not None
        and answer_matches_any_reference(payload["references"], answer, list_aware=False)
    )
    list_aware_score = float(
        answer is not None
        and answer_matches_any_reference(payload["references"], answer, list_aware=True)
    )
    return {
        "overall": format_score + accuracy_score,
        "format": format_score,
        "accuracy": accuracy_score,
        "list_aware_accuracy": list_aware_score,
        "answer_extractable": float(answer is not None),
        "think_empty": float(parsed is not None and not parsed["rationale"].strip()),
        "predicted_answer": answer,
        "ground_truth": payload,
    }


def _last_audit_batch(path: Path) -> int:
    if not path.exists():
        return 0
    last = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid reward audit JSON at {path}:{line_number}") from exc
            last = max(last, int(value.get("reward_batch", 0)))
    return last


def _next_audit_batch(path: Path) -> int:
    key = str(path.resolve())
    if key not in _AUDIT_BATCH_COUNTERS:
        _AUDIT_BATCH_COUNTERS[key] = _last_audit_batch(path)
    _AUDIT_BATCH_COUNTERS[key] += 1
    return _AUDIT_BATCH_COUNTERS[key]


def _append_audit(
    path: Path,
    reward_inputs: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    max_response_length: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reward_batch = _next_audit_batch(path)
    per_prompt_index: Counter[str] = Counter()
    with path.open("a", encoding="utf-8") as handle:
        for reward_input, result in zip(reward_inputs, scored, strict=True):
            payload = result["ground_truth"]
            sample_index = per_prompt_index[payload["id"]]
            per_prompt_index[payload["id"]] += 1
            response_length = int(reward_input["response_length"])
            record = {
                "reward_batch": reward_batch,
                "policy_version": reward_batch - 1,
                "prompt_id": payload["id"],
                "sample_index": sample_index,
                "source": payload.get("source"),
                "answer_kind": payload.get("answer_kind"),
                "references": payload["references"],
                "response": reward_input["response"],
                "response_length": response_length,
                "hit_max_response_length": bool(
                    max_response_length is not None and response_length >= max_response_length
                ),
                "predicted_answer": result["predicted_answer"],
                "format_reward": result["format"],
                "answer_reward": result["accuracy"],
                "total_reward": result["overall"],
                "list_aware_correct": bool(result["list_aware_accuracy"]),
                "think_empty": bool(result["think_empty"]),
                "group_reward_varies": bool(result["group_reward_varies"]),
                "group_answer_varies": bool(result["group_answer_varies"]),
                "group_any_nonempty_think": bool(result["group_any_nonempty_think"]),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_score(
    reward_inputs: list[dict[str, Any]],
    audit_path: str | None = None,
    max_response_length: int | None = None,
) -> list[dict[str, float]]:
    """EasyR1 batch callback returning only numeric metrics to the trainer."""
    scored = [score_response(item["response"], item["ground_truth"]) for item in reward_inputs]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in scored:
        grouped.setdefault(result["ground_truth"]["id"], []).append(result)
    for group in grouped.values():
        total_varies = float(len({item["overall"] for item in group}) > 1)
        answer_varies = float(len({item["accuracy"] for item in group}) > 1)
        any_nonempty = float(any(not bool(item["think_empty"]) and item["format"] for item in group))
        for result in group:
            result["group_reward_varies"] = total_varies
            result["group_answer_varies"] = answer_varies
            result["group_any_nonempty_think"] = any_nonempty
    if audit_path:
        _append_audit(Path(audit_path), reward_inputs, scored, max_response_length)
    metric_keys = (
        "overall",
        "format",
        "accuracy",
        "list_aware_accuracy",
        "answer_extractable",
        "think_empty",
        "group_reward_varies",
        "group_answer_varies",
        "group_any_nonempty_think",
    )
    return [{key: float(result[key]) for key in metric_keys} for result in scored]
