"""Recompute and summarize the append-only GRPO rollout reward log."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from bisight_rl.common import file_hash, now, read_jsonl, write_json
from bisight_rl.grpo.reward import encode_ground_truth, score_response


def audit(rows, expected_prompts: int = 16, expected_rollouts: int = 4):
    if not rows:
        raise ValueError("Rollout audit is empty")
    by_batch = defaultdict(list)
    for row in rows:
        by_batch[int(row["reward_batch"])].append(row)
    batch_ids = sorted(by_batch)
    if batch_ids != list(range(1, batch_ids[-1] + 1)):
        raise ValueError(f"Reward batch ids are not contiguous: {batch_ids}")

    recompute_errors = []
    batch_reports = []
    all_group_rewards = defaultdict(list)
    for batch_id in batch_ids:
        batch = by_batch[batch_id]
        expected_rows = expected_prompts * expected_rollouts
        if len(batch) != expected_rows:
            raise ValueError(f"Reward batch {batch_id} has {len(batch)} rows, expected {expected_rows}")
        counts = Counter(row["prompt_id"] for row in batch)
        if len(counts) != expected_prompts or set(counts.values()) != {expected_rollouts}:
            raise ValueError(f"Reward batch {batch_id} group layout is not {expected_prompts}x{expected_rollouts}")
        if any(int(row["policy_version"]) != batch_id - 1 for row in batch):
            raise ValueError(f"Reward batch {batch_id} has inconsistent policy_version")
        for row in batch:
            truth = encode_ground_truth(
                row["prompt_id"],
                row["references"],
                source=row.get("source"),
                answer_kind=row.get("answer_kind"),
            )
            result = score_response(row["response"], truth)
            expected = (result["format"], result["accuracy"], result["overall"])
            observed = (row["format_reward"], row["answer_reward"], row["total_reward"])
            if expected != observed:
                recompute_errors.append({"batch": batch_id, "prompt_id": row["prompt_id"]})
            all_group_rewards[(batch_id, row["prompt_id"])].append(row["total_reward"])
        groups = [values for (bid, _), values in all_group_rewards.items() if bid == batch_id]
        batch_reports.append(
            {
                "reward_batch": batch_id,
                "policy_version": batch_id - 1,
                "format_rate": sum(row["format_reward"] for row in batch) / len(batch),
                "answer_accuracy": sum(row["answer_reward"] for row in batch) / len(batch),
                "nonempty_think_rate": sum(
                    bool(row["format_reward"]) and not row["think_empty"] for row in batch
                ) / len(batch),
                "informative_reward_groups": sum(len(set(values)) > 1 for values in groups),
                "answer_varied_groups": sum(
                    len({row["answer_reward"] for row in batch if row["prompt_id"] == prompt_id}) > 1
                    for prompt_id in counts
                ),
                "max_length_hits": sum(bool(row["hit_max_response_length"]) for row in batch),
            }
        )
    if recompute_errors:
        raise ValueError(f"Stored rewards disagree with current scorer: {recompute_errors[:10]}")
    group_values = list(all_group_rewards.values())
    return {
        "status": "passed",
        "created_at": now(),
        "sampling_iterations": len(batch_ids),
        "rollouts": len(rows),
        "unique_prompt_draws": len(group_values),
        "format_rate": sum(row["format_reward"] for row in rows) / len(rows),
        "answer_accuracy": sum(row["answer_reward"] for row in rows) / len(rows),
        "nonempty_think_rate": sum(
            bool(row["format_reward"]) and not row["think_empty"] for row in rows
        ) / len(rows),
        "informative_reward_group_rate": sum(len(set(values)) > 1 for values in group_values) / len(group_values),
        "max_length_hits": sum(bool(row["hit_max_response_length"]) for row in rows),
        "batches": batch_reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-prompts", type=int, default=16)
    parser.add_argument("--expected-rollouts", type=int, default=4)
    args = parser.parse_args()
    rows = read_jsonl(args.rollouts)
    report = audit(rows, args.expected_prompts, args.expected_rollouts)
    report["rollouts_path"] = str(args.rollouts)
    report["rollouts_sha256"] = file_hash(args.rollouts)
    output = args.output or args.rollouts.with_name("rollout_audit.json")
    write_json(output, report)
    printable = dict(report)
    printable.pop("batches")
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
