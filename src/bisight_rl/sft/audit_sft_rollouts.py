"""Audit whether an SFT policy still samples both empty and non-empty reasoning."""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

from bisight_rl.common import digest, file_hash, now, read_jsonl, write_json, write_jsonl
from bisight_rl.quality import answer_matches_any_reference, extract_unique_answer, parse_response
from bisight_rl.sft.common import encode_prompt_example, move_batch
from bisight_rl.sft.evaluate_sft import (
    OFFICIAL_SCORER_URL,
    load_config,
    resolve_adapter,
    validate_evaluation_rows,
)


def select_balanced_rows(rows, sample_size, seed):
    """Select a deterministic, near-equal number of rows from each source."""
    if sample_size <= 0 or sample_size > len(rows):
        raise ValueError("--sample-size must be within the evaluation dataset")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append(row)
    if len(grouped) < 2:
        raise ValueError("Rollout audit requires at least two source strata")

    rng = random.Random(seed)
    queues = {}
    for source in sorted(grouped):
        queues[source] = list(grouped[source])
        rng.shuffle(queues[source])

    selected = []
    offsets = Counter()
    sources = sorted(queues)
    while len(selected) < sample_size:
        made_progress = False
        for source in sources:
            offset = offsets[source]
            if offset < len(queues[source]):
                selected.append(queues[source][offset])
                offsets[source] += 1
                made_progress = True
                if len(selected) == sample_size:
                    break
        if not made_progress:
            raise ValueError("Unable to fill the requested stratified sample")
    return selected


def response_audit(row, response, generated_tokens, max_new_tokens, rollout_index, generation_seed):
    parsed = parse_response(response)
    prediction = extract_unique_answer(response)
    answer_correct = prediction is not None and answer_matches_any_reference(row["answers"], prediction)
    rationale = parsed["rationale"] if parsed is not None else None
    return {
        "rollout_index": rollout_index,
        "generation_seed": generation_seed,
        "response": response,
        "format_correct": parsed is not None,
        "think_empty": parsed is not None and not rationale.strip(),
        "think_nonempty": parsed is not None and bool(rationale.strip()),
        "think_characters": len(rationale.strip()) if rationale is not None else None,
        "predicted_answer": prediction,
        "answer_correct": bool(answer_correct),
        "format_reward": int(parsed is not None),
        "answer_reward": int(answer_correct),
        "total_reward": int(parsed is not None) + int(answer_correct),
        "generated_tokens": generated_tokens,
        "max_new_tokens_reached": generated_tokens >= max_new_tokens,
    }


def validate_existing_groups(groups, selected_rows, rollouts_per_question):
    if len(groups) > len(selected_rows):
        raise ValueError("Rollout output contains more groups than this audit contract")
    expected_ids = [row["id"] for row in selected_rows[: len(groups)]]
    actual_ids = [group.get("id") for group in groups]
    if actual_ids != expected_ids:
        raise ValueError("Existing rollout groups are not an exact prefix of the selected sample")
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("Existing rollout group IDs are not unique")
    for group in groups:
        rollouts = group.get("rollouts")
        if not isinstance(rollouts, list) or len(rollouts) != rollouts_per_question:
            raise ValueError(f"Incomplete rollout group for {group.get('id')}")
        if [rollout.get("rollout_index") for rollout in rollouts] != list(range(rollouts_per_question)):
            raise ValueError(f"Invalid rollout indexes for {group.get('id')}")


def rate_block(rollouts, key):
    count = sum(bool(rollout[key]) for rollout in rollouts)
    return {"rollouts": len(rollouts), "count": count, "rate": count / len(rollouts) if rollouts else None}


def accuracy_block(rollouts):
    correct = sum(bool(rollout["answer_correct"]) for rollout in rollouts)
    return {"rollouts": len(rollouts), "correct": correct, "accuracy": correct / len(rollouts) if rollouts else None}


def population_variance(values):
    if not values:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def summarize(groups, contract_digest):
    rollouts = [rollout for group in groups for rollout in group["rollouts"]]
    empty = [rollout for rollout in rollouts if rollout["think_empty"]]
    nonempty = [rollout for rollout in rollouts if rollout["think_nonempty"]]
    invalid = [rollout for rollout in rollouts if not rollout["format_correct"]]
    by_source = {}
    for source in sorted({group["source"] for group in groups}):
        selected = [rollout for group in groups if group["source"] == source for rollout in group["rollouts"]]
        by_source[source] = {
            "format": rate_block(selected, "format_correct"),
            "nonempty_think": rate_block(selected, "think_nonempty"),
            "answer_accuracy": accuracy_block(selected),
        }

    group_actions = []
    answer_variances = []
    total_reward_variances = []
    for group in groups:
        group_rollouts = group["rollouts"]
        nonempty_count = sum(rollout["think_nonempty"] for rollout in group_rollouts)
        invalid_count = sum(not rollout["format_correct"] for rollout in group_rollouts)
        if invalid_count:
            action = "contains_invalid_format"
        elif nonempty_count == 0:
            action = "all_empty"
        elif nonempty_count == len(group_rollouts):
            action = "all_nonempty"
        else:
            action = "mixed_empty_nonempty"
        group_actions.append(action)
        answer_variances.append(population_variance([rollout["answer_reward"] for rollout in group_rollouts]))
        total_reward_variances.append(population_variance([rollout["total_reward"] for rollout in group_rollouts]))

    action_counts = Counter(group_actions)
    groups_with_any_nonempty = sum(
        any(rollout["think_nonempty"] for rollout in group["rollouts"]) for group in groups
    )
    mean_answer_variance = sum(answer_variances) / len(answer_variances) if answer_variances else None
    mean_total_variance = sum(total_reward_variances) / len(total_reward_variances) if total_reward_variances else None
    nonempty_count = sum(rollout["think_nonempty"] for rollout in rollouts)
    return {
        "status": "complete",
        "completed_at": now(),
        "contract_digest": contract_digest,
        "questions": len(groups),
        "rollouts": len(rollouts),
        "format": rate_block(rollouts, "format_correct"),
        "nonempty_think": rate_block(rollouts, "think_nonempty"),
        "empty_think": rate_block(rollouts, "think_empty"),
        "answer_accuracy": accuracy_block(rollouts),
        "accuracy_by_reasoning_action": {
            "empty_think": accuracy_block(empty),
            "nonempty_think": accuracy_block(nonempty),
            "invalid_format": accuracy_block(invalid),
        },
        "by_source": by_source,
        "group_action_counts": dict(sorted(action_counts.items())),
        "groups_with_any_nonempty_think": {
            "questions": len(groups),
            "count": groups_with_any_nonempty,
            "rate": groups_with_any_nonempty / len(groups) if groups else None,
        },
        "zero_nonempty_one_sided_95pct_upper_rate": (
            1 - math.pow(0.05, 1 / len(rollouts)) if rollouts and nonempty_count == 0 else None
        ),
        "reward_diversity": {
            "groups_with_zero_answer_reward_variance": sum(value == 0 for value in answer_variances),
            "groups_with_zero_total_reward_variance": sum(value == 0 for value in total_reward_variances),
            "mean_answer_reward_variance": mean_answer_variance,
            "mean_total_reward_variance": mean_total_variance,
        },
        "answer_parse_failures": sum(rollout["predicted_answer"] is None for rollout in rollouts),
        "max_new_tokens_reached": sum(rollout["max_new_tokens_reached"] for rollout in rollouts),
    }


def generated_length(token_ids, eos_token_ids):
    eos = set(eos_token_ids if isinstance(eos_token_ids, (list, tuple, set)) else [eos_token_ids])
    for index, token_id in enumerate(token_ids.tolist(), 1):
        if token_id in eos:
            return index
    return int(token_ids.numel())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sft_drop50.yaml"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/candidates/dev_quick.jsonl"))
    parser.add_argument("--data-manifest", type=Path, default=Path("data/manifests/build.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--rollouts-per-question", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.rollouts_per_question < 2:
        raise ValueError("--rollouts-per-question must be at least 2 for a diversity audit")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")

    config = load_config(args.config)
    adapter_contract = resolve_adapter(args.adapter, config["model"])
    all_rows, data_manifest, artifact_key = validate_evaluation_rows(
        args.data, args.data_manifest, expected_split="val"
    )
    selected_rows = select_balanced_rows(all_rows, args.sample_size, args.seed)
    max_new_tokens = args.max_new_tokens or int(config["max_response_tokens"])
    if max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    prompt_path = Path(config["prompt"])
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    contract = {
        "formal": False,
        "purpose": "stochastic_reasoning_support_audit_before_grpo",
        "device": args.device,
        "model_loading": "pinned_base_plus_peft_adapter_without_merge",
        "config_sha256": file_hash(args.config),
        "model": config["model"],
        "model_revision": config["model_revision"],
        "adapter": adapter_contract,
        "data_artifact": artifact_key,
        "data_sha256": file_hash(args.data),
        "data_manifest_sha256": file_hash(args.data_manifest),
        "dataset_revision": data_manifest.get("dataset_revision"),
        "selection": {
            "method": "deterministic_round_robin_by_source_after_seeded_within_source_shuffle",
            "seed": args.seed,
            "sample_size": len(selected_rows),
            "source_counts": dict(sorted(Counter(row["source"] for row in selected_rows).items())),
            "answer_kind_counts": dict(
                sorted(Counter(row.get("answer_kind") for row in selected_rows).items())
            ),
            "sample_ids_digest": digest([row["id"] for row in selected_rows]),
        },
        "prompt_sha256": file_hash(prompt_path),
        "max_new_tokens": max_new_tokens,
        "generation": {
            "do_sample": True,
            "rollouts_per_question": args.rollouts_per_question,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "runtime_packages": {
            "torch": version("torch"),
            "transformers": version("transformers"),
            "peft": version("peft"),
        },
        "official_scorer": OFFICIAL_SCORER_URL,
        "answer_semantics": "unique_closed_answer_block; no raw-response fallback",
        "evaluator_sha256": file_hash(Path(__file__)),
        "quality_sha256": file_hash(Path(__file__).parents[1] / "quality.py"),
        "sft_common_sha256": file_hash(Path(__file__).with_name("common.py")),
    }
    contract_digest = digest(contract)
    manifest_path = args.output_dir / "run_manifest.json"
    sample_path = args.output_dir / "sample.jsonl"
    groups_path = args.output_dir / "rollout_groups.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    if args.resume:
        if not manifest_path.is_file():
            raise ValueError("--resume requires the original run_manifest.json")
        previous = json.loads(manifest_path.read_text())
        if previous.get("contract_digest") != contract_digest:
            raise ValueError("Rollout audit contract changed; refusing to mix outputs")
        if not sample_path.is_file() or file_hash(sample_path) != previous["sample_sha256"]:
            raise ValueError("Rollout audit sample changed")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(sample_path, selected_rows)
        write_json(
            manifest_path,
            {
                "created_at": now(),
                "contract_digest": contract_digest,
                "sample_sha256": file_hash(sample_path),
                "contract": contract,
            },
        )

    groups = read_jsonl(groups_path) if groups_path.is_file() else []
    validate_existing_groups(groups, selected_rows, args.rollouts_per_question)
    if len(groups) == len(selected_rows):
        metrics = summarize(groups, contract_digest)
        write_json(metrics_path, metrics)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return

    import torch
    from peft import PeftModel
    from tqdm.auto import tqdm
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Rollout audit requires exactly one visible CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Rollout audit requires a BF16-capable CUDA GPU")
        device_map, device = {"": 0}, torch.device("cuda:0")
    else:
        device_map, device = {"": "cpu"}, torch.device("cpu")

    processor = AutoProcessor.from_pretrained(
        config["model"],
        revision=config["model_revision"],
        min_pixels=config["min_pixels"],
        max_pixels=config["max_pixels"],
        size={"shortest_edge": config["min_pixels"], "longest_edge": config["max_pixels"]},
    )
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        config["model"],
        revision=config["model_revision"],
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation=config["attention"],
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    model.config.use_cache = True
    write_json(
        args.output_dir / "runtime.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "peft": version("peft"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        },
    )

    eos_token_ids = model.generation_config.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = processor.tokenizer.eos_token_id
    if eos_token_ids is None:
        raise ValueError("Model and tokenizer do not define an EOS token")

    nonempty_count = sum(
        rollout["think_nonempty"] for group in groups for rollout in group["rollouts"]
    )
    rollout_count = len(groups) * args.rollouts_per_question
    with groups_path.open("a", encoding="utf-8", buffering=1) as output_handle:
        with tqdm(
            total=len(selected_rows),
            initial=len(groups),
            desc="SFT stochastic rollout audit",
            unit="question",
            dynamic_ncols=True,
            mininterval=0.5,
        ) as progress:
            for question_index, row in enumerate(selected_rows[len(groups) :], start=len(groups)):
                generation_seed = args.seed * 1_000_003 + question_index
                torch.manual_seed(generation_seed)
                if args.device == "cuda":
                    torch.cuda.manual_seed_all(generation_seed)
                prompt_row = {**row, "image_sha256": row["image_file_sha256"]}
                inputs = encode_prompt_example(
                    processor,
                    prompt_row,
                    Path(config["data_root"]),
                    prompt,
                    verify_image_hash=True,
                )
                input_tokens = int(inputs["input_ids"].shape[-1])
                if input_tokens > int(config["max_input_tokens"]):
                    raise ValueError(
                        f"Evaluation prompt too long for {row['id']}: "
                        f"{input_tokens} > {config['max_input_tokens']}"
                    )
                inputs = move_batch(inputs, device)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        do_sample=True,
                        num_beams=1,
                        num_return_sequences=args.rollouts_per_question,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                    )
                rollout_results = []
                prompt_tokens = inputs["input_ids"].shape[-1]
                for rollout_index, sequence in enumerate(generated):
                    new_tokens = sequence[prompt_tokens:].cpu()
                    token_count = generated_length(new_tokens, eos_token_ids)
                    response = processor.tokenizer.decode(
                        new_tokens[:token_count],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ).strip()
                    rollout_results.append(
                        response_audit(
                            row,
                            response,
                            token_count,
                            max_new_tokens,
                            rollout_index,
                            generation_seed,
                        )
                    )
                group = {
                    "id": row["id"],
                    "source": row["source"],
                    "answer_kind": row.get("answer_kind"),
                    "question": row["question"],
                    "references": row["answers"],
                    "canonical_answer": row["canonical_answer"],
                    "input_tokens": input_tokens,
                    "evaluated_at": now(),
                    "rollouts": rollout_results,
                }
                output_handle.write(json.dumps(group, ensure_ascii=False) + "\n")
                groups.append(group)
                nonempty_count += sum(rollout["think_nonempty"] for rollout in rollout_results)
                rollout_count += len(rollout_results)
                progress.update(1)
                progress.set_postfix(
                    nonempty=f"{nonempty_count / rollout_count:.2%}",
                    any_nonempty=sum(
                        any(rollout["think_nonempty"] for rollout in item["rollouts"]) for item in groups
                    ),
                    refresh=False,
                )

    metrics = summarize(groups, contract_digest)
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
