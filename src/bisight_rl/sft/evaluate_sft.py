"""Evaluate a pinned Qwen3-VL base, optionally with an SFT adapter, on ChartQA."""
from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import yaml

from bisight_rl.common import digest, file_hash, now, read_jsonl, write_json
from bisight_rl.quality import (
    answer_matches_any_reference,
    extract_unique_answer,
    parse_list_answer,
)
from bisight_rl.sft.common import encode_prompt_example, move_batch
from bisight_rl.sft.merge_sft import directory_digest


OFFICIAL_SCORER_URL = (
    "https://github.com/google-research/pix2struct/blob/"
    "6fe25c1dc8151823ee3b479519d8d5948812fee4/pix2struct/metrics.py"
)


def load_config(path: Path):
    config = yaml.safe_load(path.read_text())
    required = {
        "data_root",
        "prompt",
        "model",
        "model_revision",
        "attention",
        "min_pixels",
        "max_pixels",
        "max_input_tokens",
        "max_response_tokens",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing evaluation config keys: {missing}")
    return config


def resolve_adapter(adapter: Path | None, base_model: str):
    if adapter is None:
        return None
    adapter = Path(adapter)
    config_path = adapter / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Not a PEFT adapter directory: {adapter}")
    adapter_config = json.loads(config_path.read_text())
    adapter_base = adapter_config.get("base_model_name_or_path")
    if adapter_base and adapter_base != base_model:
        raise ValueError(f"Adapter base model differs from evaluation config: {adapter_base!r} != {base_model!r}")
    return {
        "path": str(adapter.resolve()),
        "digest": directory_digest(adapter),
    }


def validate_evaluation_rows(data_path: Path, manifest_path: Path, expected_split="test"):
    manifest = json.loads(manifest_path.read_text())
    data_root = manifest_path.parent.parent.resolve()
    try:
        artifact_key = str(data_path.resolve().relative_to(data_root))
    except ValueError as exc:
        raise ValueError("Evaluation data must be covered by the build manifest") from exc
    artifact = manifest.get("artifacts", {}).get(artifact_key)
    if artifact is None:
        raise ValueError(f"Evaluation artifact is absent from build manifest: {artifact_key}")
    if file_hash(data_path) != artifact["sha256"]:
        raise ValueError("Evaluation data differs from the build manifest")

    rows = read_jsonl(data_path)
    if len(rows) != artifact["rows"]:
        raise ValueError("Evaluation row count differs from the build manifest")
    required = {
        "id",
        "split",
        "image_path",
        "image_file_sha256",
        "question",
        "answers",
        "canonical_answer",
        "source",
        "data_errors",
    }
    ids = set()
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Evaluation row {index} is missing fields: {missing}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate evaluation ID: {row['id']}")
        ids.add(row["id"])
        if row["split"] != expected_split:
            raise ValueError(
                f"Unexpected split in evaluation data for {row['id']}: "
                f"expected {expected_split!r}, got {row['split']!r}"
            )
        if row["data_errors"]:
            raise ValueError(f"Quarantined row in evaluation data: {row['id']}")
        if not isinstance(row["answers"], list) or not row["answers"] or not all(
            isinstance(answer, str) and answer.strip() for answer in row["answers"]
        ):
            raise ValueError(f"Invalid answer references for {row['id']}")
        if row["canonical_answer"] not in row["answers"]:
            raise ValueError(f"Canonical answer is not an evaluation reference for {row['id']}")
    return rows, manifest, artifact_key


def validate_existing_predictions(predictions, rows):
    if len(predictions) > len(rows):
        raise ValueError("Predictions contain more rows than this evaluation contract")
    expected = [row["id"] for row in rows[: len(predictions)]]
    actual = [row.get("id") for row in predictions]
    if actual != expected:
        raise ValueError("Existing predictions are not an exact prefix of the evaluation set")
    if len(set(actual)) != len(actual):
        raise ValueError("Existing prediction IDs are not unique")
    for prediction in predictions:
        if not isinstance(prediction.get("official_correct"), bool):
            raise ValueError(f"Invalid official score in existing prediction: {prediction.get('id')}")
        if not isinstance(prediction.get("list_aware_correct"), bool):
            raise ValueError(f"Invalid list-aware score in existing prediction: {prediction.get('id')}")


def accuracy_block(rows, key):
    correct = sum(bool(row[key]) for row in rows)
    return {
        "samples": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
    }


def summarize(predictions, contract_digest):
    sources = sorted({row["source"] for row in predictions})
    list_rows = [row for row in predictions if row["list_valued_reference"]]
    by_source = {}
    for source in sources:
        selected = [row for row in predictions if row["source"] == source]
        by_source[source] = {
            "official_relaxed": accuracy_block(selected, "official_correct"),
            "list_aware_relaxed": accuracy_block(selected, "list_aware_correct"),
        }
    return {
        "status": "complete",
        "completed_at": now(),
        "contract_digest": contract_digest,
        "primary_metric": "official_relaxed_accuracy",
        "official_relaxed_accuracy": accuracy_block(predictions, "official_correct"),
        "list_aware_relaxed_accuracy": accuracy_block(predictions, "list_aware_correct"),
        "by_source": by_source,
        "list_valued_references": {
            "official_relaxed": accuracy_block(list_rows, "official_correct"),
            "list_aware_relaxed": accuracy_block(list_rows, "list_aware_correct"),
        },
        "answer_parse_failures": sum(row["predicted_answer"] is None for row in predictions),
        "max_new_tokens_reached": sum(row["max_new_tokens_reached"] for row in predictions),
        "official_vs_list_aware_disagreements": sum(
            row["official_correct"] != row["list_aware_correct"] for row in predictions
        ),
    }


def evaluate_response(row, response, generated_tokens, max_new_tokens):
    prediction = extract_unique_answer(response)
    official_correct = prediction is not None and answer_matches_any_reference(
        row["answers"], prediction, list_aware=False
    )
    list_aware_correct = prediction is not None and answer_matches_any_reference(
        row["answers"], prediction, list_aware=True
    )
    return {
        "id": row["id"],
        "source": row["source"],
        "answer_kind": row.get("answer_kind"),
        "question": row["question"],
        "references": row["answers"],
        "canonical_answer": row["canonical_answer"],
        "response": response,
        "predicted_answer": prediction,
        "official_correct": bool(official_correct),
        "list_aware_correct": bool(list_aware_correct),
        "list_valued_reference": any(parse_list_answer(answer) is not None for answer in row["answers"]),
        "generated_tokens": generated_tokens,
        "max_new_tokens_reached": generated_tokens >= max_new_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sft_drop50.yaml"))
    parser.add_argument("--adapter", type=Path, help="Optional PEFT adapter; omit to evaluate the base model")
    parser.add_argument("--data", type=Path, default=Path("data/candidates/test_full.jsonl"))
    parser.add_argument("--data-manifest", type=Path, default=Path("data/manifests/build.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int, help="Evaluate the first N rows for a non-formal smoke run")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    adapter_contract = resolve_adapter(args.adapter, config["model"])

    all_rows, data_manifest, artifact_key = validate_evaluation_rows(args.data, args.data_manifest)
    rows = all_rows
    if args.limit is not None:
        if args.limit <= 0 or args.limit > len(rows):
            raise ValueError("--limit must be within the evaluation dataset")
        rows = rows[: args.limit]
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else int(config["max_response_tokens"])
    if max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    prompt_path = Path(config["prompt"])
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    reference_audit = {
        "outer_reference_counts": {
            str(count): sum(len(row["answers"]) == count for row in rows)
            for count in sorted({len(row["answers"]) for row in rows})
        },
        "list_valued_rows": sum(
            any(parse_list_answer(answer) is not None for answer in row["answers"]) for row in rows
        ),
    }

    runtime_packages = {
        "torch": version("torch"),
        "transformers": version("transformers"),
    }
    if adapter_contract is not None:
        runtime_packages["peft"] = version("peft")
    contract = {
        "formal": args.limit is None,
        "device": args.device,
        "model_loading": "pinned_base_only" if adapter_contract is None else "pinned_base_plus_peft_adapter_without_merge",
        "config_sha256": file_hash(args.config),
        "model": config["model"],
        "model_revision": config["model_revision"],
        "adapter": adapter_contract,
        "data_artifact": artifact_key,
        "data_sha256": file_hash(args.data),
        "data_manifest_sha256": file_hash(args.data_manifest),
        "dataset_revision": data_manifest.get("dataset_revision"),
        "sample_ids_digest": digest([row["id"] for row in rows]),
        "reference_audit": reference_audit,
        "prompt_sha256": file_hash(prompt_path),
        "max_new_tokens": max_new_tokens,
        "generation": {"do_sample": False, "num_beams": 1},
        "runtime_packages": runtime_packages,
        "official_scorer": OFFICIAL_SCORER_URL,
        "list_semantics": "ordered_items; representation-normalized; itemwise relaxed correctness",
        "evaluator_sha256": file_hash(Path(__file__)),
        "quality_sha256": file_hash(Path(__file__).parents[1] / "quality.py"),
        "sft_common_sha256": file_hash(Path(__file__).with_name("common.py")),
    }
    contract_digest = digest(contract)
    manifest_path = args.output_dir / "run_manifest.json"
    predictions_path = args.output_dir / "predictions.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    if args.resume:
        if not manifest_path.is_file():
            raise ValueError("--resume requires the original run_manifest.json")
        previous = json.loads(manifest_path.read_text())
        if previous.get("contract_digest") != contract_digest:
            raise ValueError("Evaluation contract changed; refusing to mix predictions")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            manifest_path,
            {
                "created_at": now(),
                "contract_digest": contract_digest,
                "contract": contract,
            },
        )
    predictions = read_jsonl(predictions_path) if predictions_path.is_file() else []
    validate_existing_predictions(predictions, rows)
    if len(predictions) == len(rows):
        metrics = summarize(predictions, contract_digest)
        write_json(metrics_path, metrics)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return

    import torch
    from tqdm.auto import tqdm
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("SFT evaluation requires exactly one visible CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("SFT evaluation requires a BF16-capable CUDA GPU")
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
    if args.adapter is None:
        model = base.eval()
    else:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    model.config.use_cache = True
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "peft": version("peft") if args.adapter is not None else None,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
    }
    write_json(args.output_dir / "runtime.json", runtime)

    official_correct = sum(row["official_correct"] for row in predictions)
    list_correct = sum(row["list_aware_correct"] for row in predictions)
    with predictions_path.open("a", encoding="utf-8", buffering=1) as output_handle:
        with tqdm(
            total=len(rows),
            initial=len(predictions),
            desc="SFT answer evaluation",
            unit="sample",
            dynamic_ncols=True,
            mininterval=0.5,
        ) as progress:
            for row in rows[len(predictions) :]:
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
                        f"Evaluation prompt too long for {row['id']}: {input_tokens} > {config['max_input_tokens']}"
                    )
                inputs = move_batch(inputs, device)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                    )
                new_tokens = generated[0, inputs["input_ids"].shape[-1] :].cpu()
                response = processor.tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                result = evaluate_response(row, response, int(new_tokens.numel()), max_new_tokens)
                result["input_tokens"] = input_tokens
                result["evaluated_at"] = now()
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                predictions.append(result)
                official_correct += int(result["official_correct"])
                list_correct += int(result["list_aware_correct"])
                progress.update(1)
                progress.set_postfix(
                    official=f"{official_correct / len(predictions):.2%}",
                    list_aware=f"{list_correct / len(predictions):.2%}",
                    refresh=False,
                )

    metrics = summarize(predictions, contract_digest)
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
