"""Merge a trained SFT LoRA into the pinned BF16 Qwen3-VL base model."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import yaml

from bisight_rl.common import digest, file_hash, now, write_json
from bisight_rl.sft.common import encode_prompt_example, encode_training_example, move_batch, validate_compiled_dataset


def directory_digest(path: Path):
    entries = []
    for item in sorted(Path(path).rglob("*")):
        if item.is_file():
            entries.append((str(item.relative_to(path)), file_hash(item)))
    if not entries:
        raise ValueError(f"Adapter directory has no files: {path}")
    return digest(entries)


def validation_rows(rows, count: int):
    if count < 2:
        raise ValueError("validation sample count must be at least 2")
    empty = [row for row in rows if row["think_empty"]]
    nonempty = [row for row in rows if not row["think_empty"]]
    left = count // 2
    selected = empty[:left] + nonempty[: count - left]
    if len(selected) != count:
        raise ValueError("Not enough empty/nonempty rows for merge validation")
    return selected


def capture_outputs(model, processor, rows, config, prompt, device, logit_positions, generation_tokens):
    import torch

    captures = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            full, stats = encode_training_example(
                processor,
                row,
                Path(config["data_root"]),
                prompt,
                config["max_input_tokens"],
                config["max_response_tokens"],
                config["max_total_tokens"],
                verify_image_hash=True,
            )
            full = move_batch(full, device)
            labels = full["labels"][0]
            supervised = (labels != -100).nonzero(as_tuple=False).flatten()
            prediction_positions = (supervised - 1).clamp_min(0)
            if prediction_positions.numel() > logit_positions:
                indices = torch.linspace(0, prediction_positions.numel() - 1, logit_positions, device=device).long()
                prediction_positions = prediction_positions[indices]
            output = model(**full, use_cache=False)
            sampled_logits = output.logits[0, prediction_positions].float().cpu()
            prompt_inputs = move_batch(
                encode_prompt_example(processor, row, Path(config["data_root"]), prompt, verify_image_hash=True),
                device,
            )
            generated = model.generate(
                **prompt_inputs,
                do_sample=False,
                max_new_tokens=generation_tokens,
                use_cache=True,
            )[0, prompt_inputs["input_ids"].shape[-1] :].cpu()
            captures.append(
                {
                    "id": row["id"],
                    "stats": stats,
                    "logits": sampled_logits,
                    "argmax": sampled_logits.argmax(dim=-1),
                    "generated": generated,
                }
            )
    return captures


def compare_captures(before, after, max_abs_tolerance: float):
    import torch

    diagnostics, overall = [], 0.0
    if [item["id"] for item in before] != [item["id"] for item in after]:
        raise ValueError("Merge validation sample order changed")
    for left, right in zip(before, after):
        difference = float((left["logits"] - right["logits"]).abs().max())
        overall = max(overall, difference)
        argmax_equal = bool(torch.equal(left["argmax"], right["argmax"]))
        generation_equal = bool(torch.equal(left["generated"], right["generated"]))
        diagnostics.append(
            {
                "id": left["id"],
                "max_abs_logit_difference": difference,
                "sampled_argmax_equal": argmax_equal,
                "greedy_generation_equal": generation_equal,
            }
        )
        if not argmax_equal or not generation_equal:
            raise ValueError(f"Merged model changes predictions for {left['id']}: {diagnostics[-1]}")
    if overall > max_abs_tolerance:
        raise ValueError(f"Merged logit difference {overall} exceeds tolerance {max_abs_tolerance}")
    return {"max_abs_logit_difference": overall, "tolerance": max_abs_tolerance, "samples": diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sft_drop50.yaml"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--validation-samples", type=int, default=4)
    parser.add_argument("--logit-positions", type=int, default=8)
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--max-abs-logit-difference", type=float, default=0.5)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite merge output: {args.output}")
    if not (args.adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Not a PEFT adapter directory: {args.adapter}")
    config = yaml.safe_load(args.config.read_text())
    adapter_config = json.loads((args.adapter / "adapter_config.json").read_text())
    adapter_base = adapter_config.get("base_model_name_or_path")
    if adapter_base and adapter_base != config["model"]:
        raise ValueError(f"Adapter base model differs from merge config: {adapter_base!r} != {config['model']!r}")
    data_path, data_manifest_path, prompt_path = Path(config["data"]), Path(config["data_manifest"]), Path(config["prompt"])
    rows, data_manifest = validate_compiled_dataset(data_path, data_manifest_path, prompt_path)
    for key in ("model", "model_revision"):
        frozen = data_manifest["contract"].get(key)
        if frozen is not None and frozen != config[key]:
            raise ValueError(f"Merge config {key} differs from compiled data manifest")
    selected = validation_rows(rows, args.validation_samples)
    prompt = prompt_path.read_text(encoding="utf-8").strip()

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("CUDA merge validation requires exactly one visible GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA merge validation requires BF16 support")
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
    adapted = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    before = capture_outputs(adapted, processor, selected, config, prompt, device, args.logit_positions, args.generation_tokens)
    merged = adapted.merge_and_unload(safe_merge=True).eval()
    if any("lora_" in name.lower() for name, _ in merged.named_parameters()):
        raise ValueError("LoRA parameters remain after merge")
    after = capture_outputs(merged, processor, selected, config, prompt, device, args.logit_positions, args.generation_tokens)
    comparison = compare_captures(before, after, args.max_abs_logit_difference)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{args.output.name}-", dir=args.output.parent))
    merged.save_pretrained(temp, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(temp)
    if (temp / "adapter_config.json").exists():
        raise ValueError("Merged output unexpectedly contains adapter_config.json")
    manifest = {
        "status": "merged_and_validated",
        "created_at": now(),
        "base_model": config["model"],
        "base_revision": config["model_revision"],
        "adapter_path": str(args.adapter),
        "adapter_digest": directory_digest(args.adapter),
        "data_sha256": file_hash(data_path),
        "prompt_sha256": file_hash(prompt_path),
        "comparison": comparison,
        "merge_script_sha256": file_hash(Path(__file__)),
        "sft_common_sha256": file_hash(Path(__file__).with_name("common.py")),
        "config_sha256": file_hash(args.config),
    }
    write_json(temp / "merge_manifest.json", manifest)
    os.replace(temp, args.output)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
