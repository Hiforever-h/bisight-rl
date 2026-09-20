"""Fail-closed data, trainer, prompt-template, and batch-semantics checks for GRPO."""
from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from bisight_rl.common import file_hash, now, write_json
from bisight_rl.grpo.common import (
    disk_report,
    load_config,
    validate_artifacts,
    validate_config,
    validate_easyr1_checkout,
)
from bisight_rl.sft.common import encode_prompt_example


def _path(value) -> Path:
    return Path(value).expanduser().resolve()


def package_versions() -> dict[str, str | None]:
    result = {}
    for package in ("torch", "transformers", "peft", "vllm", "ray", "omegaconf", "wandb", "torchdata"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def validate_merged_model(model_path: Path, prompt_path: Path) -> dict:
    model_path = _path(model_path)
    required = ("config.json", "merge_manifest.json")
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Merged SFT model is missing {missing}: {model_path}")
    if (model_path / "adapter_config.json").exists():
        raise ValueError("GRPO model_path is still a PEFT adapter; use the validated merged SFT model")
    manifest = json.loads((model_path / "merge_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "merged_and_validated":
        raise ValueError("Merged SFT manifest is not validated")
    if manifest.get("prompt_sha256") != file_hash(prompt_path):
        raise ValueError("Merged SFT model used a different adaptive prompt")
    model_files = sorted(path.name for path in model_path.glob("*.safetensors"))
    if not model_files:
        raise FileNotFoundError(f"No safetensors model shards found in {model_path}")
    return {
        "path": str(model_path),
        "base_model": manifest.get("base_model"),
        "base_revision": manifest.get("base_revision"),
        "adapter_digest": manifest.get("adapter_digest"),
        "model_files": model_files,
    }


def processor_preflight(config: dict, artifacts: dict, model_path: Path):
    """Compare EasyR1's real encoded prompt with the SFT encoding path."""
    from tqdm.auto import tqdm
    from transformers import AutoProcessor

    from bisight_rl.grpo.easyr1_bridge import make_chartqa_dataset_class

    data = config["data"]
    prompt = _path(config["bisight"]["prompt"]).read_text(encoding="utf-8").strip()
    data_root = _path(data["image_dir"])
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=data["min_pixels"],
        max_pixels=data["max_pixels"],
        size={"shortest_edge": data["min_pixels"], "longest_edge": data["max_pixels"]},
    )
    DatasetClass = make_chartqa_dataset_class()
    records = []
    for split, data_path in (("train", data["train_files"]), ("dev_quick", data["val_files"])):
        dataset = DatasetClass(
            data_path=str(_path(data_path)),
            tokenizer=processor.tokenizer,
            processor=processor,
            prompt_key=data["prompt_key"],
            answer_key=data["answer_key"],
            image_key=data["image_key"],
            video_key=data["video_key"],
            image_dir=str(data_root),
            video_fps=data["video_fps"],
            max_prompt_length=data["max_prompt_length"],
            truncation="right",
            format_prompt=None,
            min_pixels=data["min_pixels"],
            max_pixels=data["max_pixels"],
            filter_overlong_prompts=False,
            filter_overlong_prompts_workers=data["filter_overlong_prompts_workers"],
            system_prompt=prompt,
        )
        rows = artifacts[split]
        if len(dataset) != len(rows):
            raise ValueError(f"EasyR1 {split} dataset length changed")
        with tqdm(range(len(dataset)), desc=f"GRPO {split} processor preflight", unit="sample", dynamic_ncols=True) as bar:
            for index in bar:
                row = rows[index]
                expected = encode_prompt_example(
                    processor,
                    {
                        "id": row["prompt_id"],
                        "image_path": row["images"][0],
                        "image_sha256": row["image_sha256"],
                        "question": row["problem"],
                    },
                    data_root,
                    prompt,
                    verify_image_hash=True,
                )
                actual = dataset[index]
                actual_ids = actual["input_ids"][actual["attention_mask"].bool()]
                expected_ids = expected["input_ids"][0]
                if actual_ids.shape != expected_ids.shape or not actual_ids.equal(expected_ids):
                    raise ValueError(
                        f"EasyR1/SFT prompt token mismatch for {row['prompt_id']}: "
                        f"easy={actual_ids.numel()}, sft={expected_ids.numel()}"
                    )
                token_count = int(actual_ids.numel())
                if token_count > data["max_prompt_length"]:
                    raise ValueError(
                        f"Prompt too long for {row['prompt_id']}: {token_count} > {data['max_prompt_length']}"
                    )
                image_token = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                visual_tokens = int((actual_ids == image_token).sum().item())
                if visual_tokens <= 0:
                    raise ValueError(f"No visual tokens for {row['prompt_id']}")
                records.append(
                    {
                        "id": row["prompt_id"],
                        "split": split,
                        "prompt_tokens": token_count,
                        "visual_tokens": visual_tokens,
                    }
                )
                bar.set_postfix(tokens=token_count, visual=visual_tokens, refresh=False)
    return {
        "samples": len(records),
        "prompt_tokens": {
            "min": min(item["prompt_tokens"] for item in records),
            "max": max(item["prompt_tokens"] for item in records),
        },
        "visual_tokens": {
            "min": min(item["visual_tokens"] for item in records),
            "max": max(item["visual_tokens"] for item in records),
        },
        "records": records,
    }


def run_preflight(config: dict, *, processor_check: bool = True, allow_dirty_easyr1: bool = False):
    project = config["bisight"]
    data = config["data"]
    batch = validate_config(config)
    easy_root = _path(project["easy_r1_root"])
    easy = validate_easyr1_checkout(easy_root, project["easy_r1_commit"], allow_dirty_easyr1)
    if str(easy_root) not in sys.path:
        sys.path.insert(0, str(easy_root))
    artifacts, manifest = validate_artifacts(
        _path(project["data_manifest"]),
        _path(data["train_files"]),
        _path(data["val_files"]),
        _path(project["prompt"]),
    )
    model = validate_merged_model(_path(config["worker"]["actor"]["model"]["model_path"]), _path(project["prompt"]))
    disk = disk_report(_path(project["output_root"]), float(project["minimum_free_disk_gb"]))
    processor = processor_preflight(config, artifacts, Path(model["path"])) if processor_check else {"status": "skipped"}
    return {
        "status": "passed",
        "created_at": now(),
        "easyr1": easy,
        "model": model,
        "data": {
            "manifest": str(_path(project["data_manifest"])),
            "manifest_sha256": file_hash(_path(project["data_manifest"])),
            "train_rows": len(artifacts["train"]),
            "dev_rows": len(artifacts["dev_quick"]),
        },
        "batch_contract": batch,
        "checkpoint_policy": {
            "save_frequency_sampling_iterations": config["trainer"]["save_freq"],
            "save_limit": config["trainer"]["save_limit"],
            "save_model_only": config["trainer"]["save_model_only"],
        },
        "packages": package_versions(),
        "disk": disk,
        "processor": processor,
        "data_manifest": manifest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/grpo_drop50.yaml"))
    parser.add_argument("--model-path", type=Path, help="Override worker.actor.model.model_path")
    parser.add_argument("--output", type=Path, default=Path("reports/grpo_preflight.json"))
    parser.add_argument("--skip-processor-check", action="store_true")
    parser.add_argument("--allow-dirty-easyr1", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.model_path:
        config["worker"]["actor"]["model"]["model_path"] = str(args.model_path)
    report = run_preflight(
        config,
        processor_check=not args.skip_processor_check,
        allow_dirty_easyr1=args.allow_dirty_easyr1,
    )
    write_json(args.output, report)
    printable = dict(report)
    printable["processor"] = {key: value for key, value in report["processor"].items() if key != "records"}
    printable.pop("data_manifest")
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
