"""Train language-only LoRA SFT for Qwen3-VL with audited assistant masks."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import yaml

from bisight_rl.common import digest, file_hash, now, write_json
from bisight_rl.sft.common import encode_training_example, move_batch, validate_compiled_dataset


LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
FORBIDDEN_MODULE_MARKERS = ("visual", "vision", "merger", "connector", "embed", "lm_head")


def load_config(path: Path):
    config = yaml.safe_load(path.read_text())
    required = {
        "data_root", "data", "data_manifest", "prompt", "output_root", "model", "model_revision",
        "min_pixels", "max_pixels", "max_input_tokens", "max_response_tokens", "max_total_tokens",
        "attention", "epochs", "learning_rate", "warmup_ratio", "weight_decay", "max_grad_norm",
        "gradient_accumulation_steps", "save_steps", "log_steps", "lora_rank", "lora_alpha",
        "lora_dropout", "language_module_marker",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing SFT config keys: {missing}")
    if config["gradient_accumulation_steps"] <= 0 or config["epochs"] <= 0:
        raise ValueError("epochs and gradient_accumulation_steps must be positive")
    if config["save_steps"] <= 0 or config["log_steps"] <= 0:
        raise ValueError("save_steps and log_steps must be positive")
    if not 0 <= float(config["warmup_ratio"]) < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if config["lora_rank"] <= 0 or config["lora_alpha"] <= 0 or config["lora_dropout"] < 0:
        raise ValueError("Invalid LoRA configuration")
    return config


def discover_lora_targets(model, language_marker: str):
    import torch

    targets = []
    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or not name.endswith(LORA_SUFFIXES):
            continue
        candidates.append(name)
        lowered = name.lower()
        if language_marker not in name or any(marker in lowered for marker in FORBIDDEN_MODULE_MARKERS):
            continue
        targets.append(name)
    if not targets:
        preview = "\n".join(candidates[:40])
        raise ValueError(f"No language LoRA targets found with marker {language_marker!r}. Candidates:\n{preview}")
    return sorted(targets)


def audit_trainable_parameters(model, expected_targets):
    trainable, total = [], 0
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable.append((name, parameter.numel()))
    if not trainable:
        raise ValueError("No trainable parameters after LoRA injection")
    bad = [name for name, _ in trainable if "lora_" not in name.lower() or any(marker in name.lower() for marker in FORBIDDEN_MODULE_MARKERS)]
    if bad:
        raise ValueError(f"Unexpected trainable non-language/LoRA parameters: {bad[:20]}")
    matched = {target for target in expected_targets if any(target in name for name, _ in trainable)}
    if matched != set(expected_targets):
        missing = sorted(set(expected_targets) - matched)
        raise ValueError(f"Some requested LoRA modules have no trainable adapter parameters: {missing[:20]}")
    return {
        "trainable_parameters": sum(size for _, size in trainable),
        "total_parameters": total,
        "fraction": sum(size for _, size in trainable) / total,
        "names": [name for name, _ in trainable],
        "target_modules": expected_targets,
    }


def epoch_order(size: int, seed: int, epoch: int):
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order


def preflight(processor, rows, config, prompt, output: Path | None = None):
    records = []
    for index, row in enumerate(rows, 1):
        _, stats = encode_training_example(
            processor,
            row,
            Path(config["data_root"]),
            prompt,
            config["max_input_tokens"],
            config["max_response_tokens"],
            config["max_total_tokens"],
            verify_image_hash=True,
        )
        records.append(stats)
        if index % 100 == 0 or index == len(rows):
            print(json.dumps({"preflight": index, "total": len(rows)}), flush=True)
    report = {
        "status": "passed",
        "created_at": now(),
        "samples": len(records),
        "empty_think": sum(item["think_empty"] for item in records),
        "prompt_tokens": {"min": min(item["prompt_tokens"] for item in records), "max": max(item["prompt_tokens"] for item in records)},
        "response_tokens": {"min": min(item["response_tokens"] for item in records), "max": max(item["response_tokens"] for item in records)},
        "total_tokens": {"min": min(item["total_tokens"] for item in records), "max": max(item["total_tokens"] for item in records)},
        "visual_tokens": {"min": min(item["visual_tokens"] for item in records), "max": max(item["visual_tokens"] for item in records)},
        "records": records,
    }
    if output is not None:
        write_json(output, report)
    return report


def capture_rng(torch, numpy):
    return {
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state, torch, numpy):
    random.setstate(state["python"])
    numpy.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def save_adapter_atomic(model, accelerator, destination: Path):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite adapter directory: {destination}")
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    accelerator.unwrap_model(model).save_pretrained(temp, safe_serialization=True)
    os.replace(temp, destination)


def save_checkpoint(model, optimizer, scheduler, accelerator, destination, state, contract_digest, torch, numpy):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    accelerator.unwrap_model(model).save_pretrained(temp / "adapter", safe_serialization=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": capture_rng(torch, numpy),
        },
        temp / "training_state.pt",
    )
    write_json(temp / "checkpoint.json", {**state, "contract_digest": contract_digest, "created_at": now()})
    os.replace(temp, destination)


def append_metric(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sft_drop50.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path, default=Path("reports/sft_processor_audit.json"))
    parser.add_argument("--smoke-limit", type=int, help="Use only the first N rows; marks the run as non-formal")
    args = parser.parse_args()

    config = load_config(args.config)
    data_path, data_manifest_path = Path(config["data"]), Path(config["data_manifest"])
    prompt_path = Path(config["prompt"])
    rows, data_manifest = validate_compiled_dataset(data_path, data_manifest_path, prompt_path)
    for key in ("model", "model_revision"):
        frozen = data_manifest["contract"].get(key)
        if frozen is not None and frozen != config[key]:
            raise ValueError(f"SFT config {key} differs from compiled data manifest")
    formal = args.smoke_limit is None
    if args.smoke_limit is not None:
        if args.smoke_limit <= 0 or args.smoke_limit > len(rows):
            raise ValueError("--smoke-limit must be within the compiled dataset")
        rows = rows[: args.smoke_limit]
    accumulation = int(config["gradient_accumulation_steps"])
    if len(rows) % accumulation:
        raise ValueError("Dataset size must be divisible by gradient_accumulation_steps; do not silently drop samples")
    prompt = prompt_path.read_text(encoding="utf-8").strip()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        config["model"],
        revision=config["model_revision"],
        min_pixels=config["min_pixels"],
        max_pixels=config["max_pixels"],
        size={"shortest_edge": config["min_pixels"], "longest_edge": config["max_pixels"]},
    )
    if args.preflight_only:
        report = preflight(processor, rows, config, prompt, args.preflight_output)
        print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
        return
    preflight_report = preflight(processor, rows, config, prompt)

    import numpy
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("SFT requires exactly one visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("SFT requires a BF16-capable CUDA GPU")
    set_seed(args.seed, device_specific=False)
    output_dir = args.output_dir or Path(config["output_root"]) / f"seed-{args.seed}"
    if args.resume_from is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if not args.resume_from.is_dir():
            raise FileNotFoundError(args.resume_from)
        if not output_dir.is_dir() or not (output_dir / "run_manifest.json").is_file():
            raise ValueError("Resume requires the original output directory and run_manifest.json")

    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "accelerate": version("accelerate"),
        "peft": version("peft"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    contract = {
        "formal": formal,
        "seed": args.seed,
        "config": config,
        "config_sha256": file_hash(args.config),
        "data_sha256": file_hash(data_path),
        "data_manifest_sha256": file_hash(data_manifest_path),
        "prompt_sha256": file_hash(prompt_path),
        "sample_ids_digest": digest([row["id"] for row in rows]),
        "model": config["model"],
        "model_revision": config["model_revision"],
        "trainer_sha256": file_hash(Path(__file__)),
        "sft_common_sha256": file_hash(Path(__file__).with_name("common.py")),
        "runtime": runtime,
    }
    contract_digest = digest(contract)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["contract_digest"] != contract_digest:
            raise ValueError("Run contract changed; resume into the original environment/configuration only")
    else:
        write_json(manifest_path, {"created_at": now(), "contract_digest": contract_digest, "contract": contract})
    write_json(output_dir / "resolved_config.json", config)
    processor.save_pretrained(output_dir / "processor")
    write_json(output_dir / "processor_audit.json", preflight_report)

    accelerator = Accelerator(gradient_accumulation_steps=accumulation, mixed_precision="bf16")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["model"],
        revision=config["model_revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation=config["attention"],
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    targets = discover_lora_targets(model, config["language_module_marker"])
    resume_meta = None
    if args.resume_from is not None:
        resume_meta = json.loads((args.resume_from / "checkpoint.json").read_text())
        if resume_meta["contract_digest"] != contract_digest or resume_meta["target_modules"] != targets:
            raise ValueError("Checkpoint contract or LoRA target list differs from this run")
        model = PeftModel.from_pretrained(model, args.resume_from / "adapter", is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=config["lora_rank"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"],
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=targets,
            ),
        )
    parameter_audit = audit_trainable_parameters(model, targets)
    write_json(output_dir / "trainable_parameters.json", parameter_audit)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    steps_per_epoch = len(rows) // accumulation
    total_steps = steps_per_epoch * int(config["epochs"])
    warmup_steps = int(math.ceil(total_steps * float(config["warmup_ratio"])))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    optimizer.zero_grad(set_to_none=True)

    start_epoch, start_offset, optimizer_step = 0, 0, 0
    if args.resume_from is not None:
        saved = torch.load(args.resume_from / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"], torch, numpy)
        start_epoch = int(resume_meta["next_epoch"])
        start_offset = int(resume_meta["next_offset"])
        optimizer_step = int(resume_meta["optimizer_step"])
        if not 0 <= start_epoch <= int(config["epochs"]):
            raise ValueError("Checkpoint epoch is outside the configured training range")
        if start_offset < 0 or start_offset >= len(rows) or start_offset % accumulation:
            raise ValueError("Checkpoint offset is not an optimizer-step boundary")
        expected_step = start_epoch * steps_per_epoch + start_offset // accumulation
        if optimizer_step != expected_step:
            raise ValueError(f"Checkpoint optimizer step is inconsistent: {optimizer_step} != {expected_step}")

    metrics_path = output_dir / "metrics.jsonl"
    window_losses, window_tokens = [], 0
    started = time.monotonic()
    for epoch in range(start_epoch, int(config["epochs"])):
        order = epoch_order(len(rows), args.seed, epoch)
        offset0 = start_offset if epoch == start_epoch else 0
        for offset in range(offset0, len(order)):
            row = rows[order[offset]]
            batch, stats = encode_training_example(
                processor,
                row,
                Path(config["data_root"]),
                prompt,
                config["max_input_tokens"],
                config["max_response_tokens"],
                config["max_total_tokens"],
                verify_image_hash=False,
            )
            batch = move_batch(batch, accelerator.device)
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    output = model(**batch)
                    loss = output.loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite loss for {row['id']}: {loss.item()}")
                accelerator.backward(loss)
                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(trainable_parameters, config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            window_losses.append(float(loss.detach().cpu()))
            window_tokens += stats["supervised_tokens"]
            if accelerator.sync_gradients:
                optimizer_step += 1
                next_epoch, next_offset = epoch, offset + 1
                if next_offset == len(order):
                    next_epoch, next_offset = epoch + 1, 0
                metric = {
                    "time": now(),
                    "epoch": epoch,
                    "next_offset": next_offset,
                    "optimizer_step": optimizer_step,
                    "loss_mean_over_microbatches": sum(window_losses) / len(window_losses),
                    "supervised_tokens": window_tokens,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm.detach().cpu()) if grad_norm is not None else None,
                    "elapsed_seconds": time.monotonic() - started,
                    "max_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                }
                if optimizer_step % int(config["log_steps"]) == 0:
                    append_metric(metrics_path, metric)
                    accelerator.print(json.dumps(metric), flush=True)
                window_losses, window_tokens = [], 0
                if optimizer_step % int(config["save_steps"]) == 0 or optimizer_step == total_steps:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            accelerator,
                            output_dir / "checkpoints" / f"step-{optimizer_step:06d}",
                            {
                                "optimizer_step": optimizer_step,
                                "next_epoch": next_epoch,
                                "next_offset": next_offset,
                                "target_modules": targets,
                            },
                            contract_digest,
                            torch,
                            numpy,
                        )
        start_offset = 0
    if optimizer_step != total_steps:
        raise RuntimeError(f"Optimizer step mismatch: {optimizer_step} != {total_steps}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_adapter_atomic(model, accelerator, output_dir / "final_adapter")
        write_json(
            output_dir / "completed.json",
            {"status": "complete", "completed_at": now(), "optimizer_steps": optimizer_step, "total_steps": total_steps},
        )


if __name__ == "__main__":
    main()
