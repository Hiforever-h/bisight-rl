"""Shared validation and frozen-contract helpers for ChartQA GRPO."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from bisight_rl.common import digest, file_hash, read_jsonl
from bisight_rl.grpo.reward import decode_ground_truth


EXPECTED_LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("GRPO config must be a mapping")
    required_sections = {"bisight", "data", "algorithm", "worker", "trainer"}
    missing = sorted(required_sections - set(config))
    if missing:
        raise ValueError(f"Missing GRPO config sections: {missing}")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> dict[str, int | float]:
    project = config["bisight"]
    data = config["data"]
    algorithm = config["algorithm"]
    worker = config["worker"]
    actor = worker["actor"]
    rollout = worker["rollout"]
    trainer = config["trainer"]

    required_project = {
        "easy_r1_root",
        "easy_r1_commit",
        "data_manifest",
        "prompt",
        "output_root",
        "dataloader_num_workers",
        "minimum_free_disk_gb",
        "wandb",
    }
    missing = sorted(required_project - set(project))
    if missing:
        raise ValueError(f"Missing bisight config keys: {missing}")
    if len(str(project["easy_r1_commit"])) != 40:
        raise ValueError("easy_r1_commit must be a full 40-character commit")
    if project["dataloader_num_workers"] < 0 or project["minimum_free_disk_gb"] < 0:
        raise ValueError("Dataloader workers and minimum free disk must be non-negative")
    wandb = project["wandb"]
    if not isinstance(wandb, dict) or wandb.get("mode") not in {"online", "offline", "disabled"}:
        raise ValueError("bisight.wandb.mode must be online, offline, or disabled")

    prompt_batch = int(data["rollout_batch_size"])
    generation_batch = data.get("mini_rollout_batch_size")
    generation_batch = prompt_batch if generation_batch is None else int(generation_batch)
    generations = int(rollout["n"])
    actor_prompt_minibatch = int(actor["global_batch_size"])
    update_micro = int(actor["micro_batch_size_per_device_for_update"])
    experience_micro = int(actor["micro_batch_size_per_device_for_experience"])
    if min(prompt_batch, generation_batch, generations, actor_prompt_minibatch, update_micro, experience_micro) <= 0:
        raise ValueError("All GRPO batch sizes must be positive")
    if prompt_batch % generation_batch:
        raise ValueError("rollout_batch_size must be divisible by mini_rollout_batch_size")
    if prompt_batch % actor_prompt_minibatch:
        raise ValueError("rollout_batch_size must be divisible by actor.global_batch_size")
    trajectories = prompt_batch * generations
    effective_actor_minibatch = actor_prompt_minibatch * generations
    if trajectories % experience_micro:
        raise ValueError("Trajectory count must be divisible by the experience micro batch")
    if effective_actor_minibatch % update_micro:
        raise ValueError("Effective actor mini-batch must be divisible by the update micro batch")
    if actor_prompt_minibatch != prompt_batch:
        raise ValueError("This experiment requires exactly one actor mini-batch per rollout step")
    if int(actor.get("ppo_epochs", 0)) != 1:
        raise ValueError("GRPO must use exactly one PPO epoch per rollout batch")

    if algorithm.get("adv_estimator") != "grpo":
        raise ValueError("algorithm.adv_estimator must be grpo")
    if algorithm.get("online_filtering") is not False:
        raise ValueError("DAPO/online filtering must remain disabled")
    if algorithm.get("disable_kl") is not False or algorithm.get("use_kl_loss") is not True:
        raise ValueError("Reference KL must be enabled as an explicit loss")
    if float(algorithm.get("kl_coef", -1)) != 0.04:
        raise ValueError("This frozen starting config requires kl_coef=0.04")
    if float(actor.get("clip_ratio_low", -1)) != 0.2 or float(actor.get("clip_ratio_high", -1)) != 0.2:
        raise ValueError("Both PPO clip bounds must be 0.2")
    if float(actor.get("clip_ratio_dual", 0)) < 1e5:
        raise ValueError("Dual-clip must be effectively disabled for the registered PPO objective")
    if float(actor["optim"].get("lr", -1)) != 1e-6:
        raise ValueError("This frozen starting config requires actor learning rate 1e-6")
    if actor["model"].get("freeze_vision_tower") is not True:
        raise ValueError("Vision tower must be frozen")
    lora = actor["model"]["lora"]
    if int(lora.get("rank", 0)) != 64 or int(lora.get("alpha", 0)) != 128:
        raise ValueError("GRPO LoRA rank/alpha must start at 64/128")
    if lora.get("target_modules") != EXPECTED_LORA_TARGETS:
        raise ValueError("GRPO LoRA target modules differ from the audited language projections")
    if data.get("format_prompt") is not None or data.get("override_chat_template") is not None:
        raise ValueError("Prompt rendering is owned by the BiSight EasyR1 bridge; template overrides must be null")
    if data.get("filter_overlong_prompts") is not False:
        raise ValueError("Overlong prompts must fail preflight instead of being silently filtered")
    if int(trainer.get("n_gpus_per_node", 0)) != 1 or int(rollout.get("tensor_parallel_size", 0)) != 1:
        raise ValueError("This experiment is frozen to one visible GPU and tensor_parallel_size=1")
    if int(trainer.get("max_steps", 0)) <= 0:
        raise ValueError("trainer.max_steps must be a positive sampling-iteration count")
    if int(trainer.get("save_freq", 0)) <= 0 or int(trainer.get("save_limit", 0)) not in {1, 2}:
        raise ValueError("Checkpoint frequency must be positive and save_limit must be 1 or 2")
    if trainer.get("save_model_only") is not False:
        raise ValueError("Checkpoints must retain optimizer/RNG state for resumability")
    if rollout.get("disable_tqdm") is not False:
        raise ValueError("Rollout progress bars must remain enabled")
    val_config = rollout.get("val_override_config", {})
    if float(val_config.get("temperature", -1)) != 0.0 or int(val_config.get("n", 0)) != 1:
        raise ValueError("Validation must use one greedy generation")

    return {
        "prompts_per_iteration": prompt_batch,
        "rollout_generation_chunk_prompts": generation_batch,
        "generation_chunks_per_iteration": prompt_batch // generation_batch,
        "generations_per_prompt": generations,
        "trajectories_per_iteration": trajectories,
        "actor_prompt_minibatch": actor_prompt_minibatch,
        "actor_trajectory_minibatch": effective_actor_minibatch,
        "update_micro_trajectories": update_micro,
        "gradient_accumulation_microbatches": effective_actor_minibatch // update_micro,
        "sampling_iterations": int(trainer["max_steps"]),
        "total_prompt_draws": int(trainer["max_steps"]) * prompt_batch,
        "total_trajectories": int(trainer["max_steps"]) * trajectories,
    }


def validate_easyr1_checkout(root: Path, expected_commit: str, allow_dirty: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    if not (root / "verl/trainer/main.py").is_file() or not (root / ".git").is_dir():
        raise FileNotFoundError(f"EasyR1 checkout not found at {root}")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != expected_commit:
        raise ValueError(f"EasyR1 commit mismatch: {commit} != {expected_commit}")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty and not allow_dirty:
        raise ValueError("EasyR1 checkout has local changes; refuse an unpinned trainer implementation")
    return {"root": str(root), "commit": commit, "dirty": bool(dirty)}


def validate_artifacts(manifest_path: Path, train_path: Path, dev_path: Path, prompt_path: Path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("status") != "validated_grpo_dataset":
        raise ValueError("GRPO dataset manifest is not validated")
    if file_hash(prompt_path) != manifest["contract"]["adaptive_prompt_sha256"]:
        raise ValueError("Adaptive prompt differs from the GRPO data manifest")
    result = {}
    for name, path in (("train", train_path), ("dev_quick", dev_path)):
        artifact = manifest["artifacts"][name]
        if file_hash(path) != artifact["sha256"]:
            raise ValueError(f"{name} data differs from its GRPO manifest")
        rows = read_jsonl(path)
        if len(rows) != artifact["rows"]:
            raise ValueError(f"{name} row count differs from its GRPO manifest")
        ids = [row.get("prompt_id") for row in rows]
        if len(set(ids)) != len(ids) or digest(ids) != artifact["id_digest"]:
            raise ValueError(f"{name} ids/order differ from its GRPO manifest")
        for row in rows:
            leaked = sorted(set(row) & {"assistant", "rationale", "think_empty"})
            if leaked:
                raise ValueError(f"Forbidden SFT fields leaked into GRPO row {row.get('prompt_id')}: {leaked}")
            if not isinstance(row.get("images"), list) or len(row["images"]) != 1:
                raise ValueError(f"Expected one image for {row.get('prompt_id')}")
            payload = decode_ground_truth(row.get("answer"))
            if payload["id"] != row.get("prompt_id"):
                raise ValueError(f"Ground-truth id mismatch for {row.get('prompt_id')}")
        result[name] = rows
    if set(row["prompt_id"] for row in result["train"]) & set(row["prompt_id"] for row in result["dev_quick"]):
        raise ValueError("GRPO train and dev ids overlap")
    return result, manifest


def disk_report(output_root: Path, minimum_free_gb: float) -> dict[str, float]:
    output_root = Path(output_root).resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root.parent)
    free_gb = usage.free / 1024**3
    if free_gb < minimum_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GiB free beside {output_root}; configured minimum is {minimum_free_gb:.1f} GiB"
        )
    return {"total_gb": usage.total / 1024**3, "used_gb": usage.used / 1024**3, "free_gb": free_gb}


def contract_digest(config: dict[str, Any], manifest: dict[str, Any], easyr1: dict[str, Any]) -> str:
    frozen_config = json.loads(json.dumps(config))
    frozen_config["worker"]["reward"]["reward_function_kwargs"]["audit_path"] = None
    # Logging destination and the mechanism used to find the same checkpoint do
    # not change the policy/data/optimizer contract and may change on resume.
    frozen_config["bisight"].pop("wandb", None)
    frozen_config["trainer"].pop("logger", None)
    frozen_config["trainer"]["load_checkpoint_path"] = None
    frozen_config["trainer"].pop("find_last_checkpoint", None)
    return digest(
        {
            "config": frozen_config,
            "data": {key: value["sha256"] for key, value in manifest["artifacts"].items()},
            "prompt": manifest["contract"]["adaptive_prompt_sha256"],
            "easyr1_commit": easyr1["commit"],
        }
    )
