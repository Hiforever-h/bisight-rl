"""Launch single-GPU ChartQA GRPO through a pinned EasyR1 checkout."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

from bisight_rl.common import now, read_jsonl, write_json, write_jsonl
from bisight_rl.grpo.common import contract_digest, load_config, validate_config
from bisight_rl.grpo.preflight import run_preflight


def _path(value) -> Path:
    return Path(value).expanduser().resolve()


def apply_overrides(config: dict, args) -> dict:
    config = copy.deepcopy(config)
    original_seed = int(config["data"]["seed"])
    if args.model_path is not None:
        config["worker"]["actor"]["model"]["model_path"] = str(args.model_path)
    seed = int(args.seed if args.seed is not None else config["data"]["seed"])
    config["data"]["seed"] = seed
    config["worker"]["rollout"]["seed"] = seed
    if args.p0 and args.max_steps is not None:
        raise ValueError("Use either --p0 or --max-steps, not both")
    if args.p0:
        config["trainer"]["max_steps"] = 5
    elif args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        config["trainer"]["max_steps"] = args.max_steps

    configured_output = Path(config["bisight"]["output_root"])
    if args.output_dir is not None:
        output_root = args.output_dir
    elif seed != original_seed:
        output_root = configured_output.parent / f"seed-{seed}"
    else:
        output_root = configured_output
    if args.p0 and args.output_dir is None:
        output_root = output_root.with_name(output_root.name + "-p0")
    config["bisight"]["output_root"] = str(output_root)
    config["trainer"]["save_checkpoint_path"] = str(Path(output_root) / "checkpoints")
    base_name = f"grpo-drop50-seed-{seed}"
    config["trainer"]["experiment_name"] = base_name + ("-p0" if args.p0 else "")
    if args.resume_from is not None:
        config["trainer"]["load_checkpoint_path"] = str(args.resume_from)
        config["trainer"]["find_last_checkpoint"] = False

    mode = args.wandb_mode or config["bisight"]["wandb"]["mode"]
    config["bisight"]["wandb"]["mode"] = mode
    loggers = list(config["trainer"]["logger"])
    if mode == "disabled":
        loggers = [item for item in loggers if item != "wandb"]
    elif "wandb" not in loggers:
        loggers.append("wandb")
    config["trainer"]["logger"] = loggers
    validate_config(config)
    return config


def configure_environment(config: dict, contract: str) -> dict[str, str]:
    output_root = _path(config["bisight"]["output_root"])
    wandb_dir = output_root / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb = config["bisight"]["wandb"]
    values = {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        "TQDM_MININTERVAL": "1",
        "WANDB_MODE": wandb["mode"],
        "WANDB_DIR": str(wandb_dir),
        "WANDB_CACHE_DIR": str(wandb_dir / ".cache"),
        "WANDB_RUN_GROUP": str(wandb.get("group") or "grpo"),
        "WANDB_TAGS": ",".join(wandb.get("tags") or []),
        "WANDB_RUN_ID": f"grpo-{contract[:20]}",
    }
    if wandb["mode"] == "online":
        values["WANDB_RESUME"] = "allow"
    source_root = Path(__file__).resolve().parents[3] / "src"
    python_paths = [str(_path(config["bisight"]["easy_r1_root"])), str(source_root)]
    if os.environ.get("PYTHONPATH"):
        python_paths.append(os.environ["PYTHONPATH"])
    values["PYTHONPATH"] = os.pathsep.join(python_paths)
    for key, value in values.items():
        os.environ[key] = value
    return values


def to_easyr1_config(config: dict):
    from omegaconf import OmegaConf
    from verl.trainer.config import PPOConfig

    easy_config = copy.deepcopy(config)
    easy_config.pop("bisight")
    default = OmegaConf.structured(PPOConfig())
    merged = OmegaConf.merge(default, OmegaConf.create(easy_config))
    result: PPOConfig = OmegaConf.to_object(merged)
    result.deep_post_init()
    return result


def reconcile_rollout_log(path: Path, checkpoint_step: int | None) -> dict:
    """Discard only uncheckpointed audit rows, preserving them in an orphan file."""
    path = Path(path)
    if not path.exists():
        if checkpoint_step is not None and checkpoint_step > 0:
            raise FileNotFoundError(f"Checkpoint step {checkpoint_step} exists but rollout audit is missing: {path}")
        return {"status": "new", "checkpoint_step": checkpoint_step, "kept_rows": 0}
    rows = read_jsonl(path)
    if checkpoint_step is None:
        raise FileExistsError("rollouts.jsonl exists without a resumable checkpoint; choose a new --output-dir")
    completed_batches = {int(row["reward_batch"]) for row in rows}
    if checkpoint_step > 0 and not all(step in completed_batches for step in range(1, checkpoint_step + 1)):
        raise ValueError("Rollout audit is missing one or more batches covered by the checkpoint")
    kept = [row for row in rows if int(row["reward_batch"]) <= checkpoint_step]
    stale = [row for row in rows if int(row["reward_batch"]) > checkpoint_step]
    orphan_path = None
    if stale:
        last_stale = max(int(row["reward_batch"]) for row in stale)
        orphan_path = path.with_name(f"orphaned_rollouts_after_step_{checkpoint_step}_through_{last_stale}.jsonl")
        if orphan_path.exists():
            raise FileExistsError(f"Refusing to overwrite prior orphaned rollout audit: {orphan_path}")
        write_jsonl(orphan_path, stale)
        write_jsonl(path, kept)
    return {
        "status": "reconciled",
        "checkpoint_step": checkpoint_step,
        "kept_rows": len(kept),
        "orphaned_rows": len(stale),
        "orphan_path": str(orphan_path) if orphan_path else None,
    }


def launch_easyr1(ppo_config, system_prompt: str, num_workers: int, runtime_env_vars: dict[str, str]):
    import ray

    @ray.remote(num_cpus=1)
    class ChartQARunner:
        def run(self, config, prompt, loader_workers):
            from copy import deepcopy

            import ray

            from verl.single_controller.ray import RayWorkerGroup
            from verl.trainer.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
            from verl.utils.tokenizer import get_processor, get_tokenizer
            from verl.workers.fsdp_workers import FSDPWorker
            from verl.workers.reward import AutoRewardManager

            from bisight_rl.grpo.easyr1_bridge import create_chartqa_dataloaders, install_append_safe_file_logger

            print(json.dumps(config.to_dict(), indent=2))
            install_append_safe_file_logger()
            tokenizer = get_tokenizer(
                config.worker.actor.model.model_path,
                override_chat_template=config.data.override_chat_template,
                trust_remote_code=config.worker.actor.model.trust_remote_code,
                use_fast=True,
            )
            processor = get_processor(
                config.worker.actor.model.model_path,
                override_chat_template=config.data.override_chat_template,
                trust_remote_code=config.worker.actor.model.trust_remote_code,
                use_fast=True,
            )
            role_worker_mapping = {
                Role.ActorRolloutRef: ray.remote(FSDPWorker),
                Role.Critic: ray.remote(FSDPWorker),
            }
            pool_id = "global_pool"
            resource_pool_manager = ResourcePoolManager(
                resource_pool_spec={pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
                mapping={Role.ActorRolloutRef: pool_id, Role.Critic: pool_id},
            )
            RemoteRewardManager = ray.remote(AutoRewardManager).options(num_cpus=config.worker.reward.num_cpus)
            train_reward = RemoteRewardManager.remote(config.worker.reward, tokenizer)
            val_reward_config = deepcopy(config.worker.reward)
            val_reward_config.reward_function_kwargs["audit_path"] = None
            val_reward = RemoteRewardManager.remote(val_reward_config, tokenizer)
            train_loader, val_loader = create_chartqa_dataloaders(
                config.data,
                tokenizer,
                processor,
                prompt,
                config.worker.rollout.n,
                loader_workers,
            )
            trainer = RayPPOTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                role_worker_mapping=role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=RayWorkerGroup,
                reward_fn=train_reward,
                val_reward_fn=val_reward,
            )
            trainer.init_workers()
            trainer.fit()

    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": runtime_env_vars})
    try:
        runner = ChartQARunner.remote()
        ray.get(runner.run.remote(ppo_config, system_prompt, num_workers))
    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/grpo_drop50.yaml"))
    parser.add_argument("--model-path", type=Path, help="Validated merged SFT model directory")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--p0", action="store_true", help="Run the five-iteration sampling/update smoke test")
    parser.add_argument("--resume-from", type=Path, help="Explicit EasyR1 global_step_* checkpoint")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-processor-check", action="store_true", help="Allowed only with --preflight-only")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    parser.add_argument("--allow-dirty-easyr1", action="store_true")
    args = parser.parse_args()
    if args.skip_processor_check and not args.preflight_only:
        raise ValueError("Training cannot skip the real EasyR1/SFT processor comparison")

    config = apply_overrides(load_config(args.config), args)
    easy_root = _path(config["bisight"]["easy_r1_root"])
    if str(easy_root) not in sys.path:
        sys.path.insert(0, str(easy_root))
    report = run_preflight(
        config,
        processor_check=not args.skip_processor_check,
        allow_dirty_easyr1=args.allow_dirty_easyr1,
    )
    output_root = _path(config["bisight"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "preflight.json", report)
    printable_processor = {key: value for key, value in report["processor"].items() if key != "records"}
    print(json.dumps({"status": "preflight passed", "batch_contract": report["batch_contract"], "processor": printable_processor}, indent=2))
    if args.preflight_only:
        return

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GRPO requires exactly one visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GRPO requires a BF16-capable CUDA GPU")

    config["worker"]["reward"]["reward_function_kwargs"]["audit_path"] = str(output_root / "rollouts.jsonl")
    contract = contract_digest(config, report["data_manifest"], report["easyr1"])
    contract_path = output_root / "run_contract.json"
    if contract_path.exists():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous.get("contract_digest") != contract:
            raise ValueError("Existing GRPO output directory belongs to a different frozen contract")
    else:
        write_json(
            contract_path,
            {
                "status": "initialized",
                "created_at": now(),
                "contract_digest": contract,
                "config": config,
                "easyr1": report["easyr1"],
                "data_manifest_sha256": report["data"]["manifest_sha256"],
            },
        )
    rollout_log = output_root / "rollouts.jsonl"
    checkpoint_tracker = _path(config["trainer"]["save_checkpoint_path"]) / "checkpoint_tracker.json"
    if args.resume_from is not None:
        checkpoint_name = args.resume_from.resolve().name
        if not checkpoint_name.startswith("global_step_"):
            raise ValueError("--resume-from must end in global_step_*")
        checkpoint_step = int(checkpoint_name.removeprefix("global_step_"))
    elif checkpoint_tracker.exists():
        tracker = json.loads(checkpoint_tracker.read_text(encoding="utf-8"))
        checkpoint_step = int(tracker["last_global_step"])
    else:
        checkpoint_step = None
    reconciliation = reconcile_rollout_log(rollout_log, checkpoint_step)
    write_json(output_root / "rollout_reconciliation.json", reconciliation)

    runtime_env = configure_environment(config, contract)
    ppo_config = to_easyr1_config(config)
    print(
        "Starting EasyR1. The 'Running step' tqdm bar counts 16-prompt sampling/update iterations "
        "and reports elapsed time plus ETA; nested bars show generation/log-prob/update work."
    )
    launch_easyr1(
        ppo_config,
        _path(config["bisight"]["prompt"]).read_text(encoding="utf-8").strip(),
        int(config["bisight"]["dataloader_num_workers"]),
        runtime_env,
    )
    write_json(
        output_root / "completed.json",
        {
            "status": "completed",
            "completed_at": now(),
            "contract_digest": contract,
            "sampling_iterations": config["trainer"]["max_steps"],
            "trajectories": report["batch_contract"]["trajectories_per_iteration"]
            * config["trainer"]["max_steps"],
        },
    )


if __name__ == "__main__":
    main()
