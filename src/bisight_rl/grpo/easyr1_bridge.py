"""Small EasyR1 bridge preserving the exact SFT system/user message layout."""
from __future__ import annotations


def install_append_safe_file_logger():
    """Keep EasyR1's local metrics across checkpoint resumes."""
    import json
    import os

    import verl.utils.logger.logger as logger_module

    class AppendSafeFileLogger(logger_module.Logger):
        def __init__(self, config):
            self.config = config
            root = config["trainer"]["save_checkpoint_path"]
            print(f"Initializing append-safe logging files in {root}.")
            os.makedirs(root, exist_ok=True)
            config_path = os.path.join(root, "experiment_config.json")
            if not os.path.exists(config_path):
                with open(config_path, "w", encoding="utf-8") as handle:
                    json.dump(config, handle, indent=2)
            for name in ("experiment_log.jsonl", "generations.log"):
                open(os.path.join(root, name), "a", encoding="utf-8").close()

        def log(self, data, step):
            path = os.path.join(self.config["trainer"]["save_checkpoint_path"], "experiment_log.jsonl")
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, **logger_module.unflatten_dict(data)}) + "\n")

    logger_module.LOGGERS["file"] = AppendSafeFileLogger


def make_chartqa_dataset_class():
    # Keep EasyR1 optional for local CPU-only unit tests.
    from verl.utils.dataset import RLHFDataset

    from bisight_rl.sft.common import student_messages

    class ChartQARLHFDataset(RLHFDataset):
        def __init__(self, *args, system_prompt: str, **kwargs):
            if not system_prompt.strip():
                raise ValueError("System prompt must not be empty")
            self.system_prompt = system_prompt.strip()
            super().__init__(*args, **kwargs)

        def _build_messages(self, example):
            return student_messages(self.system_prompt, example[self.prompt_key])

    return ChartQARLHFDataset


def create_chartqa_dataloaders(
    config,
    tokenizer,
    processor,
    system_prompt: str,
    rollout_n: int,
    num_workers: int = 4,
):
    """Mirror the pinned EasyR1 loader while using the SFT message structure."""
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler
    from torchdata.stateful_dataloader import StatefulDataLoader

    from verl.utils.dataset import collate_fn

    DatasetClass = make_chartqa_dataset_class()
    common = {
        "tokenizer": tokenizer,
        "processor": processor,
        "prompt_key": config.prompt_key,
        "answer_key": config.answer_key,
        "image_key": config.image_key,
        "video_key": config.video_key,
        "image_dir": config.image_dir,
        "video_fps": config.video_fps,
        "max_prompt_length": config.max_prompt_length,
        "truncation": "right",
        "format_prompt": None,
        "min_pixels": config.min_pixels,
        "max_pixels": config.max_pixels,
        "filter_overlong_prompts": config.filter_overlong_prompts,
        "filter_overlong_prompts_workers": config.filter_overlong_prompts_workers,
        "system_prompt": system_prompt,
    }
    train_dataset = DatasetClass(data_path=config.train_files, **common)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    sampler = RandomSampler(train_dataset, generator=generator) if config.shuffle else SequentialSampler(train_dataset)
    train_batch_size = config.mini_rollout_batch_size or config.rollout_batch_size
    train_loader = StatefulDataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_dataset = DatasetClass(data_path=config.val_files, **common)
    val_batch_size = len(val_dataset) if config.val_batch_size == -1 else config.val_batch_size
    val_loader = StatefulDataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )
    if len(train_loader) < 1 or len(val_loader) < 1:
        raise ValueError("GRPO train/validation dataloaders must not be empty")
    print(
        "BiSight batch mapping: "
        f"loader_chunk={train_batch_size} prompts, rollout_step={config.rollout_batch_size} prompts, "
        f"n={rollout_n}, trajectories={config.rollout_batch_size * rollout_n}."
    )
    return train_loader, val_loader
