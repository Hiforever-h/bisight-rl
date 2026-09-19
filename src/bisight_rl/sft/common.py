from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from bisight_rl.common import file_hash


ROLE_MARKERS = ("<|im_start|>", "<|im_end|>", "[INST]", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>")


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def load_sft_rows(path: Path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def resolve_image(data_root: Path, relative_path: str) -> Path:
    candidate = (Path(data_root).resolve() / relative_path).resolve()
    try:
        candidate.relative_to(Path(data_root).resolve())
    except ValueError as exc:
        raise ValueError(f"Image path escapes data root: {relative_path}") from exc
    return candidate


def validate_compiled_dataset(data_path: Path, manifest_path: Path, prompt_path: Path):
    manifest = load_json(manifest_path)
    if file_hash(data_path) != manifest["artifact"]["sha256"]:
        raise ValueError("Compiled SFT data differs from its manifest")
    if file_hash(prompt_path) != manifest["contract"]["adaptive_prompt_sha256"]:
        raise ValueError("Adaptive prompt differs from the frozen SFT manifest")
    rows = load_sft_rows(data_path)
    if len(rows) != manifest["artifact"]["rows"]:
        raise ValueError("Compiled SFT row count differs from its manifest")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Compiled SFT IDs are not unique")
    empty = sum(bool(row["think_empty"]) for row in rows)
    if empty != manifest["artifact"]["empty_think"]:
        raise ValueError("Compiled SFT empty-think count differs from its manifest")
    return rows, manifest


def student_messages(prompt: str, question: str):
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]


def encode_prompt_example(processor, row, data_root: Path, prompt: str, verify_image_hash: bool = True):
    image_path = resolve_image(data_root, row["image_path"])
    if not image_path.is_file():
        raise ValueError(f"Missing image for {row['id']}: {image_path}")
    if verify_image_hash and file_hash(image_path) != row["image_sha256"]:
        raise ValueError(f"Image checksum changed for {row['id']}")
    text = processor.apply_chat_template(student_messages(prompt, row["question"]), tokenize=False, add_generation_prompt=True)
    if text.rstrip().endswith(("<think>", "<answer>")):
        raise ValueError(f"Student prompt unexpectedly prefills response tags for {row['id']}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        return processor(text=[text], images=[image], return_tensors="pt", padding=False, truncation=False)


def encode_training_example(
    processor,
    row,
    data_root: Path,
    prompt: str,
    max_input_tokens: int,
    max_response_tokens: int,
    max_total_tokens: int,
    verify_image_hash: bool = True,
):
    """Apply the real multimodal template and construct an assistant-only mask.

    The prompt and complete conversation are independently processed with the
    same image. The complete token sequence must begin with the prompt token
    sequence exactly; a mismatch is an error rather than a guessed boundary.
    """
    import torch

    image_path = resolve_image(data_root, row["image_path"])
    if not image_path.is_file():
        raise ValueError(f"Missing image for {row['id']}: {image_path}")
    if verify_image_hash and file_hash(image_path) != row["image_sha256"]:
        raise ValueError(f"Image checksum changed for {row['id']}")
    messages = student_messages(prompt, row["question"])
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prompt_text.rstrip().endswith(("<think>", "<answer>")):
        raise ValueError(f"Student prompt unexpectedly prefills response tags for {row['id']}")
    full_text = processor.apply_chat_template(
        messages + [{"role": "assistant", "content": row["assistant"]}],
        tokenize=False,
        add_generation_prompt=False,
    )
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt", padding=False, truncation=False)
        full_inputs = processor(text=[full_text], images=[image], return_tensors="pt", padding=False, truncation=False)
    prompt_ids = prompt_inputs["input_ids"][0]
    full_ids = full_inputs["input_ids"][0]
    prompt_tokens, total_tokens = int(prompt_ids.numel()), int(full_ids.numel())
    response_tokens = total_tokens - prompt_tokens
    if response_tokens <= 0 or not torch.equal(full_ids[:prompt_tokens], prompt_ids):
        raise ValueError(f"Chat-template token prefix mismatch for {row['id']}")
    if prompt_tokens > max_input_tokens:
        raise ValueError(f"Prompt too long for {row['id']}: {prompt_tokens} > {max_input_tokens}")
    if response_tokens > max_response_tokens:
        raise ValueError(f"Response too long for {row['id']}: {response_tokens} > {max_response_tokens}")
    if total_tokens > max_total_tokens:
        raise ValueError(f"Sequence too long for {row['id']}: {total_tokens} > {max_total_tokens}")

    labels = full_ids.clone()
    labels[:prompt_tokens] = -100
    if "attention_mask" in full_inputs:
        labels[full_inputs["attention_mask"][0] == 0] = -100
    if not bool(torch.all(labels[:prompt_tokens] == -100)):
        raise ValueError(f"Prompt labels are not fully masked for {row['id']}")
    supervised = labels[labels != -100]
    decoded = processor.tokenizer.decode(
        supervised,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    if decoded != row["assistant"].strip():
        raise ValueError(
            f"Supervised tokens do not decode to the assistant target for {row['id']}: "
            f"expected={row['assistant']!r}, decoded={decoded!r}"
        )
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    visual_tokens = int((full_ids == image_token_id).sum().item())
    if visual_tokens <= 0:
        raise ValueError(f"No visual tokens found for {row['id']}")
    if bool(torch.any(labels[full_ids == image_token_id] != -100)):
        raise ValueError(f"Visual tokens are supervised for {row['id']}")
    full_inputs["labels"] = labels.unsqueeze(0)
    stats = {
        "id": row["id"],
        "think_empty": bool(row["think_empty"]),
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": total_tokens,
        "supervised_tokens": int(supervised.numel()),
        "visual_tokens": visual_tokens,
        "image_grid_thw": full_inputs.get("image_grid_thw").tolist() if full_inputs.get("image_grid_thw") is not None else None,
    }
    return full_inputs, stats


def move_batch(batch, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
