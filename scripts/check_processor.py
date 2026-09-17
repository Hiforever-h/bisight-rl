"""Check real Qwen multimodal preprocessing on CPU, without loading model weights."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from transformers import AutoProcessor

from bisight_rl.common import file_hash, read_jsonl, write_json
from bisight_rl.data import candidate_order, stratum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("reports/processor_smoke.json"))
    args = parser.parse_args()
    source = json.loads((args.data_root / "manifests/source.json").read_text())
    cfg = yaml.safe_load(Path("configs/generate_rationales.yaml").read_text())
    rows = read_jsonl(args.data_root / "candidates/pilot_100.jsonl")
    # Include each answer kind/source at least once, then follow frozen order.
    selected, seen = [], set()
    for row in rows:
        if stratum(row) not in seen:
            selected.append(row)
            seen.add(stratum(row))
    selected_ids = {r["id"] for r in selected}
    selected += [r for r in rows if r["id"] not in selected_ids]
    selected = selected[:args.limit]
    processor = AutoProcessor.from_pretrained(source["model"], revision=source["model_revision"],
                                              min_pixels=cfg["min_pixels"], max_pixels=cfg["max_pixels"],
                                              size={"shortest_edge": cfg["min_pixels"], "longest_edge": cfg["max_pixels"]})
    assert processor.image_processor.size == {"shortest_edge": cfg["min_pixels"], "longest_edge": cfg["max_pixels"]}
    control = "<think>\n\n</think><answer>7</answer>"
    assert processor.tokenizer.decode(processor.tokenizer.encode(control), skip_special_tokens=True) == control
    boundary = processor.image_processor(images=Image.new("RGB", (2048, 2048)), return_tensors="pt")
    grid = boundary["image_grid_thw"][0]
    actual_pixels = int(grid[1] * grid[2]) * processor.image_processor.patch_size ** 2
    assert actual_pixels <= cfg["max_pixels"]
    prompt = Path("prompts/adaptive.txt").read_text().strip()
    records = []
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    for row in selected:
        with Image.open(args.data_root / row["image_path"]) as image:
            image = image.convert("RGB")
            messages = [{"role": "system", "content": prompt}, {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": row["question"]}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt")
            assert text.count("<|image_pad|>") == 1
            assert inputs["image_grid_thw"].shape[0] == 1
            count = int(inputs["input_ids"].shape[-1])
            visual = int((inputs["input_ids"] == image_token_id).sum())
            assert count <= cfg["max_input_tokens"] and visual > 0
            empty = "<think>\n\n</think><answer>" + row["canonical_answer"] + "</answer>"
            full_text = processor.apply_chat_template(messages + [{"role": "assistant", "content": empty}], tokenize=False, add_generation_prompt=False)
            assert empty in full_text
            encoded = processor(text=[full_text], images=[image], return_tensors="pt")
            decoded = processor.tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
            assert empty in decoded
            records.append({"id": row["id"], "input_tokens": count, "visual_tokens": visual,
                            "image_grid_thw": inputs["image_grid_thw"].tolist(), "empty_think_preserved": True})
    report = {"status": "passed", "model_revision": source["model_revision"], "samples": len(records),
              "processor_class": type(processor).__name__, "large_image_processed_pixels": actual_pixels, "image_processor": processor.image_processor.to_dict(),
              "input_token_min": min(r["input_tokens"] for r in records),
              "input_token_max": max(r["input_tokens"] for r in records),
              "records": records,
              "limitation": "CPU processor/template smoke only; GPU forward and actual training-loader loss masks remain pending"}
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"records", "image_processor"}}, indent=2))


if __name__ == "__main__":
    main()
