"""A800 reverse-thinking generation. Each attempt is durable and resumable."""
import argparse
import json
import platform
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import yaml

from bisight_rl.common import digest, file_hash, now, read_jsonl, write_json, write_jsonl
from bisight_rl.data import stratum
from bisight_rl.quality import check_response

THINK_PREFIX = "<think>"


def attempt_path(run, row, attempt):
    return run / "attempts" / f"{digest(row['id'])[:24]}-{attempt}.json"


def load_attempts(run, row, max_attempts):
    attempts = []
    for n in range(max_attempts):
        path = attempt_path(run, row, n)
        if path.exists():
            result = json.loads(path.read_text())
            if result["sample_id"] != row["id"] or result["attempt"] != n:
                raise ValueError(f"Corrupt attempt record: {path}")
            attempts.append(result)
    return attempts


def get_candidate(attempts, reviews):
    for result in attempts:
        review = reviews.get(result["response_sha256"])
        if result["quality"]["auto_pass"] and not (review and review["decision"] == "reject"):
            return result
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pilot", "full"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/rationales/v2"))
    parser.add_argument("--config", type=Path, default=Path("configs/generate_rationales.yaml"))
    parser.add_argument("--limit", type=int, help="Generate only the first N pilot rows for a smoke test")
    parser.add_argument("--dry-run", action="store_true", help="Validate contracts and print plan without loading a model")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    source = json.loads((args.data_root / "manifests/source.json").read_text())
    build = json.loads((args.data_root / "manifests/build.json").read_text())
    name = "pilot_100.jsonl" if args.stage == "pilot" else "train_candidates.jsonl"
    candidates_path = args.data_root / "candidates" / name
    artifact = build["artifacts"]["candidates/" + name]
    if file_hash(candidates_path) != artifact["sha256"]:
        raise ValueError("Candidate file differs from frozen manifest")
    rows = read_jsonl(candidates_path)
    if any(r["split"] != "train" or r["data_errors"] for r in rows):
        raise ValueError("Generation input must be clean train-only rows")
    if args.limit is not None:
        if args.stage != "pilot" or args.limit <= 0:
            raise ValueError("--limit must be a positive integer and is only supported for --stage pilot")
        rows = rows[:args.limit]
    reverse_path, adaptive_path = Path("prompts/reverse_thinking.txt"), Path("prompts/adaptive.txt")
    contract = {
        "source_revision": source["dataset_revision"], "model": source["model"], "model_revision": source["model_revision"],
        "build_sha256": file_hash(args.data_root / "manifests/build.json"),
        "generation_config": cfg, "reverse_prompt_sha256": file_hash(reverse_path),
        "adaptive_prompt_sha256": file_hash(adaptive_path),
        "generator_sha256": file_hash(__file__),
        "quality_sha256": file_hash(Path(__file__).resolve().parents[1] / "src/bisight_rl/quality.py"),
    }
    if args.dry_run:
        print(json.dumps({"stage": args.stage, "candidate_count": len(rows), "target": build["target_size"], "contract": contract}, indent=2))
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    # Keep a process lock for the complete run; CPU/GPU retries cannot overwrite attempts.
    import fcntl
    lock = (args.run_dir / ".generation.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["contract"] != contract:
            raise ValueError("Run config/code/prompt changed: choose a new --run-dir")
    else:
        manifest = {"created_at": now(), "contract": contract}
    review_path = args.run_dir / "human_reviews.json"
    reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
    if args.stage == "full":
        gate_path = args.run_dir / "pilot_gate.json"
        if not gate_path.exists():
            raise ValueError("First export and complete the 100-row pilot review; see README")
        gate = json.loads(gate_path.read_text())
        if not gate["passed"] or gate["contract_sha256"] != digest(contract) or gate["reviews_digest"] != digest({k: {f: reviews[k][f] for f in ("decision", "hint_leak")} for k in gate["review_keys"]}):
            raise ValueError("Pilot review gate is stale or failed")
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU, e.g. CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU required")
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "transformers": version("transformers"),
               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
               "dependencies": {p: version(p) for p in ("Pillow", "numpy", "tokenizers", "accelerate", "huggingface_hub", "torchvision")}}
    if manifest.get("runtime") and manifest["runtime"] != runtime:
        raise ValueError("Generation runtime changed; use a separate run directory")
    manifest["runtime"] = runtime
    write_json(manifest_path, manifest)
    processor = AutoProcessor.from_pretrained(source["model"], revision=source["model_revision"],
                                              min_pixels=cfg["min_pixels"], max_pixels=cfg["max_pixels"],
                                              size={"shortest_edge": cfg["min_pixels"], "longest_edge": cfg["max_pixels"]})
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        source["model"], revision=source["model_revision"], torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation=cfg["attention"],
    ).eval()
    reverse, adaptive = reverse_path.read_text().strip(), adaptive_path.read_text().strip()
    target_counts = build["target_quotas"] if args.stage == "full" else Counter(stratum(r) for r in rows)
    counts, selected, exhausted = Counter(), [], []
    for row in rows:
        key = stratum(row)
        if counts[key] >= target_counts.get(key, 0):
            continue
        image_path = args.data_root / row["image_path"]
        if file_hash(image_path) != row["image_file_sha256"]:
            raise ValueError(f"Image checksum mismatch: {row['id']}")
        attempts = load_attempts(args.run_dir, row, cfg["max_attempts"])
        accepted = get_candidate(attempts, reviews)
        if accepted is None and len(attempts) < cfg["max_attempts"]:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                user = [{"type": "image"}, {"type": "text", "text": row["question"]}]
                student_messages = [{"role": "system", "content": adaptive}, {"role": "user", "content": user}]
                student_text = processor.apply_chat_template(student_messages, tokenize=False, add_generation_prompt=True)
                student_inputs = processor(text=[student_text], images=[image], return_tensors="pt")
                student_tokens = int(student_inputs["input_ids"].shape[-1])
                teacher_user = [{"type": "image"}, {"type": "text", "text": row["question"] + "\n\nTarget answer (private constraint; never mention it as a target in the output): " + row["canonical_answer"]}]
                teacher_messages = [{"role": "system", "content": reverse}, {"role": "user", "content": teacher_user}]
                # Prefill the structural opening tag. Qwen3-VL otherwise tends to
                # emit prose followed by <answer> while omitting <think> entirely.
                teacher_text = processor.apply_chat_template(teacher_messages, tokenize=False, add_generation_prompt=True) + THINK_PREFIX
                inputs = processor(text=[teacher_text], images=[image], return_tensors="pt").to("cuda")
                teacher_tokens = int(inputs["input_ids"].shape[-1])
                for n in range(len(attempts), cfg["max_attempts"]):
                    seed = int(digest([cfg["seed"], row["id"], n])[:8], 16)
                    set_seed(seed)
                    t0 = time.monotonic()
                    if student_tokens > cfg["max_input_tokens"] or teacher_tokens + cfg["max_new_tokens"] > model.config.text_config.max_position_embeddings:
                        text, output_count, reason = "", 0, "input_too_long"
                    else:
                        with torch.inference_mode():
                            output = model.generate(**inputs, do_sample=True, temperature=cfg["temperature"], top_p=cfg["top_p"],
                                                    max_new_tokens=cfg["max_new_tokens"], use_cache=True)
                        torch.cuda.synchronize()
                        generated = output[0, teacher_tokens:]
                        output_count = len(generated)
                        eos = model.generation_config.eos_token_id
                        eos_ids = set(eos if isinstance(eos, list) else [eos])
                        reason = "stop" if output_count and int(generated[-1]) in eos_ids else "length"
                        generated_suffix = processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        text = THINK_PREFIX + generated_suffix
                    quality = check_response(text, row["canonical_answer"], reason)
                    full_tokens = None
                    if quality["auto_pass"]:
                        assistant = "<think>" + quality["rationale"] + "</think><answer>" + row["canonical_answer"] + "</answer>"
                        sft_text = processor.apply_chat_template(student_messages + [{"role": "assistant", "content": assistant}], tokenize=False, add_generation_prompt=False)
                        sft = processor(text=[sft_text], images=[image], return_tensors="pt")
                        full_tokens = int(sft["input_ids"].shape[-1])
                        if full_tokens > cfg["max_sft_tokens"] or full_tokens - student_tokens > cfg["max_new_tokens"]:
                            quality["auto_pass"] = False
                            quality["errors"].append("sft_length_exceeded")
                    response_hash = digest([row["id"], n, text])
                    result = {"sample_id": row["id"], "attempt": n, "seed": seed, "created_at": now(),
                              "contract_sha256": digest(contract), "response_sha256": response_hash,
                              "raw_response": text, "assistant_prefix": THINK_PREFIX,
                              "finish_reason": reason, "generation_seconds": time.monotonic() - t0,
                              "student_input_tokens": student_tokens, "teacher_input_tokens": teacher_tokens,
                              "output_tokens": output_count, "sft_total_tokens": full_tokens, "quality": quality}
                    write_json(attempt_path(args.run_dir, row, n), result)
                    attempts.append(result)
                    if quality["auto_pass"] or reason == "input_too_long":
                        break
                del inputs
            accepted = get_candidate(attempts, reviews)
        if accepted:
            counts[key] += 1
            selected.append({**row, "rationale": accepted["quality"]["rationale"], "generation": accepted,
                             "quality_status": "auto_checked_pending_final_human_sample"})
        else:
            exhausted.append({"id": row["id"], "attempts": len(attempts), "errors": attempts[-1]["quality"]["errors"] if attempts else []})
        # Checkpoint the ordered candidate master after each processed question.
        write_jsonl(args.run_dir / f"{args.stage}_candidate_master.jsonl", selected)
        write_jsonl(args.run_dir / f"{args.stage}_exhausted.jsonl", exhausted)
        print(json.dumps({"id": row["id"], "accepted": len(selected), "exhausted": len(exhausted), "counts": dict(counts)}), flush=True)
    write_json(args.run_dir / f"{args.stage}_summary.json", {"status": "candidate_not_final", "counts": dict(counts), "target_counts": dict(target_counts),
                                                           "selected": len(selected), "exhausted": len(exhausted), "complete": dict(counts) == dict(target_counts)})
    if args.stage == "full" and len(selected) != build["target_size"]:
        raise RuntimeError("Candidate pool exhausted before quotas filled; inspect failures, do not silently relax quality")


if __name__ == "__main__":
    main()
