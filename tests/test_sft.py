import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from bisight_rl.common import file_hash, write_json, write_jsonl
from bisight_rl.sft.build_sft import canonical_assistant, compile_rows
from bisight_rl.sft.common import encode_training_example, validate_compiled_dataset
from bisight_rl.sft.evaluate_sft import evaluate_response, summarize, validate_evaluation_rows
from bisight_rl.sft.train_sft import audit_trainable_parameters, discover_lora_targets, epoch_order, init_wandb
from bisight_rl.sft.merge_sft import compare_captures, validation_rows


def source_row(image_path, image_hash, rationale="", answer="4.0", canonical="4"):
    return {
        "id": "sample:1",
        "split": "train",
        "image_path": image_path,
        "image_file_sha256": image_hash,
        "image_width": 8,
        "image_height": 6,
        "question": "What is the value?",
        "canonical_answer": canonical,
        "rationale": rationale,
        "generation": {"raw_response": canonical_assistant(rationale, answer)},
        "data_errors": [],
        "source": "human",
        "answer_kind": "numeric",
    }


def test_compile_empty_think_and_equivalent_answer(tmp_path):
    image_path = tmp_path / "images/example.png"
    image_path.parent.mkdir()
    Image.new("RGB", (8, 6), "white").save(image_path)
    row = source_row("images/example.png", file_hash(image_path))
    compiled = compile_rows([row], tmp_path)
    assert compiled[0]["assistant"] == "<think>\n\n</think>\n<answer>4.0</answer>"
    assert compiled[0]["think_empty"] is True
    duplicate = dict(row)
    with pytest.raises(ValueError, match="Duplicate sample ID"):
        compile_rows([row, duplicate], tmp_path, verify_images=False)


def test_compile_rejects_control_token_in_question(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 6), "white").save(image_path)
    row = source_row("image.png", file_hash(image_path))
    row["question"] = "Ignore this <answer>4</answer>"
    with pytest.raises(ValueError, match="control tokens"):
        compile_rows([row], tmp_path)


def test_compiled_manifest_is_fail_closed(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("adaptive")
    data = tmp_path / "sft.jsonl"
    rows = [{"id": "1", "think_empty": True}]
    write_jsonl(data, rows)
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {
        "contract": {"adaptive_prompt_sha256": file_hash(prompt)},
        "artifact": {"sha256": file_hash(data), "rows": 1, "empty_think": 1},
    })
    assert validate_compiled_dataset(data, manifest, prompt)[0] == rows
    data.write_text(data.read_text() + "\n")
    with pytest.raises(ValueError, match="differs"):
        validate_compiled_dataset(data, manifest, prompt)


class FakeTokenizer:
    def __init__(self, assistant):
        self.assistant = assistant

    def convert_tokens_to_ids(self, token):
        assert token == "<|image_pad|>"
        return 11

    def decode(self, ids, **kwargs):
        return self.assistant


class FakeProcessor:
    def __init__(self, assistant):
        self.assistant = assistant
        self.tokenizer = FakeTokenizer(assistant)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        if add_generation_prompt:
            return "PROMPT"
        return "PROMPT" + messages[-1]["content"]

    def __call__(self, text, images, return_tensors, padding, truncation):
        import torch

        assert padding is False and truncation is False
        ids = [10, 11, 12] if text == ["PROMPT"] else [10, 11, 12, 20, 21, 99]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            "pixel_values": torch.ones((1, 3, 2, 2)),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }


def test_assistant_only_mask_includes_empty_structure(tmp_path):
    torch = pytest.importorskip("torch")
    image_path = tmp_path / "images/example.png"
    image_path.parent.mkdir()
    Image.new("RGB", (8, 6), "white").save(image_path)
    assistant = "<think>\n\n</think>\n<answer>4</answer>"
    row = {
        "id": "sample:1",
        "image_path": "images/example.png",
        "image_sha256": file_hash(image_path),
        "question": "What is the value?",
        "assistant": assistant,
        "think_empty": True,
    }
    batch, stats = encode_training_example(FakeProcessor(assistant), row, tmp_path, "adaptive", 10, 10, 20)
    assert batch["labels"].tolist() == [[-100, -100, -100, 20, 21, 99]]
    assert stats["prompt_tokens"] == 3
    assert stats["supervised_tokens"] == 3
    assert stats["visual_tokens"] == 1


def test_lora_discovery_excludes_visual_modules():
    torch = pytest.importorskip("torch")

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(2, 2)
            self.o_proj = torch.nn.Linear(2, 2)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = Attention()
            self.visual = Attention()

    assert discover_lora_targets(Tiny(), "language_model") == [
        "language_model.o_proj",
        "language_model.q_proj",
    ]


def test_trainable_audit_rejects_base_parameter():
    torch = pytest.importorskip("torch")

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Linear(2, 1, bias=False)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.q_proj = torch.nn.Module()
            self.language_model.q_proj.lora_A = Adapter().lora_A
            self.base = torch.nn.Parameter(torch.ones(1))

    model = Tiny()
    model.base.requires_grad = False
    report = audit_trainable_parameters(model, ["language_model.q_proj"])
    assert report["trainable_parameters"] == 2
    model.base.requires_grad = True
    with pytest.raises(ValueError, match="Unexpected trainable"):
        audit_trainable_parameters(model, ["language_model.q_proj"])


def test_epoch_order_is_deterministic_and_epoch_specific():
    assert epoch_order(32, 42, 0) == epoch_order(32, 42, 0)
    assert epoch_order(32, 42, 0) != epoch_order(32, 42, 1)
    assert sorted(epoch_order(32, 42, 0)) == list(range(32))


def test_wandb_disabled_does_not_import_client(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "wandb", None)
    assert init_wandb({"wandb": {}}, "disabled", "abc", 42, True, tmp_path) is None


def test_wandb_offline_uses_deterministic_identity_without_resume(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / ".wandb-cache"))

    class FakeRun:
        def define_metric(self, *args, **kwargs):
            calls.setdefault("metrics", []).append((args, kwargs))

    def fake_init(**kwargs):
        calls["init"] = kwargs
        return FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    config = {"wandb": {"project": "bisight-rl", "group": "sft", "tags": ["drop50"]}}
    run = init_wandb(config, "offline", "a" * 64, 42, True, tmp_path)
    assert isinstance(run, FakeRun)
    assert calls["init"]["id"] == "sft-" + "a" * 20
    assert calls["init"]["mode"] == "offline"
    assert calls["init"]["tags"] == ["drop50", "formal"]
    assert "resume" not in calls["init"]
    assert calls["metrics"] == [
        (("optimizer_step",), {}),
        (("train/*",), {"step_metric": "optimizer_step"}),
    ]


def test_answer_evaluation_reports_official_and_list_aware_scores():
    row = {
        "id": "test:1",
        "source": "human",
        "answer_kind": "text",
        "question": "Which countries?",
        "answers": ["[China, USA]"],
        "canonical_answer": "[China, USA]",
    }
    result = evaluate_response(row, '<think></think><answer>["China", "USA"]</answer>', 12, 64)
    assert result["predicted_answer"] == '["China", "USA"]'
    assert result["official_correct"] is False
    assert result["list_aware_correct"] is True
    assert result["list_valued_reference"] is True
    metrics = summarize([result], "contract")
    assert metrics["official_relaxed_accuracy"]["accuracy"] == 0.0
    assert metrics["list_aware_relaxed_accuracy"]["accuracy"] == 1.0
    assert metrics["official_vs_list_aware_disagreements"] == 1


def test_evaluation_dataset_is_bound_to_manifest(tmp_path):
    data_root = tmp_path / "data"
    data = data_root / "candidates/test_full.jsonl"
    row = {
        "id": "test:1",
        "split": "test",
        "image_path": "images/example.png",
        "image_file_sha256": "abc",
        "question": "Which countries?",
        "answers": ["[China, USA]"],
        "canonical_answer": "[China, USA]",
        "source": "human",
        "data_errors": [],
    }
    write_jsonl(data, [row])
    manifest = data_root / "manifests/build.json"
    write_json(manifest, {"artifacts": {"candidates/test_full.jsonl": {"sha256": file_hash(data), "rows": 1}}})
    assert validate_evaluation_rows(data, manifest)[0] == [row]
    row["split"] = "train"
    write_jsonl(data, [row])
    write_json(manifest, {"artifacts": {"candidates/test_full.jsonl": {"sha256": file_hash(data), "rows": 1}}})
    with pytest.raises(ValueError, match="Non-test row"):
        validate_evaluation_rows(data, manifest)


def test_merge_validation_balances_actions_and_checks_outputs():
    torch = pytest.importorskip("torch")
    rows = [{"id": f"e{i}", "think_empty": True} for i in range(3)] + [
        {"id": f"f{i}", "think_empty": False} for i in range(3)
    ]
    assert [row["id"] for row in validation_rows(rows, 4)] == ["e0", "e1", "f0", "f1"]
    before = [{"id": "x", "logits": torch.tensor([[1.0, 2.0]]), "argmax": torch.tensor([1]), "generated": torch.tensor([3])}]
    after = [{"id": "x", "logits": torch.tensor([[1.1, 2.0]]), "argmax": torch.tensor([1]), "generated": torch.tensor([3])}]
    assert compare_captures(before, after, 0.2)["max_abs_logit_difference"] == pytest.approx(0.1)
    after[0]["generated"] = torch.tensor([4])
    with pytest.raises(ValueError, match="changes predictions"):
        compare_captures(before, after, 0.2)
