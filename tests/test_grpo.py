import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from bisight_rl.common import file_hash, write_jsonl
from bisight_rl.grpo.build_grpo import compile_rows
from bisight_rl.grpo.audit_grpo import audit
from bisight_rl.grpo.common import contract_digest, load_config, validate_config
from bisight_rl.grpo.reward import compute_score, decode_ground_truth, encode_ground_truth, score_response
from bisight_rl.grpo.train_grpo import apply_overrides, reconcile_rollout_log


def test_reward_components_are_independent_and_additive():
    truth = encode_ground_truth("train:1", ["Poland"])
    correct = score_response("<think>\n\n</think>\n<answer>Poland</answer>", truth)
    assert correct["format"] == 1.0
    assert correct["accuracy"] == 1.0
    assert correct["overall"] == 2.0
    assert correct["think_empty"] == 1.0

    malformed = score_response("prefix <answer>Poland</answer>", truth)
    assert malformed["format"] == 0.0
    assert malformed["accuracy"] == 1.0
    assert malformed["overall"] == 1.0

    wrong = score_response("<think>evidence</think><answer>USA</answer>", truth)
    assert wrong["format"] == 1.0
    assert wrong["accuracy"] == 0.0
    assert wrong["overall"] == 1.0


def test_reward_requires_one_answer_block_and_supports_outer_references():
    truth = encode_ground_truth("train:2", ["China", "USA"])
    assert score_response("<think></think><answer>USA</answer>", truth)["accuracy"] == 1.0
    duplicate = "<think></think><answer>China</answer><answer>USA</answer>"
    result = score_response(duplicate, truth)
    assert result["accuracy"] == 0.0
    assert result["format"] == 0.0


def test_reward_reports_group_variance_without_changing_total():
    truth = encode_ground_truth("train:variance", ["4"])
    inputs = [
        {"response": "<think></think><answer>4</answer>", "response_length": 8, "ground_truth": truth},
        {"response": "<think></think><answer>5</answer>", "response_length": 8, "ground_truth": truth},
    ]
    scores = compute_score(inputs)
    assert [item["overall"] for item in scores] == [2.0, 1.0]
    assert [item["group_reward_varies"] for item in scores] == [1.0, 1.0]
    assert [item["group_answer_varies"] for item in scores] == [1.0, 1.0]


def test_list_aware_is_diagnostic_only():
    truth = encode_ground_truth("train:3", ["[China, USA]"])
    result = score_response('<think></think><answer>["China", "USA"]</answer>', truth)
    assert result["accuracy"] == 0.0
    assert result["list_aware_accuracy"] == 1.0
    assert result["overall"] == 1.0


def test_reward_audit_is_resumable_and_contains_no_extra_reward(tmp_path):
    truth = encode_ground_truth("train:4", ["4"], source="human", answer_kind="numeric")
    audit = tmp_path / "rollouts.jsonl"
    item = {
        "response": "<think>\n\n</think><answer>4.0</answer>",
        "response_length": 12,
        "ground_truth": truth,
    }
    scores = compute_score([item], audit_path=str(audit), max_response_length=12)
    assert scores == [{
        "overall": 2.0,
        "format": 1.0,
        "accuracy": 1.0,
        "list_aware_accuracy": 1.0,
        "answer_extractable": 1.0,
        "think_empty": 1.0,
        "group_reward_varies": 0.0,
        "group_answer_varies": 0.0,
        "group_any_nonempty_think": 0.0,
    }]
    compute_score([item], audit_path=str(audit), max_response_length=12)
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [row["reward_batch"] for row in rows] == [1, 2]
    assert rows[0]["policy_version"] == 0
    assert rows[0]["hit_max_response_length"] is True


def test_rollout_audit_recomputes_group_layout_and_rewards(tmp_path):
    inputs = []
    for prompt_id in ("p1", "p2"):
        truth = encode_ground_truth(prompt_id, ["4"])
        inputs.extend(
            [
                {"response": "<think></think><answer>4</answer>", "response_length": 8, "ground_truth": truth},
                {"response": "<think>x</think><answer>5</answer>", "response_length": 8, "ground_truth": truth},
            ]
        )
    path = tmp_path / "groups.jsonl"
    compute_score(inputs, audit_path=str(path), max_response_length=32)
    report = audit([json.loads(line) for line in path.read_text().splitlines()], 2, 2)
    assert report["sampling_iterations"] == 1
    assert report["rollouts"] == 4
    assert report["informative_reward_group_rate"] == 1.0
    assert report["nonempty_think_rate"] == 0.5


def test_ground_truth_rejects_invalid_payload():
    with pytest.raises(ValueError, match="no references"):
        encode_ground_truth("train:5", [])
    with pytest.raises(ValueError, match="schema"):
        decode_ground_truth('{"id":"x","references":["1"]}')


def test_compile_grpo_row_has_no_sft_target_leakage(tmp_path):
    image = tmp_path / "images" / "chart.png"
    image.parent.mkdir()
    Image.new("RGB", (8, 6), "white").save(image)
    row = {
        "id": "chartqa:train:1",
        "split": "train",
        "image_path": "images/chart.png",
        "image_file_sha256": file_hash(image),
        "question": "Which country?",
        "answers": ["Poland", "Polska"],
        "canonical_answer": "Poland",
        "rationale": "This must never enter the RL artifact.",
        "assistant": "This must never enter the RL artifact.",
        "source": "human",
        "answer_kind": "text",
        "data_errors": [],
    }
    compiled = compile_rows([row], tmp_path, "train")
    assert set(compiled[0]) == {
        "prompt_id", "problem", "images", "answer", "image_sha256", "source", "answer_kind"
    }
    assert decode_ground_truth(compiled[0]["answer"])["references"] == ["Poland", "Polska"]
    assert "Poland" not in compiled[0]["problem"]


def test_registered_batch_mapping_and_checkpoint_policy():
    config = load_config(Path("configs/grpo_drop50.yaml"))
    batch = validate_config(config)
    assert batch == {
        "prompts_per_iteration": 16,
        "rollout_generation_chunk_prompts": 4,
        "generation_chunks_per_iteration": 4,
        "generations_per_prompt": 4,
        "trajectories_per_iteration": 64,
        "actor_prompt_minibatch": 16,
        "actor_trajectory_minibatch": 64,
        "update_micro_trajectories": 1,
        "gradient_accumulation_microbatches": 64,
        "sampling_iterations": 200,
        "total_prompt_draws": 3200,
        "total_trajectories": 12800,
    }
    assert config["trainer"]["save_freq"] == 100
    assert config["trainer"]["save_limit"] == 2


def test_p0_and_seed_overrides_use_separate_output():
    args = SimpleNamespace(
        model_path=None,
        seed=43,
        p0=True,
        max_steps=None,
        output_dir=None,
        resume_from=None,
        wandb_mode="disabled",
    )
    config = apply_overrides(load_config(Path("configs/grpo_drop50.yaml")), args)
    assert config["trainer"]["max_steps"] == 5
    assert config["bisight"]["output_root"].endswith("seed-43-p0")
    assert "wandb" not in config["trainer"]["logger"]


def test_contract_allows_logging_and_resume_path_changes():
    config = load_config(Path("configs/grpo_drop50.yaml"))
    manifest = {"artifacts": {"train": {"sha256": "a"}}, "contract": {"adaptive_prompt_sha256": "b"}}
    easy = {"commit": "c" * 40}
    expected = contract_digest(config, manifest, easy)
    config["bisight"]["wandb"]["mode"] = "disabled"
    config["trainer"]["logger"] = ["console"]
    config["trainer"]["load_checkpoint_path"] = "some/global_step_100"
    config["trainer"]["find_last_checkpoint"] = False
    assert contract_digest(config, manifest, easy) == expected


def test_resume_orphans_rollouts_newer_than_checkpoint(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    rows = [
        {"reward_batch": step, "value": index}
        for step in range(1, 4)
        for index in range(2)
    ]
    write_jsonl(path, rows)
    report = reconcile_rollout_log(path, checkpoint_step=2)
    assert report["kept_rows"] == 4
    assert report["orphaned_rows"] == 2
    assert {row["reward_batch"] for row in map(json.loads, path.read_text().splitlines())} == {1, 2}
    orphan = Path(report["orphan_path"])
    assert orphan.is_file()
    assert {row["reward_batch"] for row in map(json.loads, orphan.read_text().splitlines())} == {3}
