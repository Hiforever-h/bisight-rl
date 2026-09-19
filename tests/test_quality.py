import pytest

from bisight_rl.quality import (
    answer_matches_any_reference,
    check_response,
    equivalent_for_supervision,
    extract_unique_answer,
    list_aware_relaxed_correctness,
    normalize_response_format,
    parse_list_answer,
    parse_response,
    relaxed_correctness,
    safe_arithmetic,
)


@pytest.mark.parametrize("target,prediction,expected", [
    ("100", "104.9", True), ("100", "105.1", False), ("-100", "-104", True),
    ("0", "0.0", False), ("0", "0", True), ("35%", "0.35", True),
    ("35%", "35", False), ("1,000", "1000", False), ("1e3", "1000", True),
    ("Yes", "yes", True), ("2019", "2020", True), ("nan", "nan", False),
    ("inf", "inf", False), ("['A', 'B']", "A", False),
])
def test_scorer_edges(target, prediction, expected):
    assert relaxed_correctness(target, prediction) == expected


def test_relaxed_answer_does_not_certify_supervision():
    assert relaxed_correctness("100", "104")
    assert not equivalent_for_supervision("100", "104")
    assert equivalent_for_supervision("0", "0.0")
    q = check_response("<think>The chart reads 104.</think><answer>104</answer>", "100", "stop")
    assert not q["auto_pass"]


@pytest.mark.parametrize("text,expected", [
    ('["China", "USA"]', ["China", "USA"]),
    ("['China', 'USA']", ["China", "USA"]),
    ("[China, USA]", ["China", "USA"]),
    ("[2014, 2016]", ["2014", "2016"]),
    ("China, USA", None),
])
def test_parse_chartqa_list_answers(text, expected):
    assert parse_list_answer(text) == expected


def test_list_aware_relaxed_accuracy_is_ordered_and_representation_tolerant():
    assert not relaxed_correctness("[China, USA]", '["China", "USA"]')
    assert list_aware_relaxed_correctness("[China, USA]", '["China", "USA"]')
    assert list_aware_relaxed_correctness("[100, 200]", "[104, 190]")
    assert not list_aware_relaxed_correctness("[China, USA]", "[USA, China]")
    assert not list_aware_relaxed_correctness("[China, USA]", "[China]")
    assert answer_matches_any_reference(["UK", "United Kingdom"], "united kingdom")


def test_answer_only_extraction_ignores_think_format_but_rejects_ambiguity():
    assert extract_unique_answer("prefix <answer> Poland </answer> suffix") == "Poland"
    assert extract_unique_answer("<answer>Poland</answer><answer>Poland</answer>") is None
    assert extract_unique_answer("<think>x</think>") is None


@pytest.mark.parametrize("response", [
    "<think>x</think><answer>1</answer><answer>1</answer>",
    "<think><think>x</think></think><answer>1</answer>",
    "<think>x</think><answer></answer>", "<think>x</think><answer>1",
    "<think>x</think><answer>1</answer> trailing",
])
def test_rejects_ambiguous_format(response):
    assert parse_response(response) is None


def test_repairs_only_unambiguous_missing_think_close():
    malformed = "<think>Read 7 from the bar.<answer>7</answer>"
    repaired, repairs = normalize_response_format(malformed)
    assert repaired == "<think>Read 7 from the bar.</think><answer>7</answer>"
    assert repairs == ["insert_missing_think_close_before_answer"]
    assert check_response(repaired, "7", "stop")["auto_pass"]


@pytest.mark.parametrize("response", [
    "<think>truncated",
    "prefix<think>x<answer>1</answer>",
    "<think>x<answer>1</answer><answer>1</answer>",
    "<think>x</think><answer>1</answer>",
])
def test_does_not_repair_ambiguous_or_complete_responses(response):
    assert normalize_response_format(response) == (response, [])


def test_empty_think_valid_for_student_but_not_master():
    text = "<think>\n\n</think><answer>2</answer>"
    assert parse_response(text)["rationale"] == "\n\n"
    assert not check_response(text, "2", "stop")["auto_pass"]


def test_arithmetic_and_leakage():
    assert check_response("<think>The two bars are 3 and 4; 3 + 4 = 7.</think><answer>7</answer>", "7", "stop")["auto_pass"]
    assert not check_response("<think>3 + 4 = 9</think><answer>9</answer>", "9", "stop")["auto_pass"]
    assert not check_response("<think>The provided answer is 7.</think><answer>7</answer>", "7", "stop")["auto_pass"]
    assert not check_response("<think>The target answer is 7.</think><answer>7</answer>", "7", "stop")["auto_pass"]
    assert not check_response("<think>UNSUPPORTED</think><answer>UNSUPPORTED</answer>", "7", "stop")["auto_pass"]
    assert not check_response("<think>3 + 4 = 7</think><answer>7</answer>", "7", "length")["auto_pass"]
    with pytest.raises(ValueError):
        safe_arithmetic("__import__('os').system('whoami')")
