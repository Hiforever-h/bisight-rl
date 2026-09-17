import pytest

from bisight_rl.quality import check_response, equivalent_for_supervision, parse_response, relaxed_correctness, safe_arithmetic


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


@pytest.mark.parametrize("response", [
    "<think>x</think><answer>1</answer><answer>1</answer>",
    "<think><think>x</think></think><answer>1</answer>",
    "<think>x</think><answer></answer>", "<think>x</think><answer>1",
    "<think>x</think><answer>1</answer> trailing",
])
def test_rejects_ambiguous_format(response):
    assert parse_response(response) is None


def test_empty_think_valid_for_student_but_not_master():
    text = "<think>\n\n</think><answer>2</answer>"
    assert parse_response(text)["rationale"] == "\n\n"
    assert not check_response(text, "2", "stop")["auto_pass"]


def test_arithmetic_and_leakage():
    assert check_response("<think>The two bars are 3 and 4; 3 + 4 = 7.</think><answer>7</answer>", "7", "stop")["auto_pass"]
    assert not check_response("<think>3 + 4 = 9</think><answer>9</answer>", "9", "stop")["auto_pass"]
    assert not check_response("<think>The provided answer is 7.</think><answer>7</answer>", "7", "stop")["auto_pass"]
    assert not check_response("<think>3 + 4 = 7</think><answer>7</answer>", "7", "length")["auto_pass"]
    with pytest.raises(ValueError):
        safe_arithmetic("__import__('os').system('whoami')")
