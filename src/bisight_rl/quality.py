"""Conservative checks; a pass is not a certification of visual correctness."""
from __future__ import annotations

import ast
import math
import operator
import re

FORMAT = re.compile(r"\A\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*\Z", re.DOTALL)
CONTROL = re.compile(r"</?(?:think|answer)>")
LEAK = re.compile(r"\b(?:provided|given|supplied|reference|target|ground[- ]truth|known correct)\s+(?:correct\s+)?answer\b|\banswer\s+(?:provided|given|supplied)\b|\bprivate constraint\b|标准答案|已知答案", re.I)
EQUATION = re.compile(r"(?<![\w.,])([-+]?\d+(?:\.\d+)?(?:\s*[-+*/×÷]\s*[-+]?\d+(?:\.\d+)?)+)\s*=\s*([-+]?\d+(?:\.\d+)?)(?![\w.,%])")
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def parse_response(text):
    m = FORMAT.fullmatch(text)
    if not m:
        return None
    think, answer = m.group("think"), m.group("answer").strip()
    if not answer or CONTROL.search(think) or CONTROL.search(answer):
        return None
    return {"rationale": think, "answer": answer}


def _number(text):
    try:
        return float(text[:-1]) / 100 if text.endswith("%") else float(text)
    except (ValueError, TypeError):
        return None


def relaxed_correctness(target, prediction):
    """Pix2Struct-compatible finite ChartQA answers; upstream zero behavior retained.

    Nonfinite numbers are explicitly rejected as invalid outputs. Outer whitespace
    is removed by the answer-block parser, not here. No comma/unit normalization.
    """
    a, b = _number(target), _number(prediction)
    if any(x is not None and not math.isfinite(x) for x in (a, b)):
        return False
    if b is not None and a:
        return abs(b - a) / abs(a) <= 0.05
    return prediction.lower() == target.lower()


def equivalent_for_supervision(target, prediction):
    a, b = _number(target.strip()), _number(prediction.strip())
    if a is not None and b is not None:
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return target.strip().lower() == prediction.strip().lower()


def safe_arithmetic(expr):
    if len(expr) > 200:
        raise ValueError("Expression too long")
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError("Unsupported arithmetic")
    return visit(ast.parse(expr.replace("×", "*").replace("÷", "/"), mode="eval").body)


def check_response(text, canonical_answer, finish_reason):
    errors, arithmetic = [], []
    parsed = parse_response(text)
    if finish_reason != "stop":
        errors.append("truncated_or_nonstop_finish")
    if parsed is None:
        return {"auto_pass": False, "errors": errors + ["invalid_format"], "arithmetic_checks": []}
    rationale = parsed["rationale"]
    if not rationale.strip():
        errors.append("empty_rationale")
    if LEAK.search(rationale):
        errors.append("answer_hint_reference")
    if rationale.strip() == "UNSUPPORTED" or parsed["answer"].strip() == "UNSUPPORTED":
        errors.append("unsupported_chart_evidence")
    if any(token in text for token in ("<|im_start|>", "<|im_end|>", "[INST]")):
        errors.append("role_token_echo")
    if not equivalent_for_supervision(canonical_answer, parsed["answer"]):
        errors.append("conclusion_differs_from_reference")
    for m in EQUATION.finditer(rationale):
        try:
            actual, stated = safe_arithmetic(m[1]), float(m[2])
            # Allow rounding to the precision actually written in the explanation.
            decimals = len(m[2].split(".")[1]) if "." in m[2] else 0
            ok = math.isclose(actual, stated, rel_tol=1e-9, abs_tol=0.5 * 10 ** (-decimals) + 1e-10)
            arithmetic.append({"expression": m[1], "stated": m[2], "calculated": actual, "ok": ok})
        except (ValueError, ZeroDivisionError, OverflowError, SyntaxError):
            arithmetic.append({"expression": m[1], "ok": False})
    if any(not item["ok"] for item in arithmetic):
        errors.append("arithmetic_inconsistency")
    return {"auto_pass": not errors, "errors": errors, "arithmetic_checks": arithmetic,
            "arithmetic_status": "checked_subset" if arithmetic else "not_verifiable",
            "visual_status": "not_human_verified",
            "reference_relaxed_match": relaxed_correctness(canonical_answer, parsed["answer"]), **parsed}
