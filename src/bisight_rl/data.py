from __future__ import annotations

import ast
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict


def normalize_question(text):
    # Deliberately preserve case, punctuation, units, and numbers.
    return " ".join(unicodedata.normalize("NFC", text).split())


def answer_kind(answer):
    try:
        parsed = ast.literal_eval(answer)
        if isinstance(parsed, (list, tuple)):
            return "list"
    except (ValueError, SyntaxError):
        pass
    value = answer.strip().replace(",", "")
    if value.endswith("%"):
        value = value[:-1]
    try:
        if math.isfinite(float(value)):
            return "numeric"
    except ValueError:
        pass
    return "text"


def stratum(row):
    return row["source"] + ":" + row["answer_kind"]


def quotas(weights, total, capacities):
    """Largest remainder allocation, with explicit capacity-limited redistribution."""
    result = {key: 0 for key in sorted(capacities)}
    if sum(capacities.values()) < total:
        raise ValueError(f"Only {sum(capacities.values())} eligible rows for target {total}")
    while sum(result.values()) < total:
        remaining = total - sum(result.values())
        active = [k for k in result if result[k] < capacities[k]]
        weight_sum = sum(weights.get(k, 0) for k in active)
        shares = {k: remaining * (weights.get(k, 0) / weight_sum if weight_sum else 1 / len(active)) for k in active}
        for key in active:
            result[key] += min(capacities[key] - result[key], math.floor(shares[key]))
        spare = total - sum(result.values())
        for key in sorted(active, key=lambda k: (-(shares[k] % 1), k)):
            if spare and result[key] < capacities[key]:
                result[key] += 1
                spare -= 1
    return result


def candidate_order(rows, weights, seed):
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["id"]):
        groups[stratum(row)].append(row)
    rng = random.Random(seed)
    for key in sorted(groups):
        rng.shuffle(groups[key])
    emitted = Counter()
    ordered = []
    while groups:
        key = min(groups, key=lambda k: ((emitted[k] + 1) / max(weights.get(k, 1), 1), k))
        ordered.append(groups[key].pop())
        emitted[key] += 1
        if not groups[key]:
            del groups[key]
    return ordered


def select_quota(rows, targets):
    selected = []
    counts = Counter()
    for row in rows:
        key = stratum(row)
        if counts[key] < targets.get(key, 0):
            selected.append(row)
            counts[key] += 1
    if sum(counts.values()) != sum(targets.values()):
        raise ValueError("Not enough eligible rows to fill frozen quotas")
    return selected


class HammingTree:
    """BK-tree for 64-bit perceptual hashes; values may share a hash."""

    def __init__(self):
        self.root = None

    def add(self, value, item):
        if self.root is None:
            self.root = [value, [item], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(item)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [item], {}]
                return
            node = node[2][distance]

    def query(self, value, radius):
        stack = [self.root] if self.root else []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                for item in node[1]:
                    yield distance, item
            stack.extend(child for edge, child in node[2].items() if distance - radius <= edge <= distance + radius)


def deduplicate_train(rows, excluded_images):
    groups = defaultdict(list)
    rejected = []
    for row in rows:
        reason = None
        if row.get("data_errors"):
            reason = "invalid_or_unresolved_answer"
        elif row["image_id"] in excluded_images:
            reason = "cross_split_image_overlap_or_unresolved_near_duplicate"
        if reason:
            rejected.append({"id": row["id"], "reason": reason})
        else:
            groups[(row["image_id"], normalize_question(row["question"]))].append(row)
    kept = []
    for group in groups.values():
        labels = {tuple(r["answers"]) for r in group}
        if len(labels) != 1:
            rejected.extend({"id": r["id"], "reason": "conflicting_labels", "group_ids": [x["id"] for x in group]} for r in group)
            continue
        group.sort(key=lambda r: r["provenance"]["row_index"])
        representative = dict(group[0], duplicate_ids=[r["id"] for r in group])
        kept.append(representative)
        rejected.extend({"id": r["id"], "reason": "duplicate_qa", "kept_id": representative["id"]} for r in group[1:])
    return kept, rejected

