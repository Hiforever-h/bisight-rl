from collections import Counter

from bisight_rl.data import HammingTree, answer_kind, candidate_order, deduplicate_train, quotas, select_quota, stratum


def row(n, image=None, answer="10", question="Question", source="human"):
    return {"id": str(n), "image_id": image or str(n), "question": question, "answers": [answer],
            "source": source, "answer_kind": answer_kind(answer), "provenance": {"row_index": n}}


def test_duplicate_conflicts_and_whole_chart_isolation():
    rows = [row(0, "a"), row(1, "a"), row(2, "b", "10"), row(3, "b", "11"),
            row(4, "heldout", question="different question")]
    keep, rejected = deduplicate_train(rows, {"heldout"})
    assert [r["id"] for r in keep] == ["0"]
    assert keep[0]["duplicate_ids"] == ["0", "1"]
    assert Counter(r["reason"] for r in rejected)["conflicting_labels"] == 2


def test_capacities_and_exact_total():
    assert quotas({"a": 90, "b": 10}, 20, {"a": 5, "b": 100}) == {"a": 5, "b": 15}
    q = quotas({"a": 1, "b": 1, "c": 1}, 100, {"a": 100, "b": 100, "c": 100})
    assert sum(q.values()) == 100 and max(q.values()) - min(q.values()) == 1


def test_deterministic_order_and_sampling():
    rows = [row(i, source="human" if i < 20 else "machine") for i in range(100)]
    weights = Counter(stratum(r) for r in rows)
    first = candidate_order(rows, weights, 17)
    assert first == candidate_order(list(reversed(rows)), weights, 17)
    selected = select_quota(first, {"human:numeric": 2, "machine:numeric": 8})
    assert Counter(stratum(r) for r in selected) == {"human:numeric": 2, "machine:numeric": 8}


def test_phash_matches_bruteforce_and_keeps_collisions():
    tree = HammingTree()
    values = [(0, "a"), (1, "b"), (1, "c"), (255, "d"), (1024, "e")]
    for value, name in values:
        tree.add(value, name)
    for value, _ in values:
        for radius in (0, 1, 3, 6):
            assert sorted(tree.query(value, radius)) == sorted(((value ^ v).bit_count(), n) for v, n in values if (value ^ v).bit_count() <= radius)


def test_multivalue_and_units_preserved():
    assert answer_kind("['A', 'B']") == "list"
    assert answer_kind("35%") == "numeric"
    assert answer_kind("2019") == "numeric"
    assert answer_kind("12 million") == "text"
