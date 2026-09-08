"""
The sweep arithmetic, checked against known inputs.

Escalation metrics are easy to get subtly wrong - swapping a missed
escalation for an over-escalation inverts the safety story while
leaving the totals intact. These use hand-built frames so every count
is checkable by eye.
"""

import pandas as pd

from evaluation.threshold_sweep import score_threshold


def _frame(rows):
    return pd.DataFrame([
        {"ticket_id": tid, "true_escalate": truth, "confidence": conf,
         "validation_failed": False, "chars": 100}
        for tid, truth, conf in rows
    ])


def test_counts_are_assigned_to_the_right_class():
    predictions = _frame([
        ("A", True, 0.40),    # needs a human, low confidence  -> caught
        ("B", True, 0.95),    # needs a human, high confidence -> MISSED
        ("C", False, 0.30),   # fine alone, low confidence     -> over-escalation
        ("D", False, 0.90),   # fine alone, high confidence    -> correct silence
    ])

    result = score_threshold(predictions, 0.70)

    assert result["correct_escalations"] == 1
    assert result["missed_escalations"] == 1
    assert result["missed_ids"] == ["B"]
    assert result["over_escalations"] == 1
    assert result["over_escalation_rate"] == 0.5   # 1 of 2 non-cases


def test_threshold_zero_never_escalates():
    predictions = _frame([("A", True, 0.10), ("B", False, 0.10)])
    result = score_threshold(predictions, 0.0)
    assert result["correct_escalations"] == 0
    assert result["missed_escalations"] == 1
    assert result["over_escalations"] == 0


def test_threshold_one_escalates_everything():
    """The useless-but-safe baseline: catches all, over-escalates all."""
    predictions = _frame([("A", True, 0.99), ("B", False, 0.99)])
    result = score_threshold(predictions, 1.0)
    assert result["missed_escalations"] == 0
    assert result["over_escalation_rate"] == 1.0


def test_validation_failure_escalates_at_every_threshold():
    predictions = pd.DataFrame([
        {"ticket_id": "A", "true_escalate": True, "confidence": None,
         "validation_failed": True, "chars": 10},
    ])
    for threshold in (0.0, 0.5, 1.0):
        assert score_threshold(predictions, threshold)["correct_escalations"] == 1


def test_raising_the_threshold_never_increases_misses():
    """Monotonicity - a sanity property the real curve must also have."""
    predictions = _frame([
        ("A", True, 0.30), ("B", True, 0.60), ("C", True, 0.85),
        ("D", False, 0.45), ("E", False, 0.75),
    ])
    misses = [score_threshold(predictions, t)["missed_escalations"]
              for t in (0.5, 0.6, 0.7, 0.8, 0.9)]
    assert misses == sorted(misses, reverse=True)
