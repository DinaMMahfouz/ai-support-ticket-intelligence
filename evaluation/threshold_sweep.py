"""
Turns the priority model's confidence into a human-escalation decision,
and measures what that costs on data/hard_eval_set.csv.

    python -m evaluation.threshold_sweep --model anthropic
    python -m evaluation.threshold_sweep --model ollama
    python -m evaluation.threshold_sweep --model stub      # no provider needed

The rule under test is deliberately the simplest one that could work:
escalate when the model's self-reported confidence falls below a
threshold. The 18 hard cases already carry a ground-truth `escalate`
column, so the rule can be scored rather than asserted.

Three numbers, and they trade against each other:

  Correct escalations  it was unsure on a case that genuinely needed a
                       human (HARD-001, HARD-007: urgent-sounding
                       tickets with no actionable detail).

  Missed escalations   it answered confidently on a case that needed a
                       human. This is the dangerous class. HARD-018
                       (successful auth from an unrecognised IP, phrased
                       as a calm question) and HARD-016 (customer says
                       "no action needed" about an unexplained outage)
                       exist specifically to catch this.

  Over-escalation      it punted on a case it should have handled. A
                       system that escalates everything is perfectly
                       safe and completely useless - on this file,
                       always-escalate scores 6/6 on the escalate class
                       while being wrong on all 12 others.

Every ticket is predicted once and the thresholds are swept over the
stored confidences, so the sweep costs no extra provider calls.

Two honest caveats, both structural:

hard_eval_set.csv has no environment / channel / region / customer_tier
columns - the tickets genuinely lack that context, which is part of
what makes them hard. They are sent as "unspecified". That is a real
difference from how the model is scored elsewhere and it is stated in
the output, not hidden.

n = 18. Six of them are escalate=yes. A single case moving changes a
rate by 5-8 points. Treat the shape of the curve as the finding, never
a specific decimal.
"""

import argparse
import hashlib

import pandas as pd

from ticket_intelligence.guardrails import describe_findings, find_invented_details
from ticket_intelligence.llm.priority_validation import (
    PriorityValidationError,
    parse_stats,
    reset_parse_stats
)
from ticket_intelligence.priority_features import HARD_EVAL_PATH

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# The hard cases have no metadata columns. One placeholder string,
# named so it is obvious in any log that it was not real input.
MISSING_METADATA = "unspecified"


def _predict_ollama(row) -> dict:
    from ticket_intelligence.llm.priority_llm import predict_priority_llm

    return predict_priority_llm(
        subject=row["subject"],
        description=row["description"],
        category=row["category"],
        environment=MISSING_METADATA,
        channel=MISSING_METADATA,
        region=MISSING_METADATA,
        customer_tier=MISSING_METADATA
    )


def _predict_anthropic(row) -> dict:
    from ticket_intelligence.llm.priority_llm_anthropic import predict_priority_llm_anthropic

    return predict_priority_llm_anthropic(
        subject=row["subject"],
        description=row["description"],
        category=row["category"],
        environment=MISSING_METADATA,
        channel=MISSING_METADATA,
        region=MISSING_METADATA,
        customer_tier=MISSING_METADATA
    )


def _predict_stub(row) -> dict:
    """
    A deterministic fake predictor. No provider, no network.

    It exists so the sweep arithmetic and the reporting can be tested
    in CI - the tests assert on known inputs, and a real model would
    make those assertions non-deterministic. Confidence is derived from
    a hash of the subject so it is stable across runs but uncorrelated
    with anything meaningful, which is exactly what you want from a
    control: any structure the sweep reports for the stub is structure
    the sweep invented.
    """

    digest = hashlib.sha256(row["subject"].encode("utf-8")).hexdigest()

    return {
        "priority": ["Critical", "High", "Medium", "Low"][int(digest[:2], 16) % 4],
        "confidence": round(0.40 + (int(digest[2:4], 16) / 255) * 0.55, 3),
        "reasoning": "Deterministic stub prediction for testing the sweep."
    }


PREDICTORS = {
    "ollama": _predict_ollama,
    "anthropic": _predict_anthropic,
    "stub": _predict_stub
}


def collect_predictions(predictor) -> pd.DataFrame:
    """
    Predict every hard case once. A ticket whose response never
    validates is recorded with confidence None and is treated as an
    escalation at every threshold - failing closed, consistent with
    priority_validation.py refusing to default a label.
    """

    df = pd.read_csv(HARD_EVAL_PATH)
    rows = []

    for _, row in df.iterrows():
        record = {
            "ticket_id": row["ticket_id"],
            "true_escalate": str(row["escalate"]).strip().lower() == "yes",
            "chars": len(str(row["description"])),
            "subject": row["subject"]
        }

        try:
            result = predictor(row)
            record["priority"] = result["priority"]
            record["confidence"] = float(result["confidence"])
            record["reasoning"] = result["reasoning"]
            record["validation_failed"] = False
        except PriorityValidationError as exc:
            record.update(
                priority=None, confidence=None, reasoning=str(exc),
                validation_failed=True
            )

        ticket_text = f"{row['subject']} {row['description']} {row['category']}"
        record["invented"] = find_invented_details(ticket_text, record.get("reasoning") or "")
        rows.append(record)

        print(
            f"  {record['ticket_id']}  conf="
            f"{'FAILED' if record['validation_failed'] else format(record['confidence'], '.2f')}"
            f"  priority={record['priority']}",
            flush=True
        )

    return pd.DataFrame(rows)


def score_threshold(predictions: pd.DataFrame, threshold: float) -> dict:
    """A validation failure escalates, whatever the threshold."""

    escalated = predictions["validation_failed"] | (predictions["confidence"] < threshold)
    truth = predictions["true_escalate"]

    correct = int((escalated & truth).sum())
    missed = int((~escalated & truth).sum())
    over = int((escalated & ~truth).sum())
    n_should_not = int((~truth).sum())

    return {
        "threshold": threshold,
        "correct_escalations": correct,
        "missed_escalations": missed,
        "over_escalations": over,
        "over_escalation_rate": over / n_should_not if n_should_not else 0.0,
        "missed_ids": list(predictions.loc[~escalated & truth, "ticket_id"])
    }


def print_sweep(predictions: pd.DataFrame) -> None:
    should_escalate = int(predictions["true_escalate"].sum())
    total = len(predictions)

    print(f"\n=== threshold sweep (n={total}, {should_escalate} need a human) ===")
    print(f"{'thresh':>7}  {'caught':>6}  {'MISSED':>6}  {'over':>5}  {'over-rate':>9}   missed cases")

    for threshold in THRESHOLDS:
        row = score_threshold(predictions, threshold)
        print(
            f"{row['threshold']:>7.2f}  {row['correct_escalations']:>6}"
            f"  {row['missed_escalations']:>6}  {row['over_escalations']:>5}"
            f"  {row['over_escalation_rate']:>9.0%}   {', '.join(row['missed_ids']) or '-'}"
        )

    print(
        f"\nBaselines: never-escalate misses all {should_escalate}; "
        f"always-escalate catches all {should_escalate} at "
        f"{(total - should_escalate) / (total - should_escalate):.0%} over-escalation "
        f"({total - should_escalate} of {total - should_escalate} non-cases)."
    )


def print_length_confound(predictions: pd.DataFrame) -> None:
    """
    The check worth running before celebrating.

    If the system escalates HARD-018 (the calm security question), that
    could mean it understood the security implication - or just that
    the ticket is short and short tickets get low confidence. Those look
    identical in the metrics above and are completely different
    capabilities.

    A strong positive correlation between description length and
    confidence means the threshold is largely a length detector wearing
    a judgement costume.
    """

    usable = predictions[~predictions["validation_failed"]]
    if len(usable) < 3:
        print("\n=== length confound: not enough validated predictions to check ===")
        return

    correlation = usable["chars"].corr(usable["confidence"], method="spearman")

    print("\n=== length confound check ===")
    print(f"Spearman correlation, description length vs confidence: {correlation:+.2f}")
    print(
        "  |r| > 0.5 means confidence is tracking ticket length more than content,\n"
        "  and any 'correct' escalation of a short ticket should be read as a\n"
        "  length artefact until shown otherwise."
    )

    print(f"\n{'ticket':<10}{'chars':>6}{'conf':>7}  needs human")
    for _, row in usable.sort_values("chars").iterrows():
        print(
            f"{row['ticket_id']:<10}{row['chars']:>6}{row['confidence']:>7.2f}"
            f"  {'YES' if row['true_escalate'] else 'no'}"
        )


def print_guardrails(predictions: pd.DataFrame, threshold: float) -> None:
    stats = parse_stats()
    chosen = score_threshold(predictions, threshold)
    fabricating = predictions[predictions["invented"].map(len) > 0]

    print("\n=== guardrails ===")
    if stats["calls"]:
        print(f"parse-failure rate      {stats['parse_failure_rate']:.1%} "
              f"({stats['parse_failures']} failed attempts over {stats['calls']} tickets)")
    else:
        print("parse-failure rate      n/a (the stub predictor bypasses validation)")
    print(f"hard failures           {stats['hard_failures']} "
          f"(no valid response after retry; each escalates)")
    print(f"chosen threshold        {threshold:.2f}")
    print(f"missed escalations      {chosen['missed_escalations']} "
          f"{chosen['missed_ids'] if chosen['missed_ids'] else ''}")
    print(f"over-escalation rate    {chosen['over_escalation_rate']:.0%}")
    print(f"invented-detail flags   {len(fabricating)} of {len(predictions)} responses")

    for _, row in fabricating.iterrows():
        print(f"  {row['ticket_id']}: {describe_findings(row['invented'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(PREDICTORS), required=True)
    parser.add_argument(
        "--threshold", type=float, default=0.70,
        help="Threshold reported in the guardrails section (default 0.70)."
    )
    args = parser.parse_args()

    reset_parse_stats()

    print(f"=== confidence-based escalation: {args.model} ({HARD_EVAL_PATH.name}) ===")
    print(f"Note: this file has no environment/channel/region/customer_tier columns; "
          f"they are sent as '{MISSING_METADATA}'.\n")

    predictions = collect_predictions(PREDICTORS[args.model])

    print_sweep(predictions)
    print_length_confound(predictions)
    print_guardrails(predictions, args.threshold)


if __name__ == "__main__":
    main()
