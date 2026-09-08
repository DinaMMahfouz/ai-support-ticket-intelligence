"""
Scores an escalation-judgment predictor against data/hard_eval_set.csv,
the only labeled escalation data that exists. Prints accuracy,
precision/recall/F1 for the "escalate" class, and a confusion matrix -
twice: once for the model's raw verdict, once for the final decision
after escalation_policy.apply_policy() (the confidence-threshold safety
net) is applied to the same responses. Each ticket is only called once
per provider; the policy is applied to the response already fetched,
not by making a second round of API calls.

    python -m evaluation.escalation_eval_runner --model ollama
    python -m evaluation.escalation_eval_runner --model anthropic
    python -m evaluation.escalation_eval_runner --model both

This is a separate, small runner rather than a mode of eval_runner.py:
eval_runner.py scores 4-class priority against a label space defined by
LabelEncoder; this scores binary escalate yes/no against ground truth
that lives directly in hard_eval_set.csv, with no shared machinery
between the two - forcing them into one tool would mean more branching
for no reuse.
"""

import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ticket_intelligence.llm.escalation_policy import apply_policy
from ticket_intelligence.priority_features import HARD_EVAL_PATH

ESCALATE_LABELS = ["yes", "no"]


def _call_ollama(row) -> dict:
    from ticket_intelligence.llm.escalation_llm import assess_escalation

    return assess_escalation(
        subject=row["subject"],
        description=row["description"],
        category=row["category"],
        stated_priority=row["priority"]
    )


def _call_anthropic(row) -> dict:
    from ticket_intelligence.llm.escalation_llm_anthropic import assess_escalation_anthropic

    return assess_escalation_anthropic(
        subject=row["subject"],
        description=row["description"],
        category=row["category"],
        stated_priority=row["priority"]
    )


CALLERS = {
    "ollama": _call_ollama,
    "anthropic": _call_anthropic
}


def _print_metrics(label: str, y_true, y_pred, ticket_ids) -> None:
    print(f"\n--- {label} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, labels=ESCALATE_LABELS, zero_division=0))
    print(f"Confusion matrix (rows=true, cols=pred), labels = {ESCALATE_LABELS}")
    print(confusion_matrix(y_true, y_pred, labels=ESCALATE_LABELS))

    misses = [
        f"{tid}: true={t} predicted={p}"
        for tid, t, p in zip(ticket_ids, y_true, y_pred)
        if t != p
    ]
    if misses:
        print("Misclassified:")
        for line in misses:
            print(f"  {line}")


def run_eval(model_name: str) -> None:
    df = pd.read_csv(HARD_EVAL_PATH)
    y_true = df["escalate"].str.lower()

    print(f"\n=== escalation judgment: {model_name} ({HARD_EVAL_PATH.name}, n={len(df)}) ===")

    raw_preds = []
    policy_preds = []
    policy_reasons = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        response = CALLERS[model_name](row)
        raw_preds.append(response["escalate"])

        policy_result = apply_policy(response)
        policy_preds.append("yes" if policy_result["escalate"] else "no")
        policy_reasons.append(policy_result["reason"])

        flip = " (POLICY FLIPPED)" if raw_preds[-1] != policy_preds[-1] else ""
        print(
            f"  [{i}/{len(df)}] {row['ticket_id']} -> "
            f"raw={response['escalate']} ({response['confidence']}) "
            f"policy={policy_preds[-1]} [{policy_result['reason']}]{flip}",
            flush=True
        )

    _print_metrics("raw model verdict", y_true, raw_preds, df["ticket_id"])
    _print_metrics("after confidence-threshold policy", y_true, policy_preds, df["ticket_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["ollama", "anthropic", "both"],
        required=True,
        help="Which escalation predictor to score."
    )
    args = parser.parse_args()

    model_names = list(CALLERS) if args.model == "both" else [args.model]

    for model_name in model_names:
        run_eval(model_name)


if __name__ == "__main__":
    main()
