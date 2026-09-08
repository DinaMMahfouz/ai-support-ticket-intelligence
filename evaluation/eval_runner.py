"""
Scores a priority predictor - the classical baseline, the local-Ollama
LLM path, or the hosted-Anthropic LLM path - against a labeled eval CSV
and prints accuracy, a per-class classification report, and a
confusion matrix.

    python -m evaluation.eval_runner --model classical --set holdout
    python -m evaluation.eval_runner --model classical --set all
    python -m evaluation.eval_runner --model llm-ollama --set llm_sample
    python -m evaluation.eval_runner --model llm-anthropic --set llm_sample

Results for different eval sets are never merged into one number -
"--set all" prints one separate, clearly labeled report per set.

Before scoring, validates that the eval file actually has the columns
the predictor needs and that its category/priority values exist in the
label space the model was trained on. If not, this prints exactly what's
wrong and skips scoring that set - it never drops rows, remaps labels,
or reports a number computed against data the predictor wasn't built to
handle.

"hard" (data/hard_eval_set.csv) is expected to always skip against the
classical model, by design, not as an unresolved bug: it uses a
Sev1-4 severity scale and a 14-value category taxonomy that share only
partial overlap with this model's training labels, and it omits the
environment/channel/region/customer_tier columns the model requires.
That's because the file is the guardrail/escalation eval set for the
LLM path (its `escalate` column, not `priority`, is the intended ground
truth) - it isn't meant to be forced through a metadata-heavy classical
classifier it was never designed to feed. See docs/INTERVIEW_NOTES.md.

"llm_sample" (data/eval_holdout_llm_sample_v1.csv) is a frozen 240-row
stratified sample of the eval holdout, used for the classical-vs-LLM
head-to-head - see make_llm_eval_sample.py for why it's a sample rather
than the full 2,400 rows.
"""

import argparse

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ticket_intelligence.priority_features import (
    MODEL_PATH,
    LABEL_ENCODER_PATH,
    EVAL_HOLDOUT_PATH,
    EVAL_HOLDOUT_LLM_SAMPLE_PATH,
    HARD_EVAL_PATH,
    FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    TEXT_COLUMN,
    TARGET_COLUMN,
    load_ticket_csv
)

EVAL_SETS = {
    "holdout": EVAL_HOLDOUT_PATH,
    "hard": HARD_EVAL_PATH,
    "llm_sample": EVAL_HOLDOUT_LLM_SAMPLE_PATH
}


def check_scoring_compatibility(df, label_encoder, model=None):
    """
    Collect every reason this file can't be scored, rather than
    stopping at the first one - so a run reports the full picture in
    one shot instead of surfacing problems one at a time.

    label_encoder defines the task's ground-truth label space and
    applies to any predictor. The known-category check is specific to
    the classical model's fitted OneHotEncoder, so it only runs when
    `model` is given.
    """

    problems = []

    required_columns = [c for c in FEATURE_COLUMNS if c != TEXT_COLUMN]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        problems.append(
            f"missing required feature column(s): {missing_columns}"
        )

    if TARGET_COLUMN not in df.columns:
        problems.append(f"missing target column: '{TARGET_COLUMN}'")
    else:
        known_priorities = set(label_encoder.classes_)
        seen_priorities = set(df[TARGET_COLUMN].dropna().unique())
        unknown_priorities = sorted(seen_priorities - known_priorities)
        if unknown_priorities:
            problems.append(
                f"'{TARGET_COLUMN}' values not in the training label set "
                f"{sorted(known_priorities)}: {unknown_priorities}"
            )

    if model is not None and "category" in df.columns:
        category_encoder = model.named_steps["preprocessor"].named_transformers_["categorical"]
        known_categories = set(
            category_encoder.categories_[CATEGORICAL_COLUMNS.index("category")]
        )
        seen_categories = set(df["category"].dropna().unique())
        unknown_categories = sorted(seen_categories - known_categories)
        if unknown_categories:
            problems.append(
                f"'category' values not in the training label set "
                f"{sorted(known_categories)}: {unknown_categories}"
            )

    if problems:
        raise ValueError(
            "incompatible with this predictor:\n    "
            + "\n    ".join(problems)
        )


def make_classical_predict_fn(model, label_encoder):
    """Wraps the classical pipeline's vectorized .predict() into the
    common predict_fn(df) -> list[str] interface."""

    def predict(df):
        encoded = model.predict(df[FEATURE_COLUMNS])
        return list(label_encoder.inverse_transform(encoded))

    return predict


def _make_row_by_row_predict_fn(predict_one):
    """
    Shared loop for any LLM provider: calls predict_one(row) once per
    ticket and prints progress, since these calls are slow enough
    (seconds each) that a silent multi-minute run would look hung.
    """

    def predict(df):
        predictions = []
        total = len(df)
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            result = predict_one(
                subject=row["subject"],
                description=row["description"],
                category=row["category"],
                environment=row["environment"],
                channel=row["channel"],
                region=row["region"],
                customer_tier=row["customer_tier"]
            )
            predictions.append(result["priority"])
            print(f"  [{i}/{total}] predicted {result['priority']}", flush=True)
        return predictions

    return predict


def make_llm_ollama_predict_fn():
    """
    Wraps priority_llm.predict_priority_llm (one local Ollama call per
    row). Imported lazily so that scoring the classical model never
    requires Ollama to be running.
    """

    from ticket_intelligence.llm.priority_llm import predict_priority_llm

    return _make_row_by_row_predict_fn(predict_priority_llm)


def make_llm_anthropic_predict_fn():
    """
    Wraps priority_llm_anthropic.predict_priority_llm_anthropic (one
    Anthropic API call per row). Imported lazily so that scoring the
    classical model or the Ollama model never requires
    ANTHROPIC_API_KEY to be set.
    """

    from ticket_intelligence.llm.priority_llm_anthropic import predict_priority_llm_anthropic

    return _make_row_by_row_predict_fn(predict_priority_llm_anthropic)


def run_eval(csv_path, predict_fn, label_encoder, set_name, model=None):
    df = load_ticket_csv(csv_path)

    check_scoring_compatibility(df, label_encoder, model=model)

    y_true = df[TARGET_COLUMN]

    print(f"\n=== {set_name} ({csv_path.name}, n={len(df)}) ===")
    y_pred = predict_fn(df)

    print("\nAccuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=label_encoder.classes_, zero_division=0))
    print(f"Confusion matrix (rows=true, cols=pred), labels = {list(label_encoder.classes_)}")
    print(confusion_matrix(y_true, y_pred, labels=label_encoder.classes_))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        choices=sorted(EVAL_SETS) + ["all"],
        required=True,
        help="Which eval set to score against."
    )
    parser.add_argument(
        "--model",
        choices=["classical", "llm-ollama", "llm-anthropic"],
        default="classical",
        help="Which predictor to score (default: classical)."
    )
    args = parser.parse_args()

    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    if args.model == "classical":
        model = joblib.load(MODEL_PATH)
        predict_fn = make_classical_predict_fn(model, label_encoder)
    elif args.model == "llm-ollama":
        model = None
        predict_fn = make_llm_ollama_predict_fn()
    else:
        model = None
        predict_fn = make_llm_anthropic_predict_fn()

    set_names = list(EVAL_SETS) if args.set == "all" else [args.set]

    for set_name in set_names:
        csv_path = EVAL_SETS[set_name]
        try:
            run_eval(csv_path, predict_fn, label_encoder, set_name, model=model)
        except ValueError as exc:
            print(f"\n=== {set_name} ({csv_path.name}) [{args.model}] ===")
            print(f"SKIPPED - {exc}")


if __name__ == "__main__":
    main()
