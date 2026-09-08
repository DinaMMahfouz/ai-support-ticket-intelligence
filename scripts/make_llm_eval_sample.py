"""
Draws a frozen, stratified 10% sample of data/eval_holdout_v1.csv for
the LLM-vs-classical benchmark.

Why a sample and not the full 2,400-row holdout: the classical baseline
scores the full holdout in under a second; the LLM path (priority_llm.py,
via local Ollama) costs several seconds per ticket on this hardware with
no GPU, so scoring all 2,400 rows would take roughly two and a half
hours. A 240-row stratified sample keeps the head-to-head runnable in
minutes while still giving each of the four priority classes on the
order of 50-70 examples - enough for a per-class number to mean
something.

The classical model is scored against this exact file too (in addition
to its own already-reported full-holdout number), so the comparison is
on identical rows, not just "the same distribution."

Run explicitly:

    python -m scripts.make_llm_eval_sample
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from ticket_intelligence.priority_features import (
    EVAL_HOLDOUT_PATH,
    EVAL_HOLDOUT_LLM_SAMPLE_PATH,
    TARGET_COLUMN,
    LLM_SAMPLE_FRACTION,
    LLM_SAMPLE_RANDOM_STATE
)


def main():
    if EVAL_HOLDOUT_LLM_SAMPLE_PATH.exists():
        raise FileExistsError(
            f"{EVAL_HOLDOUT_LLM_SAMPLE_PATH.name} already exists. Delete "
            f"it first if you deliberately want a new sample - any "
            f"report already citing llm_sample numbers describes the "
            f"current file."
        )

    df = pd.read_csv(EVAL_HOLDOUT_PATH)

    _, sample = train_test_split(
        df,
        test_size=LLM_SAMPLE_FRACTION,
        random_state=LLM_SAMPLE_RANDOM_STATE,
        stratify=df[TARGET_COLUMN]
    )

    sample.to_csv(EVAL_HOLDOUT_LLM_SAMPLE_PATH, index=False)

    print(f"Eval holdout rows: {len(df)}")
    print(f"LLM sample: {len(sample)} rows -> {EVAL_HOLDOUT_LLM_SAMPLE_PATH}")

    print("\nSample priority distribution:")
    print(sample[TARGET_COLUMN].value_counts())


if __name__ == "__main__":
    main()
