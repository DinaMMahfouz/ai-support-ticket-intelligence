"""
Splits the raw labeled dataset into two permanent, disk-frozen files:

  data/train_pool_v1.csv    everything priority_train.py is allowed to
                             see (for its own internal train/CV/test work
                             while fitting and tuning the model)
  data/eval_holdout_v1.csv  a stratified 20% slice priority_train.py
                             never sees, used only by eval_runner.py to
                             score any classifier - the classical
                             baseline now, an LLM path later - on
                             identical data.

Run explicitly:

    python -m scripts.make_eval_split

Refuses to overwrite existing split files. Delete them yourself first if
you deliberately want to re-split, and be aware that anything already
written down using v1 numbers would then be describing different data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from ticket_intelligence.priority_features import (
    RAW_DATA_PATH,
    DATA_DIR,
    TRAIN_POOL_PATH,
    EVAL_HOLDOUT_PATH,
    TARGET_COLUMN,
    EVAL_HOLDOUT_FRACTION,
    EVAL_SPLIT_RANDOM_STATE
)


def main():
    if TRAIN_POOL_PATH.exists() or EVAL_HOLDOUT_PATH.exists():
        raise FileExistsError(
            f"{TRAIN_POOL_PATH.name} and/or {EVAL_HOLDOUT_PATH.name} "
            f"already exist in {DATA_DIR}. Delete them first if you "
            f"intend to re-split - anything already reported against "
            f"the current files would otherwise silently start "
            f"describing different data."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    train_pool, eval_holdout = train_test_split(
        df,
        test_size=EVAL_HOLDOUT_FRACTION,
        random_state=EVAL_SPLIT_RANDOM_STATE,
        stratify=df[TARGET_COLUMN]
    )

    DATA_DIR.mkdir(exist_ok=True)

    train_pool.to_csv(TRAIN_POOL_PATH, index=False)
    eval_holdout.to_csv(EVAL_HOLDOUT_PATH, index=False)

    print(f"Raw rows: {len(df)}")
    print(f"Train pool: {len(train_pool)} rows -> {TRAIN_POOL_PATH}")
    print(f"Eval holdout: {len(eval_holdout)} rows -> {EVAL_HOLDOUT_PATH}")

    print("\nEval holdout priority distribution:")
    print(eval_holdout[TARGET_COLUMN].value_counts())

    print("\nTrain pool priority distribution:")
    print(train_pool[TARGET_COLUMN].value_counts())


if __name__ == "__main__":
    main()
