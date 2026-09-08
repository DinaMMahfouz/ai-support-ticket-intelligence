"""
Shared constants and shared IO for the priority classifier.

priority_train.py, make_eval_split.py, and eval_runner.py all need to
build the same-shaped feature frame from a raw ticket CSV. Importing
these from one place means "how do you build the text column" and
"what are the feature columns" can never drift into two different
answers in two different files.
"""

import os
from pathlib import Path

import pandas as pd

# This file lives in ticket_intelligence/, so the repo root - which is
# what data/, models/ and .env are relative to - is one level up.
PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent
DATA_DIR = BASE_DIR / "data"


def _load_dotenv(path: Path) -> None:
    """
    Minimal stdlib .env loader: reads KEY=VALUE lines into os.environ,
    skipping blank lines and comments. Doesn't override a variable
    already set in the real environment. A dedicated package
    (python-dotenv) would do the same thing for a task this small -
    plain line parsing, no nested values, no interpolation - so it
    isn't justified here.
    """

    if not path.exists():
        return

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        os.environ.setdefault(key, value)


def load_env() -> None:
    """
    Loads ANTHROPIC_API_KEY (and anything else) from the repo's .env
    file into os.environ, if not already set there. Every Anthropic
    provider module calls this explicitly at import time rather than
    relying on "importing priority_features happens to trigger it" -
    an explicit call says outright that the module needs an environment
    variable loaded, instead of depending on unstated import ordering.
    """

    _load_dotenv(BASE_DIR / ".env")


load_env()

# The original, full labeled dataset. Nothing trains or scores on this
# file directly anymore - make_eval_split.py splits it once into the two
# files below, so training and evaluation always draw from disjoint,
# frozen data. Lives in data/ (moved there from the repo root during
# this project).
RAW_DATA_PATH = DATA_DIR / "ai_support_ticket_training_data_realistic_v2.csv"

# Versioned, disk-frozen splits ("v1" so a future re-split can't silently
# change what a report that already cites "the eval set" is talking
# about - that would require a new v2 file, not an overwrite).
TRAIN_POOL_PATH = DATA_DIR / "train_pool_v1.csv"
EVAL_HOLDOUT_PATH = DATA_DIR / "eval_holdout_v1.csv"

# A frozen stratified sample of eval_holdout_v1.csv, used for the
# LLM-vs-classical head-to-head. The classical model scores the full
# 2,400-row holdout in under a second; the LLM path costs several
# seconds per ticket on this hardware, so the fair-comparison run uses
# this smaller, identical-for-both-models file instead of all 2,400 rows.
EVAL_HOLDOUT_LLM_SAMPLE_PATH = DATA_DIR / "eval_holdout_llm_sample_v1.csv"
LLM_SAMPLE_FRACTION = 0.10
LLM_SAMPLE_RANDOM_STATE = 7

# A separate, hand-written hard-case set (ambiguous/adversarial tickets),
# not sampled from the training distribution.
HARD_EVAL_PATH = DATA_DIR / "hard_eval_set.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "priority_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

TEXT_COLUMN = "text"

CATEGORICAL_COLUMNS = [
    "category",
    "environment",
    "channel",
    "region",
    "customer_tier"
]

FEATURE_COLUMNS = [TEXT_COLUMN] + CATEGORICAL_COLUMNS

TARGET_COLUMN = "priority"

PRIORITY_LABELS = ["Critical", "High", "Medium", "Low"]

# How make_eval_split.py carves the raw CSV into train_pool vs.
# eval_holdout. Deliberately a different random_state from the
# random_state=32 used inside priority_train.py's own internal
# train/test split, so it's obvious in code review that these are two
# unrelated splits, not the same one reused for two purposes.
EVAL_HOLDOUT_FRACTION = 0.20
EVAL_SPLIT_RANDOM_STATE = 42

# How many labeled examples per class any LLM provider shows the model,
# and a fixed seed - shared here (not duplicated per provider module) so
# every provider is shown the exact same example tickets. If two
# providers scored differently because they saw different examples
# rather than because of the model itself, the comparison would be
# meaningless.
#
# Raised from 2 to 8 after the first few-shot benchmark: both models'
# errors were traceable to specific gaps in what 2 examples/class could
# show (e.g. Claude Haiku's poor Critical recall came from only 2
# Critical examples not covering the range of what this dataset calls
# Critical) - a data-volume problem, not a hard capability ceiling, so
# worth testing directly rather than assumed.
FEW_SHOT_EXAMPLES_PER_CLASS = 8
FEW_SHOT_RANDOM_STATE = 11


def load_ticket_csv(path: Path) -> pd.DataFrame:
    """
    Load a ticket CSV and build the combined text column used as a
    model feature. Used identically by training and eval scoring, so
    "subject + description" is built exactly one way everywhere.
    """

    df = pd.read_csv(path)

    df["subject"] = df["subject"].fillna("")
    df["description"] = df["description"].fillna("")
    df[TEXT_COLUMN] = df["subject"] + " " + df["description"]

    return df


def sample_few_shot_rows() -> pd.DataFrame:
    """
    Samples FEW_SHOT_EXAMPLES_PER_CLASS rows per priority class from the
    training pool only - never from an eval file. This function has no
    way to see eval_holdout_v1.csv, eval_holdout_llm_sample_v1.csv, or
    hard_eval_set.csv, so there's no code path by which an eval row
    could end up inside any LLM provider's prompt.
    """

    if not TRAIN_POOL_PATH.exists():
        raise FileNotFoundError(
            f"{TRAIN_POOL_PATH} not found. Run 'python -m scripts.make_eval_split' "
            f"first - few-shot examples are sampled from it."
        )

    df = pd.read_csv(TRAIN_POOL_PATH)
    df["subject"] = df["subject"].fillna("")
    df["description"] = df["description"].fillna("")

    per_class_samples = [
        df[df[TARGET_COLUMN] == label].sample(
            n=FEW_SHOT_EXAMPLES_PER_CLASS,
            random_state=FEW_SHOT_RANDOM_STATE
        )
        for label in PRIORITY_LABELS
    ]

    return pd.concat(per_class_samples, ignore_index=True)
