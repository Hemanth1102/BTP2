"""
sentiment/preprocess.py
=======================
Cleans raw comments from course_comments.csv and prepares
them for DistilBERT inference.

Steps:
  1. Lowercase
  2. Remove special characters and punctuation
  3. Strip extra whitespace
  4. Drop comments shorter than 3 words (no signal)
  5. Save to data/processed/cleaned_comments.csv

Input  : data/raw/course_comments.csv
Output : data/processed/cleaned_comments.csv
"""

import os
import re
import pandas as pd

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

RAW_PATH       = "data/raw/course_comments.csv"
PROCESSED_PATH = "data/processed/cleaned_comments.csv"
MIN_WORD_COUNT = 3       # drop comments shorter than this


# ─────────────────────────────────────────────────────────────
# CLEANING FUNCTIONS
# ─────────────────────────────────────────────────────────────

def lowercase(text: str) -> str:
    return text.lower()


def remove_special_characters(text: str) -> str:
    # Keep only letters, numbers, spaces, and basic punctuation (. , !)
    # Everything else (emojis, symbols, brackets) gets removed
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_comment(text: str) -> str:
    """Apply all cleaning steps in order."""
    text = lowercase(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)
    return text


def is_valid(text: str) -> bool:
    """Return False if comment is too short to carry any signal."""
    return len(text.split()) >= MIN_WORD_COUNT


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def preprocess(raw_path: str = RAW_PATH,
               output_path: str = PROCESSED_PATH) -> pd.DataFrame:

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} comments from {raw_path}")

    original_count = len(df)

    # Clean
    df["comment"] = df["comment"].astype(str).apply(clean_comment)

    # Drop too-short comments
    df = df[df["comment"].apply(is_valid)].reset_index(drop=True)
    dropped = original_count - len(df)
    print(f"Dropped {dropped} comments (too short after cleaning)")
    print(f"Remaining: {len(df)} comments")

    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned comments to {output_path}")

    return df


if __name__ == "__main__":
    df = preprocess()

    # Quick sample to verify output
    print("\n--- Sample cleaned comments ---")
    print(df[["prof_id", "comment"]].sample(5).to_string(index=False))
