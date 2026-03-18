"""
sentiment/sentiment_model.py
=============================
Runs DistilBERT on each cleaned comment and produces
a sentiment score between 0.0 (very negative) and 1.0 (very positive).

Score conversion:
  POSITIVE confidence p  →  score = p
  NEGATIVE confidence p  →  score = 1 - p

Input  : data/processed/cleaned_comments.csv
Output : data/processed/scored_comments.csv
         columns: course_id, prof_id, semester, comment, sentiment_score

Install:
  pip install transformers torch
"""

import os
import torch
import pandas as pd
from transformers import pipeline
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

INPUT_PATH  = "data/processed/cleaned_comments.csv"
OUTPUT_PATH = "data/processed/scored_comments.csv"
MODEL_NAME  = "distilbert-base-uncased-finetuned-sst-2-english"
BATCH_SIZE  = 32      # safe for RTX 3050 4GB VRAM
MAX_LENGTH  = 128     # DistilBERT max token length, comments are short so 128 is fine


# ─────────────────────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────────────────────

def get_device() -> int:
    """
    Returns device index for HuggingFace pipeline.
      0  → GPU (CUDA)
     -1  → CPU fallback
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb     = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected  : {device_name}")
        print(f"VRAM          : {vram_gb:.1f} GB")
        return 0
    else:
        print("No GPU detected — running on CPU (will be slower)")
        return -1


# ─────────────────────────────────────────────────────────────
# SCORE CONVERSION
# ─────────────────────────────────────────────────────────────

def to_sentiment_score(label: str, score: float) -> float:
    """
    Convert DistilBERT output to a 0-1 sentiment score.
      POSITIVE, 0.95 → 0.95
      NEGATIVE, 0.95 → 0.05
    """
    if label == "POSITIVE":
        return round(score, 4)
    else:
        return round(1.0 - score, 4)


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────

def run_sentiment(input_path: str  = INPUT_PATH,
                  output_path: str = OUTPUT_PATH) -> pd.DataFrame:

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load cleaned comments
    df = pd.read_csv(input_path)
    print(f"\nLoaded {len(df)} cleaned comments")

    # Device
    device = get_device()

    # Load DistilBERT pipeline
    # First run will download the model (~250MB), cached after that
    print(f"\nLoading model : {MODEL_NAME}")
    classifier = pipeline(
        task            = "text-classification",
        model           = MODEL_NAME,
        device          = device,
        truncation      = True,
        max_length      = MAX_LENGTH,
    )
    print("Model loaded successfully")

    # Run inference in batches
    comments    = df["comment"].tolist()
    scores      = []
    total       = len(comments)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nRunning inference — {total} comments in {num_batches} batches (batch size {BATCH_SIZE})\n")

    for i in range(0, total, BATCH_SIZE):
        batch      = comments[i : i + BATCH_SIZE]
        results    = classifier(batch)
        batch_scores = [to_sentiment_score(r["label"], r["score"]) for r in results]
        scores.extend(batch_scores)

        # Progress
        done = min(i + BATCH_SIZE, total)
        print(f"  Processed {done:>5} / {total}  "
              f"[batch avg score: {sum(batch_scores)/len(batch_scores):.3f}]")

        # Clear GPU cache after each batch to stay within 4GB VRAM
        if device == 0:
            torch.cuda.empty_cache()

    # Attach scores
    df["sentiment_score"] = scores

    # Save
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved scored comments to {output_path}")

    # Summary stats
    print("\n--- Sentiment Score Distribution ---")
    print(f"  Mean   : {df['sentiment_score'].mean():.3f}")
    print(f"  Std    : {df['sentiment_score'].std():.3f}")
    print(f"  Min    : {df['sentiment_score'].min():.3f}")
    print(f"  Max    : {df['sentiment_score'].max():.3f}")
    print(f"  >0.6 (positive) : {(df['sentiment_score'] > 0.6).sum()} comments")
    print(f"  0.4–0.6 (neutral): {((df['sentiment_score'] >= 0.4) & (df['sentiment_score'] <= 0.6)).sum()} comments")
    print(f"  <0.4 (negative) : {(df['sentiment_score'] < 0.4).sum()} comments")

    return df


if __name__ == "__main__":
    df = run_sentiment()

    # Quick sample to verify scores look reasonable
    print("\n--- Sample scored comments ---")
    sample = df[["prof_id", "comment", "sentiment_score"]].sample(8)
    for _, row in sample.iterrows():
        bar = "█" * int(row["sentiment_score"] * 20)
        print(f"  [{row['sentiment_score']:.2f}] {bar:<20} {row['comment'][:60]}")
