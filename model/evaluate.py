"""
model/evaluate.py
==================
Full evaluation of trained NeuMF on the test set (sem 7).

Metrics:
  NDCG@10  → ranking quality
  Hit@5    → actual OE in top 5
  Hit@10   → actual OE in top 10
  RMSE     → predicted score vs actual grade score

Outputs:
  results/evaluation_sem{N}.csv  → per-student results
  results/metrics_sem{N}.csv     → aggregated metrics

Usage:
  python -m model.evaluate
  python -m model.evaluate --semester 6
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd

from model.dataset import OEDataset, STUDENT_FEATURE_COLS, OE_FEATURE_COLS
from model.neumf   import NeuMF
from model.train   import load_checkpoint, EMBEDDING_DIM, MLP_LAYERS, DROPOUT_RATE

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

CHECKPOINT_DIR  = "checkpoints"
RESULTS_DIR     = "results"
PROCESSED_DIR   = "data/processed"


# ─────────────────────────────────────────────────────────────
# METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────

def hit_at_k(ranked_oes: list, actual_oe: str, k: int) -> int:
    """1 if actual OE is in top-k ranked list, else 0."""
    return int(actual_oe in ranked_oes[:k])


def ndcg_at_k(ranked_oes: list, actual_oe: str, k: int) -> float:
    """
    NDCG@K for a single student.
    Higher rank position = higher score.
    Actual OE not in top-k = 0.
    """
    if actual_oe not in ranked_oes[:k]:
        return 0.0
    rank = ranked_oes.index(actual_oe)     # 0-indexed
    return 1.0 / np.log2(rank + 2)         # +2 because log2(1) = 0


def rmse(predicted: list, actual: list) -> float:
    """Root mean squared error between predicted and actual scores."""
    predicted = np.array(predicted)
    actual    = np.array(actual)
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


# ─────────────────────────────────────────────────────────────
# PER-STUDENT EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_student(student_id   : str,
                     student_rows : pd.DataFrame,
                     model        : torch.nn.Module,
                     student_lookup: dict,
                     oe_lookup    : dict,
                     device       : torch.device) -> dict:
    """
    For one student:
      1. Score all their eligible OEs (positives + negatives)
      2. Rank by predicted score
      3. Compute all metrics against their actual OE choice
    """
    actual_row = student_rows[student_rows["is_negative"] == 0]
    if actual_row.empty:
        return None

    actual_oe    = actual_row["oe_id"].values[0]
    actual_score = actual_row["score"].values[0]

    # Score all OEs for this student
    student_vec  = torch.tensor(
        student_lookup[student_id], dtype=torch.float32
    ).unsqueeze(0).to(device)

    all_oe_ids   = student_rows["oe_id"].values
    pred_scores  = []

    with torch.no_grad():
        for oe_id in all_oe_ids:
            if oe_id not in oe_lookup:
                continue
            oe_vec = torch.tensor(
                oe_lookup[oe_id], dtype=torch.float32
            ).unsqueeze(0).to(device)
            score  = model(student_vec, oe_vec).item()
            pred_scores.append((oe_id, score))

    # Sort by predicted score descending → ranked list
    pred_scores.sort(key=lambda x: x[1], reverse=True)
    ranked_oes   = [oe for oe, _ in pred_scores]
    ranked_scores = [s for _, s in pred_scores]

    # Predicted score for actual OE
    actual_pred_score = next(
        (s for oe, s in pred_scores if oe == actual_oe), 0.0
    )

    # Rank of actual OE (1-indexed for display)
    actual_rank = ranked_oes.index(actual_oe) + 1 if actual_oe in ranked_oes else -1

    return {
        "student_id"       : student_id,
        "actual_oe"        : actual_oe,
        "actual_rank"      : actual_rank,
        "actual_score"     : round(actual_score, 4),
        "predicted_score"  : round(actual_pred_score, 4),
        "hit@5"            : hit_at_k(ranked_oes, actual_oe, 5),
        "hit@10"           : hit_at_k(ranked_oes, actual_oe, 10),
        "ndcg@10"          : round(ndcg_at_k(ranked_oes, actual_oe, 10), 4),
        "total_candidates" : len(ranked_oes),
        "top5_oes"         : " | ".join(ranked_oes[:5]),
    }


# ─────────────────────────────────────────────────────────────
# FULL EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate(semester: int = 7) -> pd.DataFrame:

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")

    # Load checkpoint
    checkpoint_path = f"{CHECKPOINT_DIR}/model_sem{semester - 2}.pt"
    if not os.path.exists(checkpoint_path):
        # Fall back to sem5 checkpoint if specific one not found
        checkpoint_path = f"{CHECKPOINT_DIR}/model_sem5.pt"

    print(f"Checkpoint: {checkpoint_path}")

    # Load dataset for lookups
    test_ds = OEDataset(split="test")

    # Build model
    model = NeuMF(
        student_dim   = test_ds.student_dim,
        oe_dim        = test_ds.oe_dim,
        embedding_dim = EMBEDDING_DIM,
        mlp_layers    = MLP_LAYERS,
        dropout       = DROPOUT_RATE,
    ).to(device)

    checkpoint = load_checkpoint(checkpoint_path, model)
    model.eval()

    print(f"Model loaded (trained at epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.4f})\n")

    # Load test interactions
    interaction_df = pd.read_csv(f"{PROCESSED_DIR}/interaction_matrix.csv")
    test_df        = interaction_df[interaction_df["split"] == "test"]
    students       = test_df["student_id"].unique()

    print(f"Evaluating {len(students)} students on test set (sem {semester})...")
    print("─" * 60)

    # Per-student evaluation
    results = []
    for student_id in students:
        student_rows = test_df[test_df["student_id"] == student_id]
        result       = evaluate_student(
            student_id, student_rows, model,
            test_ds._student_lookup, test_ds._oe_lookup, device
        )
        if result:
            results.append(result)

    results_df = pd.DataFrame(results)

    # ── Aggregate metrics ────────────────────────────────────
    mean_ndcg   = results_df["ndcg@10"].mean()
    mean_hit5   = results_df["hit@5"].mean()
    mean_hit10  = results_df["hit@10"].mean()
    mean_rmse   = rmse(
        results_df["predicted_score"].tolist(),
        results_df["actual_score"].tolist()
    )

    # Rank distribution
    rank_dist = {
        "rank_1"    : (results_df["actual_rank"] == 1).sum(),
        "rank_1_5"  : (results_df["actual_rank"] <= 5).sum(),
        "rank_1_10" : (results_df["actual_rank"] <= 10).sum(),
        "rank_11_32": (results_df["actual_rank"] > 10).sum(),
    }

    # ── Print results ────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"  EVALUATION RESULTS — Semester {semester}")
    print(f"{'='*40}")
    print(f"  Students evaluated : {len(results_df)}")
    print(f"  NDCG@10            : {mean_ndcg:.4f}")
    print(f"  Hit@5              : {mean_hit5:.4f}  "
          f"({int(mean_hit5 * len(results_df))}/{len(results_df)} students)")
    print(f"  Hit@10             : {mean_hit10:.4f}  "
          f"({int(mean_hit10 * len(results_df))}/{len(results_df)} students)")
    print(f"  RMSE               : {mean_rmse:.4f}")
    print(f"\n  Rank distribution:")
    print(f"    Ranked #1        : {rank_dist['rank_1']} students")
    print(f"    Ranked top 5     : {rank_dist['rank_1_5']} students")
    print(f"    Ranked top 10    : {rank_dist['rank_1_10']} students")
    print(f"    Ranked 11–32     : {rank_dist['rank_11_32']} students")
    print(f"{'='*40}")

    # ── Save ────────────────────────────────────────────────
    results_df.to_csv(f"{RESULTS_DIR}/evaluation_sem{semester}.csv", index=False)

    metrics = pd.DataFrame([{
        "semester" : semester,
        "ndcg@10"  : round(mean_ndcg,  4),
        "hit@5"    : round(mean_hit5,  4),
        "hit@10"   : round(mean_hit10, 4),
        "rmse"     : round(mean_rmse,  4),
        "n_students": len(results_df),
    }])
    metrics.to_csv(f"{RESULTS_DIR}/metrics_sem{semester}.csv", index=False)

    print(f"\n✓ Results saved to {RESULTS_DIR}/")
    print(f"  evaluation_sem{semester}.csv  → per-student breakdown")
    print(f"  metrics_sem{semester}.csv     → aggregated metrics")

    return results_df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--semester", type=int, default=7,
                        help="Semester to evaluate on (default: 7)")
    args = parser.parse_args()
    evaluate(semester=args.semester)
