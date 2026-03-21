"""
model/evaluate.py
==================
FIXED evaluation — ranks all 32 eligible OEs per student,
not just the 5 from the interaction matrix.

Also includes:
  - Fairness analysis (NDCG per branch)
  - Popularity baseline (most popular OE per branch vs NeuMF)

Metrics:
  NDCG@10  → ranking quality across full 32 candidates
  Hit@5    → actual OE in top 5 of 32
  Hit@10   → actual OE in top 10 of 32
  RMSE     → predicted score vs actual grade score

Outputs:
  results/evaluation_sem{N}.csv     → per-student results
  results/metrics_sem{N}.csv        → aggregated metrics
  results/fairness_sem{N}.csv       → metrics per branch
  results/baseline_sem{N}.csv       → NeuMF vs popularity baseline

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

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"
PROCESSED_DIR  = "data/processed"
RAW_DIR        = "data/raw"
OE_SEMESTERS   = [5, 6, 7]


# ─────────────────────────────────────────────────────────────
# METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────

def hit_at_k(ranked_oes: list, actual_oe: str, k: int) -> int:
    """1 if actual OE is in top-k of full ranked list, else 0."""
    return int(actual_oe in ranked_oes[:k])


def ndcg_at_k(ranked_oes: list, actual_oe: str, k: int) -> float:
    """
    NDCG@K for a single student over full candidate list.
    Rank 1 of 32 → high score. Rank 25 of 32 → 0 if outside top k.
    """
    if actual_oe not in ranked_oes[:k]:
        return 0.0
    rank = ranked_oes.index(actual_oe)   # 0-indexed
    return 1.0 / np.log2(rank + 2)       # +2 because log2(1)=0


def rmse(predicted: list, actual: list) -> float:
    predicted = np.array(predicted)
    actual    = np.array(actual)
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


# ─────────────────────────────────────────────────────────────
# POPULARITY BASELINE
# ─────────────────────────────────────────────────────────────

def build_popularity_baseline(interaction_df: pd.DataFrame,
                               oe_info_df    : pd.DataFrame,
                               prof_features_path: str = "data/processed/prof_features.csv") -> dict:
    """
    Ranks OEs by their professor's avg_overall_rating as a popularity proxy.

    Why: each semester has completely different OEs (sem 5 vs sem 7 are
    different subjects), so raw pick-count from training doesn't transfer.
    Professor rating is a semester-agnostic quality signal that applies
    to any OE they teach.

    Returns {(student_branch, semester): [oe_id ranked by prof rating]}
    """
    prof_features = pd.read_csv(prof_features_path)

    # Attach professor rating to each OE
    oe_with_rating = oe_info_df.merge(
        prof_features[["prof_id", "avg_overall_rating"]],
        on="prof_id",
        how="left"
    )
    oe_with_rating["avg_overall_rating"] = \
        oe_with_rating["avg_overall_rating"].fillna(
            oe_with_rating["avg_overall_rating"].mean()
        )

    baseline = {}
    for sem in OE_SEMESTERS:
        for branch in oe_info_df["offering_branch"].unique():
            # Eligible OEs for this branch+sem sorted by professor rating
            eligible = oe_with_rating[
                (oe_with_rating["available_semester"] == sem) &
                (oe_with_rating["offering_branch"]    != branch)
            ].sort_values("avg_overall_rating", ascending=False)["oe_id"].tolist()
            baseline[(branch, sem)] = eligible

    return baseline


# ─────────────────────────────────────────────────────────────
# FULL RANKING EVALUATION (FIXED)
# ─────────────────────────────────────────────────────────────

def evaluate_student_full_ranking(student_id    : str,
                                   student_branch: str,
                                   actual_oe     : str,
                                   actual_score  : float,
                                   semester      : int,
                                   oe_info_df    : pd.DataFrame,
                                   taken_oes     : set,
                                   model         : torch.nn.Module,
                                   student_lookup: dict,
                                   oe_lookup     : dict,
                                   device        : torch.device) -> dict:
    """
    Scores ALL 32 eligible OEs for a student — not just the ones
    in the interaction matrix. This is the correct evaluation.

    eligible OEs = same semester + not own branch + not already taken
    """

    # Get all 32 eligible OEs
    eligible = oe_info_df[
        (oe_info_df["available_semester"] == semester) &
        (oe_info_df["offering_branch"]    != student_branch) &
        (~oe_info_df["oe_id"].isin(taken_oes - {actual_oe}))
    ]["oe_id"].tolist()

    if not eligible or actual_oe not in eligible:
        return None

    # Score all eligible OEs
    student_vec = torch.tensor(
        student_lookup[student_id], dtype=torch.float32
    ).unsqueeze(0).to(device)

    pred_scores = []
    with torch.no_grad():
        for oe_id in eligible:
            if oe_id not in oe_lookup:
                continue
            oe_vec = torch.tensor(
                oe_lookup[oe_id], dtype=torch.float32
            ).unsqueeze(0).to(device)
            score  = model(student_vec, oe_vec).item()
            pred_scores.append((oe_id, score))

    # Sort descending → full ranked list of 32
    pred_scores.sort(key=lambda x: x[1], reverse=True)
    ranked_oes    = [oe for oe, _ in pred_scores]
    actual_pred   = next((s for oe, s in pred_scores if oe == actual_oe), 0.0)
    actual_rank   = ranked_oes.index(actual_oe) + 1 if actual_oe in ranked_oes else -1

    return {
        "student_id"      : student_id,
        "branch"          : student_branch,
        "semester"        : semester,
        "actual_oe"       : actual_oe,
        "actual_rank"     : actual_rank,
        "total_candidates": len(ranked_oes),
        "actual_score"    : round(actual_score, 4),
        "predicted_score" : round(actual_pred, 4),
        "hit@5"           : hit_at_k(ranked_oes, actual_oe, 5),
        "hit@10"          : hit_at_k(ranked_oes, actual_oe, 10),
        "ndcg@10"         : round(ndcg_at_k(ranked_oes, actual_oe, 10), 4),
        "top5_oes"        : " | ".join(ranked_oes[:5]),
    }


# ─────────────────────────────────────────────────────────────
# FAIRNESS ANALYSIS
# ─────────────────────────────────────────────────────────────

def compute_fairness(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group metrics by student branch.
    Checks if students from all branches are served equally well.
    """
    fairness = (
        results_df.groupby("branch")
        .agg(
            n_students  = ("student_id", "count"),
            ndcg_10     = ("ndcg@10",    "mean"),
            hit_5       = ("hit@5",      "mean"),
            hit_10      = ("hit@10",     "mean"),
            avg_rank    = ("actual_rank","mean"),
            rmse        = ("predicted_score", lambda x:
                           rmse(x.tolist(),
                                results_df.loc[x.index, "actual_score"].tolist()))
        )
        .reset_index()
        .round(4)
    )
    return fairness


# ─────────────────────────────────────────────────────────────
# BASELINE COMPARISON
# ─────────────────────────────────────────────────────────────

def compute_baseline_comparison(results_df  : pd.DataFrame,
                                 baseline_map: dict) -> pd.DataFrame:
    """
    Compare NeuMF metrics against professor-rating baseline.
    Baseline = always recommend OEs with highest professor rating first.
    No personalization — same ranking for every student of same branch.
    """
    baseline_rows = []

    for _, row in results_df.iterrows():
        branch   = row["branch"]
        semester = row["semester"]
        actual   = row["actual_oe"]

        ranked = baseline_map.get((branch, semester), [])
        if not ranked:
            continue

        baseline_rows.append({
            "student_id"      : row["student_id"],
            "branch"          : branch,
            "hit@5_baseline"  : hit_at_k(ranked, actual, 5),
            "hit@10_baseline" : hit_at_k(ranked, actual, 10),
            "ndcg@10_baseline": round(ndcg_at_k(ranked, actual, 10), 4),
        })

    baseline_df = pd.DataFrame(baseline_rows)
    if baseline_df.empty:
        return pd.DataFrame()

    # Merge with NeuMF results
    comparison = results_df.merge(
        baseline_df[["student_id",
                     "hit@5_baseline", "hit@10_baseline",
                     "ndcg@10_baseline"]],
        on="student_id", how="left"
    )

    summary = pd.DataFrame([{
        "metric"            : "NDCG@10",
        "NeuMF"             : round(comparison["ndcg@10"].mean(), 4),
        "prof_rating_baseline": round(comparison["ndcg@10_baseline"].mean(), 4),
        "improvement"       : round(
            comparison["ndcg@10"].mean() -
            comparison["ndcg@10_baseline"].mean(), 4),
    }, {
        "metric"            : "Hit@5",
        "NeuMF"             : round(comparison["hit@5"].mean(), 4),
        "prof_rating_baseline": round(comparison["hit@5_baseline"].mean(), 4),
        "improvement"       : round(
            comparison["hit@5"].mean() -
            comparison["hit@5_baseline"].mean(), 4),
    }, {
        "metric"            : "Hit@10",
        "NeuMF"             : round(comparison["hit@10"].mean(), 4),
        "prof_rating_baseline": round(comparison["hit@10_baseline"].mean(), 4),
        "improvement"       : round(
            comparison["hit@10"].mean() -
            comparison["hit@10_baseline"].mean(), 4),
    }])

    return summary


# ─────────────────────────────────────────────────────────────
# FULL EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate(semester: int = 7) -> pd.DataFrame:

    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    # Find best checkpoint
    checkpoint_path = None
    for sem in [semester - 1, semester - 2, 5]:
        path = f"{CHECKPOINT_DIR}/model_sem{sem}.pt"
        if os.path.exists(path):
            checkpoint_path = path
            break

    if not checkpoint_path:
        raise FileNotFoundError("No checkpoint found. Run model/train.py first.")

    print(f"Checkpoint : {checkpoint_path}")

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

    print(f"Model loaded (epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.4f})\n")

    # Load tables
    interaction_df = pd.read_csv(f"{PROCESSED_DIR}/interaction_matrix.csv")
    students_df    = pd.read_csv(f"{RAW_DIR}/students.csv")
    oe_info_df     = pd.read_csv(f"{RAW_DIR}/oe_info.csv")

    # Test split — only positive interactions (actual choices)
    test_positives = interaction_df[
        (interaction_df["split"]       == "test") &
        (interaction_df["is_negative"] == 0)
    ]

    # Build taken_oes per student (all semesters, to exclude from candidates)
    all_taken = (
        interaction_df[interaction_df["is_negative"] == 0]
        .groupby("student_id")["oe_id"]
        .apply(set)
        .to_dict()
    )

    # Branch per student
    student_branch = students_df.set_index("student_id")["branch"].to_dict()

    # Build professor-rating baseline
    baseline_map = build_popularity_baseline(interaction_df, oe_info_df)

    students = test_positives["student_id"].unique()
    print(f"Evaluating {len(students)} students on FULL 32-OE ranking...")
    print(f"(Each student ranked against all eligible OEs, not just sampled negatives)")
    print("─" * 65)

    # Per-student full ranking evaluation
    results = []
    for student_id in students:
        student_rows = test_positives[test_positives["student_id"] == student_id]

        for _, row in student_rows.iterrows():
            actual_oe    = row["oe_id"]
            actual_score = row["score"]
            sem          = row["semester"]
            branch       = student_branch.get(student_id, "")
            taken        = all_taken.get(student_id, set())

            result = evaluate_student_full_ranking(
                student_id     = student_id,
                student_branch = branch,
                actual_oe      = actual_oe,
                actual_score   = actual_score,
                semester       = sem,
                oe_info_df     = oe_info_df,
                taken_oes      = taken,
                model          = model,
                student_lookup = test_ds._student_lookup,
                oe_lookup      = test_ds._oe_lookup,
                device         = device,
            )
            if result:
                results.append(result)

    results_df = pd.DataFrame(results)

    # ── Aggregate metrics ────────────────────────────────────
    mean_ndcg  = results_df["ndcg@10"].mean()
    mean_hit5  = results_df["hit@5"].mean()
    mean_hit10 = results_df["hit@10"].mean()
    mean_rmse  = rmse(
        results_df["predicted_score"].tolist(),
        results_df["actual_score"].tolist()
    )
    mean_rank  = results_df["actual_rank"].mean()

    rank_dist = {
        "rank_1"    : (results_df["actual_rank"] == 1).sum(),
        "rank_1_5"  : (results_df["actual_rank"] <= 5).sum(),
        "rank_1_10" : (results_df["actual_rank"] <= 10).sum(),
        "rank_11_32": (results_df["actual_rank"] > 10).sum(),
    }

    # ── Fairness analysis ────────────────────────────────────
    fairness_df = compute_fairness(results_df)

    # ── Baseline comparison ──────────────────────────────────
    baseline_df = compute_baseline_comparison(
        results_df, baseline_map
    )

    # ── Print results ────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  EVALUATION RESULTS — Semester {semester}")
    print(f"  (Full ranking: {results_df['total_candidates'].mode()[0]} candidates per student)")
    print(f"{'='*55}")
    print(f"  Students evaluated : {len(results_df)}")
    print(f"  NDCG@10            : {mean_ndcg:.4f}")
    print(f"  Hit@5              : {mean_hit5:.4f}  "
          f"({int(mean_hit5 * len(results_df))}/{len(results_df)} students)")
    print(f"  Hit@10             : {mean_hit10:.4f}  "
          f"({int(mean_hit10 * len(results_df))}/{len(results_df)} students)")
    print(f"  RMSE               : {mean_rmse:.4f}")
    print(f"  Avg rank of actual : {mean_rank:.1f} / "
          f"{results_df['total_candidates'].mode()[0]}")
    print(f"\n  Rank distribution:")
    print(f"    Ranked #1        : {rank_dist['rank_1']} students")
    print(f"    Ranked top 5     : {rank_dist['rank_1_5']} students")
    print(f"    Ranked top 10    : {rank_dist['rank_1_10']} students")
    print(f"    Ranked 11-32     : {rank_dist['rank_11_32']} students")

    print(f"\n{'='*55}")
    print(f"  FAIRNESS ANALYSIS — by student branch")
    print(f"{'='*55}")
    print(fairness_df[["branch", "n_students", "ndcg_10",
                        "hit_5", "hit_10", "avg_rank"]].to_string(index=False))

    if not baseline_df.empty:
        print(f"\n{'='*55}")
        print(f"  BASELINE COMPARISON — NeuMF vs Prof Rating Baseline")
        print(f"  (Baseline = rank all OEs by professor avg_overall_rating)")
        print(f"{'='*55}")
        print(baseline_df.to_string(index=False))

    print(f"{'='*55}")

    # ── Save ────────────────────────────────────────────────
    results_df.to_csv(
        f"{RESULTS_DIR}/evaluation_sem{semester}.csv", index=False
    )
    fairness_df.to_csv(
        f"{RESULTS_DIR}/fairness_sem{semester}.csv", index=False
    )
    if not baseline_df.empty:
        baseline_df.to_csv(
            f"{RESULTS_DIR}/baseline_sem{semester}.csv", index=False
        )

    metrics = pd.DataFrame([{
        "semester"        : semester,
        "ndcg@10"         : round(mean_ndcg,  4),
        "hit@5"           : round(mean_hit5,  4),
        "hit@10"          : round(mean_hit10, 4),
        "rmse"            : round(mean_rmse,  4),
        "avg_rank"        : round(mean_rank,  2),
        "total_candidates": int(results_df["total_candidates"].mode()[0]),
        "n_students"      : len(results_df),
    }])
    metrics.to_csv(f"{RESULTS_DIR}/metrics_sem{semester}.csv", index=False)

    print(f"\n✓ Results saved to {RESULTS_DIR}/")
    print(f"  evaluation_sem{semester}.csv  → per-student full ranking")
    print(f"  fairness_sem{semester}.csv    → metrics by branch")
    print(f"  baseline_sem{semester}.csv    → NeuMF vs popularity baseline")
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