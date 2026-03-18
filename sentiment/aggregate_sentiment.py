"""
sentiment/aggregate_sentiment.py
==================================
Collapses per-comment sentiment scores into one
sentiment_score per professor using weighted averaging.

Weight = num_responses from course_feedback
(more student responses = more reliable signal)

Also applies the three-level fallback chain:
  Level 1 → prof's own weighted sentiment score
  Level 2 → department average (same offering_branch via oe_info)
  Level 3 → global average (all professors)

Produces prof_features table:
  prof_id, avg_clarity, avg_organization, avg_overall,
  avg_interaction, avg_assignment_usefulness,
  sentiment_score, is_new_prof

Input  : data/processed/scored_comments.csv
         data/raw/course_feedback.csv
         data/raw/oe_info.csv
Output : data/processed/prof_features.csv
"""

import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

SCORED_COMMENTS_PATH = "data/processed/scored_comments.csv"
COURSE_FEEDBACK_PATH = "data/raw/course_feedback.csv"
OE_INFO_PATH         = "data/raw/oe_info.csv"
OUTPUT_PATH          = "data/processed/prof_features.csv"

RATING_COLS = [
    "avg_assignment_usefulness",
    "avg_teaching_clarity",
    "avg_course_organization",
    "avg_interaction",
    "avg_overall_rating",
]


# ─────────────────────────────────────────────────────────────
# STEP 1 — weighted sentiment score per professor
# ─────────────────────────────────────────────────────────────

def compute_weighted_sentiment(scored_df: pd.DataFrame,
                                feedback_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (prof_id, course_id, semester) group, compute
    the weighted average sentiment score using num_responses as weight.

    Returns DataFrame: prof_id, sentiment_score
    """

    # Merge comments with num_responses so we have a weight per row
    merged = scored_df.merge(
        feedback_df[["course_id", "prof_id", "semester", "num_responses"]],
        on=["course_id", "prof_id", "semester"],
        how="left",
    )

    # Fill missing num_responses with 1 (equal weight fallback)
    merged["num_responses"] = merged["num_responses"].fillna(1)

    # Weighted sentiment per comment = score × num_responses
    merged["weighted_score"] = merged["sentiment_score"] * merged["num_responses"]

    # Aggregate per professor
    agg = merged.groupby("prof_id").agg(
        total_weighted = ("weighted_score", "sum"),
        total_weight   = ("num_responses",  "sum"),
        comment_count  = ("sentiment_score", "count"),
    ).reset_index()

    agg["sentiment_score"] = (agg["total_weighted"] / agg["total_weight"]).round(4)

    print(f"Computed weighted sentiment for {len(agg)} professors")
    return agg[["prof_id", "sentiment_score", "comment_count"]]


# ─────────────────────────────────────────────────────────────
# STEP 2 — numeric rating averages per professor
# ─────────────────────────────────────────────────────────────

def compute_rating_averages(feedback_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average all numeric rating columns per professor
    weighted by num_responses.
    """
    result_rows = []

    for prof_id, group in feedback_df.groupby("prof_id"):
        weights = group["num_responses"].values
        row     = {"prof_id": prof_id}

        for col in RATING_COLS:
            values     = group[col].values
            row[col]   = round(float(np.average(values, weights=weights)), 4)

        result_rows.append(row)

    return pd.DataFrame(result_rows)


# ─────────────────────────────────────────────────────────────
# STEP 3 — three-level fallback chain
# ─────────────────────────────────────────────────────────────

def apply_fallback(prof_features_df: pd.DataFrame,
                   oe_info_df: pd.DataFrame) -> pd.DataFrame:
    """
    For professors with no sentiment data:
      Level 2 → use average of professors in same offering_branch
      Level 3 → use global average

    Sets is_new_prof = 1 for any professor who needed fallback.
    """

    # Build branch → prof mapping from oe_info
    branch_prof_map = oe_info_df[["prof_id", "offering_branch"]].drop_duplicates()
    prof_features_df = prof_features_df.merge(branch_prof_map, on="prof_id", how="left")

    # Global averages (Level 3)
    global_avg = {
        col: prof_features_df[col].mean()
        for col in RATING_COLS + ["sentiment_score"]
    }

    # Department averages (Level 2)
    dept_avg = (
        prof_features_df.groupby("offering_branch")[RATING_COLS + ["sentiment_score"]]
        .mean()
        .to_dict(orient="index")
    )

    # All known professors from oe_info
    all_profs  = oe_info_df["prof_id"].unique()
    known_profs = set(prof_features_df["prof_id"].tolist())
    new_profs   = [p for p in all_profs if p not in known_profs]

    print(f"Professors with feedback  : {len(known_profs)}")
    print(f"New professors (no data)  : {len(new_profs)}")

    new_rows = []
    for prof_id in new_profs:
        # Find which branch this prof teaches in
        branch = oe_info_df[oe_info_df["prof_id"] == prof_id]["offering_branch"].values
        branch = branch[0] if len(branch) > 0 else None

        row = {"prof_id": prof_id, "is_new_prof": 1}

        if branch and branch in dept_avg:
            # Level 2 — dept average
            for col in RATING_COLS + ["sentiment_score"]:
                row[col] = round(dept_avg[branch][col], 4)
            row["fallback_level"] = 2
        else:
            # Level 3 — global average
            for col in RATING_COLS + ["sentiment_score"]:
                row[col] = round(global_avg[col], 4)
            row["fallback_level"] = 3

        new_rows.append(row)

    # Mark existing profs as not new
    prof_features_df["is_new_prof"]     = 0
    prof_features_df["fallback_level"]  = 1

    if new_rows:
        new_df           = pd.DataFrame(new_rows)
        prof_features_df = pd.concat([prof_features_df, new_df], ignore_index=True)

    return prof_features_df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def aggregate(scored_comments_path : str = SCORED_COMMENTS_PATH,
              course_feedback_path  : str = COURSE_FEEDBACK_PATH,
              oe_info_path          : str = OE_INFO_PATH,
              output_path           : str = OUTPUT_PATH) -> pd.DataFrame:

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load
    scored_df   = pd.read_csv(scored_comments_path)
    feedback_df = pd.read_csv(course_feedback_path)
    oe_info_df  = pd.read_csv(oe_info_path)

    print(f"Loaded {len(scored_df)} scored comments")
    print(f"Loaded {len(feedback_df)} feedback rows")
    print(f"Loaded {len(oe_info_df)} OE records\n")

    # Step 1 — weighted sentiment per prof
    sentiment_df = compute_weighted_sentiment(scored_df, feedback_df)

    # Step 2 — numeric rating averages per prof
    ratings_df = compute_rating_averages(feedback_df)

    # Merge sentiment + ratings
    prof_features_df = ratings_df.merge(sentiment_df[["prof_id", "sentiment_score"]],
                                         on="prof_id", how="left")

    # Step 3 — fallback for new profs
    prof_features_df = apply_fallback(prof_features_df, oe_info_df)

    # Final column order
    final_cols = [
        "prof_id",
        "avg_teaching_clarity",
        "avg_course_organization",
        "avg_overall_rating",
        "avg_interaction",
        "avg_assignment_usefulness",
        "sentiment_score",
        "is_new_prof",
        "fallback_level",
    ]
    prof_features_df = prof_features_df[final_cols]

    # Save
    prof_features_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved prof_features to {output_path}")

    # Summary
    print("\n--- Prof Features Summary ---")
    print(f"  Total professors         : {len(prof_features_df)}")
    print(f"  With own data (level 1)  : {(prof_features_df['fallback_level'] == 1).sum()}")
    print(f"  Dept avg fallback (lvl 2): {(prof_features_df['fallback_level'] == 2).sum()}")
    print(f"  Global avg fallback(lvl3): {(prof_features_df['fallback_level'] == 3).sum()}")
    print(f"\n  Sentiment score stats:")
    print(f"    Mean : {prof_features_df['sentiment_score'].mean():.3f}")
    print(f"    Std  : {prof_features_df['sentiment_score'].std():.3f}")
    print(f"    Min  : {prof_features_df['sentiment_score'].min():.3f}")
    print(f"    Max  : {prof_features_df['sentiment_score'].max():.3f}")

    return prof_features_df


if __name__ == "__main__":
    df = aggregate()
    print("\n--- Sample prof_features ---")
    print(df.sample(5).to_string(index=False))
