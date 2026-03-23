"""
features/feature_engineering.py
=================================
Joins all raw tables and prof_features to produce:

  1. student_features.csv   → one row per student
  2. oe_features.csv        → one row per OE
  3. interaction_matrix.csv → positive + negative (1:4) interactions

Negative sampling rules:
  - sampled only from student's eligible pool
    (correct semester + not own branch + not already taken)
  - ratio: 1 positive : 4 negatives
  - negatives get score=0.0, label=0, is_negative=1

Train/val/test split (time-based, never shuffle across semesters):
  sem 5 → train
  sem 6 → val
  sem 7 → test

Inputs:
  data/raw/students.csv
  data/raw/student_courses.csv
  data/raw/student_oe.csv
  data/raw/oe_info.csv
  data/raw/course_feedback.csv
  data/processed/prof_features.csv

Outputs:
  data/processed/student_features.csv
  data/processed/oe_features.csv
  data/processed/interaction_matrix.csv
"""

import os
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from features.encoders import BranchEncoder, GradeEncoder

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"

NEGATIVE_RATIO = 4        # 1 positive : 4 negatives
OE_SEMESTERS   = [5, 6, 7]

RATING_COLS = [
    "avg_teaching_clarity",
    "avg_course_organization",
    "avg_overall_rating",
    "avg_interaction",
    "avg_assignment_usefulness",
]

# Semester split — locked decision
SPLIT_MAP = {5: "train", 6: "val", 7: "test"}


# ─────────────────────────────────────────────────────────────
# LOAD ALL TABLES
# ─────────────────────────────────────────────────────────────

def load_tables():
    students_df        = pd.read_csv(f"{RAW_DIR}/students.csv")
    student_courses_df = pd.read_csv(f"{RAW_DIR}/student_courses.csv")
    student_oe_df      = pd.read_csv(f"{RAW_DIR}/student_oe.csv")
    oe_info_df         = pd.read_csv(f"{RAW_DIR}/oe_info.csv")
    prof_features_df   = pd.read_csv(f"{PROCESSED_DIR}/prof_features.csv")

    print("Tables loaded:")
    print(f"  students         : {len(students_df)} rows")
    print(f"  student_courses  : {len(student_courses_df)} rows")
    print(f"  student_oe       : {len(student_oe_df)} rows")
    print(f"  oe_info          : {len(oe_info_df)} rows")
    print(f"  prof_features    : {len(prof_features_df)} rows")

    return students_df, student_courses_df, student_oe_df, oe_info_df, prof_features_df


# ─────────────────────────────────────────────────────────────
# BUILD STUDENT FEATURES
# ─────────────────────────────────────────────────────────────

def build_student_features(students_df: pd.DataFrame,
                            student_courses_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each student:
      - OHE branch
      - cgpa (raw)
      - avg_core_grade: average grade score across all core courses
      - sem{1..4}_avg:  average grade score per semester (sems 1-4 only)
                        sem 5,6 are OE semesters — excluded to avoid leakage
    """
    ge = GradeEncoder()
    be = BranchEncoder()

    # Convert core course grades to scores
    courses = student_courses_df.copy()
    courses["score"] = courses["grade"].apply(ge.to_score)

    # Overall core grade average per student
    overall_avg = (
        courses.groupby("student_id")["score"]
        .mean()
        .round(4)
        .reset_index()
        .rename(columns={"score": "avg_core_grade"})
    )

    # Per-semester average (sem 1–4 only, avoid leakage from sem 5+)
    sem_avg = (
        courses[courses["semester"] <= 4]
        .groupby(["student_id", "semester"])["score"]
        .mean()
        .round(4)
        .unstack(level="semester")
    )
    sem_avg.columns = [f"sem{int(c)}_avg" for c in sem_avg.columns]
    sem_avg         = sem_avg.reset_index().fillna(0.0)

    # Merge everything
    student_features = students_df[["student_id", "branch", "cgpa"]].copy()
    student_features = student_features.merge(overall_avg, on="student_id", how="left")
    student_features = student_features.merge(sem_avg,     on="student_id", how="left")

    # OHE branch
    student_features = be.encode_df(student_features, "branch")
    
    # Final column order
    branch_cols = be.feature_names
    sem_cols    = [f"sem{s}_avg" for s in range(1, 5)]
    final_cols  = ["student_id"] + branch_cols + ["cgpa", "avg_core_grade"] + sem_cols
    student_features = student_features[final_cols]

    # Normalize continuous columns only
    # Branch OHE is already 0/1 — do not normalize
    cols_to_normalize = ["cgpa", "avg_core_grade",
                         "sem1_avg", "sem2_avg", "sem3_avg", "sem4_avg"]

    # Fix: fit scaler ONLY on training students (sem 5) to prevent data leakage
    # Training students are those whose OE semester is 5
    # We identify them via the student_id list — passed in from run()
    # For now we return features unscaled and let run() handle the split-aware scaling
    # (see run() below for the correct fit/transform logic)

    print(f"\nStudent features built: {len(student_features)} rows, "
          f"{len(student_features.columns)} columns")
    return student_features


# ─────────────────────────────────────────────────────────────
# BUILD OE FEATURES
# ─────────────────────────────────────────────────────────────

def build_oe_features(oe_info_df: pd.DataFrame,
                      prof_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each OE:
      - OHE offering_branch
      - sem5, sem6, sem7 (OHE of available_semester — categorical not continuous)
      - professor rating columns (from prof_features)
      - sentiment_score (from prof_features)
      - is_new_oe flag
      - is_new_prof flag
    """
    be = BranchEncoder()

    # Join OE info with professor features
    oe_features = oe_info_df[[
        "oe_id", "offering_branch", "available_semester",
        "prof_id", "is_new_oe", "total_seats"
    ]].merge(
        prof_features_df[[
            "prof_id", "is_new_prof", "sentiment_score"
        ] + RATING_COLS],
        on="prof_id",
        how="left",
    )

    # OHE offering_branch
    oe_features = be.encode_df(oe_features, "offering_branch")

    # Rename rating cols to oe_ prefix to distinguish from raw feedback cols
    rename_map  = {col: f"oe_{col}" for col in RATING_COLS}
    oe_features = oe_features.rename(columns=rename_map)

    branch_cols = be.feature_names

    # OHE available_semester — treat as categorical not continuous
    # sem5, sem6, sem7 are distinct categories with no ordinal relationship
    oe_features["sem5"] = (oe_features["available_semester"] == 5).astype(int)
    oe_features["sem6"] = (oe_features["available_semester"] == 6).astype(int)
    oe_features["sem7"] = (oe_features["available_semester"] == 7).astype(int)

    # Final column order — replace available_semester with sem5/sem6/sem7
    sem_cols    = ["sem5", "sem6", "sem7"]
    rating_cols = [f"oe_{c}" for c in RATING_COLS]
    final_cols  = (
        ["oe_id"] + branch_cols + sem_cols + rating_cols +
        ["sentiment_score", "is_new_oe", "is_new_prof"]
    )
    oe_features = oe_features[final_cols]

    # Normalize OE continuous columns only
    # Rating columns are 1-5 scale → normalize to 0-1
    # sem OHE is already 0/1, branch OHE is 0/1, flags are 0/1
    # sentiment_score is already 0-1 but normalize for consistency
    oe_cols_to_normalize = [f"oe_{c}" for c in RATING_COLS] + ["sentiment_score"]

    oe_scaler = MinMaxScaler()
    oe_features[oe_cols_to_normalize] = oe_scaler.fit_transform(
        oe_features[oe_cols_to_normalize]
    ).round(4)

    # Save OE scaler — must be applied at inference time
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    joblib.dump(oe_scaler, f"{PROCESSED_DIR}/oe_scaler.pkl")
    print(f"  OE scaler saved to {PROCESSED_DIR}/oe_scaler.pkl")

    print(f"OE features built     : {len(oe_features)} rows, "
          f"{len(oe_features.columns)} columns  (sem OHE added, dim=16)")
    return oe_features


# ─────────────────────────────────────────────────────────────
# BUILD INTERACTION MATRIX (positives + negatives)
# ─────────────────────────────────────────────────────────────

def build_interaction_matrix(students_df: pd.DataFrame,
                              student_oe_df: pd.DataFrame,
                              oe_info_df: pd.DataFrame) -> pd.DataFrame:
    """
    Positive interactions: all rows from student_oe
    Negative interactions: NEGATIVE_RATIO × negatives per positive
      - sampled from eligible pool only
        (correct semester, not own branch, not already taken)

    Each row: student_id, oe_id, grade, score, label,
              semester, split, is_negative
    """
    ge = GradeEncoder()

    # Encode positives
    positives           = student_oe_df.copy()
    positives           = ge.encode_df(positives, "grade")
    positives["is_negative"] = 0
    positives["split"]  = positives["semester"].map(SPLIT_MAP)

    # Build lookup: student → set of OEs they already took
    student_taken = (
        student_oe_df.groupby("student_id")["oe_id"]
        .apply(set)
        .to_dict()
    )

    # Branch per student
    student_branch = students_df.set_index("student_id")["branch"].to_dict()

    # Eligible OEs per (branch, semester)
    eligible_map = {}
    for sem in OE_SEMESTERS:
        for branch in students_df["branch"].unique():
            key = (branch, sem)
            eligible_map[key] = oe_info_df[
                (oe_info_df["available_semester"] == sem) &
                (oe_info_df["offering_branch"]    != branch)
            ]["oe_id"].tolist()

    # Sample negatives
    negative_rows = []
    for _, pos_row in positives.iterrows():
        student_id = pos_row["student_id"]
        sem        = pos_row["semester"]
        branch     = student_branch[student_id]
        taken      = student_taken.get(student_id, set())

        # Pool = eligible for this sem - already taken
        pool = [
            oe for oe in eligible_map.get((branch, sem), [])
            if oe not in taken
        ]

        # Sample up to NEGATIVE_RATIO negatives
        n_samples = min(NEGATIVE_RATIO, len(pool))
        if n_samples == 0:
            continue

        sampled = random.sample(pool, n_samples)
        for oe_id in sampled:
            negative_rows.append({
                "student_id" : student_id,
                "oe_id"      : oe_id,
                "grade"      : None,
                "score"      : 0.0,
                "label"      : 0,
                "semester"   : sem,
                "split"      : SPLIT_MAP[sem],
                "is_negative": 1,
            })

    negatives = pd.DataFrame(negative_rows)

    # Combine
    interaction_matrix = pd.concat([positives, negatives], ignore_index=True)
    interaction_matrix = interaction_matrix.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Interaction matrix    : {len(interaction_matrix)} rows")
    print(f"  Positives           : {(interaction_matrix['is_negative'] == 0).sum()}")
    print(f"  Negatives           : {(interaction_matrix['is_negative'] == 1).sum()}")
    print(f"  Train (sem 5)       : {(interaction_matrix['split'] == 'train').sum()}")
    print(f"  Val   (sem 6)       : {(interaction_matrix['split'] == 'val').sum()}")
    print(f"  Test  (sem 7)       : {(interaction_matrix['split'] == 'test').sum()}")

    return interaction_matrix


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    students_df, student_courses_df, student_oe_df, \
        oe_info_df, prof_features_df = load_tables()

    print("\nBuilding features...")

    student_features   = build_student_features(students_df, student_courses_df)
    oe_features        = build_oe_features(oe_info_df, prof_features_df)
    interaction_matrix = build_interaction_matrix(students_df, student_oe_df, oe_info_df)

    # ── Split-aware student normalization ────────────────────────
    # Fix data leakage: fit scaler ONLY on training students (sem 5)
    # then transform val (sem 6) and test (sem 7) students separately
    cols_to_normalize = ["cgpa", "avg_core_grade",
                         "sem1_avg", "sem2_avg", "sem3_avg", "sem4_avg"]

    # Get training student IDs — those who appear in sem 5 interactions
    train_student_ids = interaction_matrix[
        interaction_matrix["split"] == "train"
    ]["student_id"].unique()

    train_mask = student_features["student_id"].isin(train_student_ids)

    # Fit ONLY on training students
    student_scaler = MinMaxScaler()
    student_scaler.fit(student_features.loc[train_mask, cols_to_normalize])

    # Transform ALL students using training-fitted scaler
    student_features[cols_to_normalize] = student_scaler.transform(
        student_features[cols_to_normalize]
    ).round(4)

    # Save scaler
    joblib.dump(student_scaler, f"{PROCESSED_DIR}/student_scaler.pkl")
    print(f"  Student scaler fitted on {train_mask.sum()} training students")
    print(f"  Student scaler saved to {PROCESSED_DIR}/student_scaler.pkl")

    # Save
    student_features.to_csv(f"{PROCESSED_DIR}/student_features.csv",   index=False)
    oe_features.to_csv(f"{PROCESSED_DIR}/oe_features.csv",             index=False)
    interaction_matrix.to_csv(f"{PROCESSED_DIR}/interaction_matrix.csv", index=False)

    print(f"\n✓ Saved all feature files to {PROCESSED_DIR}/")

    # Sanity checks
    print("\n--- Sanity Checks ---")

    # No student feature NaNs
    nan_count = student_features.isnull().sum().sum()
    print(f"  Student feature NaNs     : {nan_count}  (expected 0)")

    # No OE feature NaNs
    nan_count_oe = oe_features.isnull().sum().sum()
    print(f"  OE feature NaNs          : {nan_count_oe}  (expected 0)")

    # Positive/negative ratio check
    pos = (interaction_matrix["is_negative"] == 0).sum()
    neg = (interaction_matrix["is_negative"] == 1).sum()
    print(f"  Neg/pos ratio            : {neg/pos:.2f}  (expected ~{NEGATIVE_RATIO}.00)")

    # No positives in negative rows
    neg_with_label = interaction_matrix[
        (interaction_matrix["is_negative"] == 1) &
        (interaction_matrix["label"] == 1)
    ]
    print(f"  Negatives with label=1   : {len(neg_with_label)}  (expected 0)")
    print("---------------------")

    return student_features, oe_features, interaction_matrix


if __name__ == "__main__":
    run()
