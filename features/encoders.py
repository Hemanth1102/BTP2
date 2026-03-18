"""
features/encoders.py
=====================
Provides two encoders used throughout feature engineering:

  1. BranchEncoder  — one-hot encodes branch column
  2. GradeEncoder   — converts grade string to:
                        - continuous score (0.0 to 1.0)
                        - binary label    (1 if score >= 0.6 else 0)

Both encoders are stateless and deterministic — no fitting needed
since branches and grades are fully known upfront.

Usage:
  from features.encoders import BranchEncoder, GradeEncoder

  be = BranchEncoder()
  be.encode("CSE")           → {"branch_CSE": 1, "branch_ECE": 0, ...}
  be.encode_df(df, "branch") → df with branch_* columns added

  ge = GradeEncoder()
  ge.to_score("A+")          → 0.9
  ge.to_label("A+")          → 1
  ge.encode_df(df, "grade")  → df with score and label columns added
"""

import pandas as pd
from typing import Dict

# ─────────────────────────────────────────────────────────────
# LOCKED CONSTANTS
# ─────────────────────────────────────────────────────────────

BRANCHES = ["CSE", "ECE", "ME", "CE", "EEE"]

GRADE_TO_SCORE: Dict[str, float] = {
    "O" : 1.0,
    "A+": 0.9,
    "A" : 0.8,
    "B+": 0.7,
    "B" : 0.6,
    "C" : 0.5,
    "F" : 0.0,
}

LABEL_THRESHOLD = 0.6


# ─────────────────────────────────────────────────────────────
# BRANCH ENCODER
# ─────────────────────────────────────────────────────────────

class BranchEncoder:
    """
    One-hot encodes a branch string into a fixed-length binary vector.
    Column names: branch_CSE, branch_ECE, branch_ME, branch_CE, branch_EEE
    """

    def __init__(self, branches: list = BRANCHES):
        self.branches     = branches
        self.col_names    = [f"branch_{b}" for b in branches]
        self.branch_index = {b: i for i, b in enumerate(branches)}

    def encode(self, branch: str) -> Dict[str, int]:
        """
        Encode a single branch string.
        Returns dict of {branch_CSE: 0/1, branch_ECE: 0/1, ...}
        """
        if branch not in self.branch_index:
            raise ValueError(
                f"Unknown branch '{branch}'. Expected one of {self.branches}"
            )
        return {col: int(col == f"branch_{branch}") for col in self.col_names}

    def encode_df(self, df: pd.DataFrame, branch_col: str = "branch") -> pd.DataFrame:
        """
        Add one-hot branch columns to a DataFrame.
        Original branch column is kept.
        """
        ohe = df[branch_col].apply(lambda b: pd.Series(self.encode(b)))
        return pd.concat([df, ohe], axis=1)

    @property
    def feature_names(self) -> list:
        return self.col_names


# ─────────────────────────────────────────────────────────────
# GRADE ENCODER
# ─────────────────────────────────────────────────────────────

class GradeEncoder:
    """
    Converts grade strings to continuous scores and binary labels.

    Continuous score → used as sample weight during training
    Binary label     → used as training target (BCE loss)
    """

    def __init__(self,
                 grade_map: Dict[str, float] = GRADE_TO_SCORE,
                 threshold: float            = LABEL_THRESHOLD):
        self.grade_map = grade_map
        self.threshold = threshold

    def to_score(self, grade: str) -> float:
        """O → 1.0, A+ → 0.9, ..., F → 0.0"""
        if grade not in self.grade_map:
            raise ValueError(
                f"Unknown grade '{grade}'. Expected one of {list(self.grade_map.keys())}"
            )
        return self.grade_map[grade]

    def to_label(self, grade: str) -> int:
        """score >= 0.6 → 1 (positive),  score < 0.6 → 0 (negative)"""
        return int(self.to_score(grade) >= self.threshold)

    def encode_df(self, df: pd.DataFrame, grade_col: str = "grade") -> pd.DataFrame:
        """Add 'score' and 'label' columns to a DataFrame."""
        df         = df.copy()
        df["score"] = df[grade_col].apply(self.to_score)
        df["label"] = df[grade_col].apply(self.to_label)
        return df

    @property
    def all_grades(self) -> list:
        return list(self.grade_map.keys())


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Branch encoder test
    be = BranchEncoder()
    print("Branch Encoder")
    print("--------------")
    for branch in BRANCHES:
        vec = list(be.encode(branch).values())
        print(f"  {branch:4s} → {vec}")

    test_df    = pd.DataFrame({"student_id": ["S1", "S2", "S3"],
                                "branch"    : ["CSE", "ME", "EEE"]})
    encoded_df = be.encode_df(test_df)
    print(f"\n  DataFrame OHE:\n{encoded_df.to_string(index=False)}")

    # Grade encoder test
    ge = GradeEncoder()
    print("\n\nGrade Encoder")
    print("-------------")
    print(f"  {'Grade':<6} {'Score':<8} {'Label'}")
    for grade in ge.all_grades:
        score = ge.to_score(grade)
        label = ge.to_label(grade)
        note  = " ← boundary" if score == LABEL_THRESHOLD else ""
        print(f"  {grade:<6} {score:<8.1f} {label}{note}")

    test_df2    = pd.DataFrame({"student_id": ["S1", "S2", "S3", "S4"],
                                 "grade"     : ["O", "B", "C", "F"]})
    encoded_df2 = ge.encode_df(test_df2)
    print(f"\n  DataFrame grade encoding:\n{encoded_df2.to_string(index=False)}")
