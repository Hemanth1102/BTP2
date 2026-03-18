"""
features/cold_start.py
=======================
Handles cold start scenarios:

  1. cold_start_student → student has no OE history yet (sem 5 first-timers)
                          Strategy: find top-k similar students using
                          branch + cgpa + core grade profile,
                          recommend what they chose and did well in

  2. cold_start_oe      → brand new OE, no interaction history
                          Strategy: find top-k similar OEs using
                          cosine similarity on OE feature vector,
                          inherit their interaction patterns

  3. cold_start_prof    → already handled in aggregate_sentiment.py
                          via three-level fallback chain

Usage:
  from features.cold_start import ColdStartHandler

  handler = ColdStartHandler()
  handler.load()

  # For a new student
  recs = handler.recommend_for_new_student(student_id, current_semester)

  # For a new OE
  similar = handler.find_similar_oes(oe_id, top_k=5)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing   import MinMaxScaler

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

PROCESSED_DIR = "data/processed"
RAW_DIR       = "data/raw"

TOP_K_STUDENTS = 10     # how many similar students to look at
TOP_K_OES      = 5      # how many similar OEs to inherit from
MIN_GRADE_SCORE = 0.7   # only recommend OEs where similar students got B+ or above


# ─────────────────────────────────────────────────────────────
# COLD START HANDLER
# ─────────────────────────────────────────────────────────────

class ColdStartHandler:

    def __init__(self,
                 processed_dir: str = PROCESSED_DIR,
                 raw_dir: str       = RAW_DIR):
        self.processed_dir = processed_dir
        self.raw_dir       = raw_dir

        # Loaded in self.load()
        self.student_features_df   = None
        self.oe_features_df        = None
        self.interaction_matrix_df = None
        self.students_df           = None
        self.oe_info_df            = None

        # Scaled feature matrices for similarity search
        self._student_matrix = None
        self._oe_matrix      = None
        self._scaler_student = MinMaxScaler()
        self._scaler_oe      = MinMaxScaler()

    # ─────────────────────────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────────────────────────

    def load(self):
        """Load all required tables and precompute scaled feature matrices."""
        self.student_features_df   = pd.read_csv(f"{self.processed_dir}/student_features.csv")
        self.oe_features_df        = pd.read_csv(f"{self.processed_dir}/oe_features.csv")
        self.interaction_matrix_df = pd.read_csv(f"{self.processed_dir}/interaction_matrix.csv")
        self.students_df           = pd.read_csv(f"{self.raw_dir}/students.csv")
        self.oe_info_df            = pd.read_csv(f"{self.raw_dir}/oe_info.csv")

        self._build_student_matrix()
        self._build_oe_matrix()

        print("ColdStartHandler loaded")
        print(f"  Students : {len(self.student_features_df)}")
        print(f"  OEs      : {len(self.oe_features_df)}")

    # ─────────────────────────────────────────────────────────
    # BUILD FEATURE MATRICES FOR SIMILARITY SEARCH
    # ─────────────────────────────────────────────────────────

    def _build_student_matrix(self):
        """
        Build scaled numeric matrix for student similarity.
        Uses: branch OHE, cgpa, avg_core_grade, sem1-4 averages
        """
        student_cols = (
            [c for c in self.student_features_df.columns if c.startswith("branch_")] +
            ["cgpa", "avg_core_grade", "sem1_avg", "sem2_avg", "sem3_avg", "sem4_avg"]
        )
        matrix = self.student_features_df[student_cols].values.astype(float)
        self._student_matrix = self._scaler_student.fit_transform(matrix)

    def _build_oe_matrix(self):
        """
        Build scaled numeric matrix for OE similarity.
        Uses: branch OHE, available_semester, rating cols, sentiment_score
        """
        oe_cols = (
            [c for c in self.oe_features_df.columns if c.startswith("branch_")] +
            ["available_semester",
             "oe_avg_teaching_clarity", "oe_avg_course_organization",
             "oe_avg_overall_rating",   "oe_avg_interaction",
             "oe_avg_assignment_usefulness", "sentiment_score"]
        )
        matrix = self.oe_features_df[oe_cols].values.astype(float)
        self._oe_matrix = self._scaler_oe.fit_transform(matrix)

    # ─────────────────────────────────────────────────────────
    # COLD START — NEW STUDENT
    # ─────────────────────────────────────────────────────────

    def recommend_for_new_student(self,
                                   student_id: str,
                                   current_semester: int,
                                   top_k: int = TOP_K_STUDENTS) -> pd.DataFrame:
        """
        For a student with no OE history:
          1. Find top-k most similar students using cosine similarity
          2. Collect OEs those students took and did well in (score >= MIN_GRADE_SCORE)
          3. Filter to current semester + not student's own branch
          4. Rank by how frequently similar students chose + how well they did

        Returns DataFrame: oe_id, score, recommendation_count, avg_similar_score
        """
        if self.student_features_df is None:
            raise RuntimeError("Call .load() before using ColdStartHandler")

        # Get this student's branch for exclusion filter
        student_row = self.students_df[self.students_df["student_id"] == student_id]
        if student_row.empty:
            raise ValueError(f"Student '{student_id}' not found")
        student_branch = student_row["branch"].values[0]

        # Get student's position in feature matrix
        student_idx = self.student_features_df[
            self.student_features_df["student_id"] == student_id
        ].index

        if student_idx.empty:
            raise ValueError(f"Student '{student_id}' not found in student_features")

        idx = student_idx[0]

        # Compute cosine similarity against all students
        student_vec  = self._student_matrix[idx].reshape(1, -1)
        similarities = cosine_similarity(student_vec, self._student_matrix)[0]

        # Top-k similar students (exclude self)
        sim_indices = np.argsort(similarities)[::-1]
        sim_indices = [i for i in sim_indices if i != idx][:top_k]
        sim_ids     = self.student_features_df.iloc[sim_indices]["student_id"].tolist()

        # Get their positive OE interactions for current_semester
        similar_interactions = self.interaction_matrix_df[
            (self.interaction_matrix_df["student_id"].isin(sim_ids)) &
            (self.interaction_matrix_df["semester"]   == current_semester) &
            (self.interaction_matrix_df["is_negative"] == 0) &
            (self.interaction_matrix_df["score"]       >= MIN_GRADE_SCORE)
        ]

        if similar_interactions.empty:
            print(f"  No similar student history found for {student_id} in sem {current_semester}")
            return pd.DataFrame(columns=["oe_id", "avg_similar_score", "recommendation_count"])

        # Filter: not student's own branch
        eligible_oes = self.oe_info_df[
            (self.oe_info_df["available_semester"] == current_semester) &
            (self.oe_info_df["offering_branch"]    != student_branch)
        ]["oe_id"].tolist()

        similar_interactions = similar_interactions[
            similar_interactions["oe_id"].isin(eligible_oes)
        ]

        # Aggregate: count how many similar students chose it + avg score
        agg = (
            similar_interactions
            .groupby("oe_id")
            .agg(
                recommendation_count = ("student_id", "count"),
                avg_similar_score    = ("score", "mean"),
            )
            .reset_index()
            .sort_values(["recommendation_count", "avg_similar_score"],
                          ascending=False)
        )

        return agg

    # ─────────────────────────────────────────────────────────
    # COLD START — NEW OE
    # ─────────────────────────────────────────────────────────

    def find_similar_oes(self,
                          oe_id: str,
                          top_k: int = TOP_K_OES) -> pd.DataFrame:
        """
        For a new OE with no interaction history:
          1. Find top-k most similar existing OEs using cosine similarity
          2. Return their oe_ids and similarity scores
          → caller uses these to warm-start the OE's embedding

        Returns DataFrame: oe_id, similarity_score
        """
        if self.oe_features_df is None:
            raise RuntimeError("Call .load() before using ColdStartHandler")

        oe_idx = self.oe_features_df[
            self.oe_features_df["oe_id"] == oe_id
        ].index

        if oe_idx.empty:
            raise ValueError(f"OE '{oe_id}' not found in oe_features")

        idx     = oe_idx[0]
        oe_vec  = self._oe_matrix[idx].reshape(1, -1)
        sims    = cosine_similarity(oe_vec, self._oe_matrix)[0]

        # Top-k similar OEs (exclude self)
        sim_indices = np.argsort(sims)[::-1]
        sim_indices = [i for i in sim_indices if i != idx][:top_k]

        result = pd.DataFrame({
            "oe_id"            : self.oe_features_df.iloc[sim_indices]["oe_id"].values,
            "similarity_score" : sims[sim_indices].round(4),
        })

        return result

    # ─────────────────────────────────────────────────────────
    # UTILITY — check if student is cold start
    # ─────────────────────────────────────────────────────────

    def is_cold_start_student(self, student_id: str) -> bool:
        """Returns True if student has no OE interaction history."""
        history = self.interaction_matrix_df[
            (self.interaction_matrix_df["student_id"]  == student_id) &
            (self.interaction_matrix_df["is_negative"] == 0)
        ]
        return len(history) == 0

    def is_cold_start_oe(self, oe_id: str) -> bool:
        """Returns True if OE has no interaction history."""
        history = self.interaction_matrix_df[
            (self.interaction_matrix_df["oe_id"]       == oe_id) &
            (self.interaction_matrix_df["is_negative"] == 0)
        ]
        return len(history) == 0


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    handler = ColdStartHandler()
    handler.load()

    # Test 1 — new student recommendation
    print("\n--- Cold Start: New Student ---")
    test_student = handler.student_features_df["student_id"].iloc[0]
    recs = handler.recommend_for_new_student(test_student, current_semester=5)
    print(f"  Student : {test_student}")
    print(f"  Top recommendations from similar students:")
    print(recs.head(5).to_string(index=False))

    # Test 2 — new OE similarity
    print("\n--- Cold Start: New OE ---")
    test_oe    = handler.oe_features_df["oe_id"].iloc[0]
    similar_oes = handler.find_similar_oes(test_oe, top_k=5)
    print(f"  OE      : {test_oe}")
    print(f"  Most similar existing OEs:")
    print(similar_oes.to_string(index=False))

    # Test 3 — cold start checks
    print("\n--- Cold Start Flags ---")
    print(f"  is_cold_start_student('{test_student}') : "
          f"{handler.is_cold_start_student(test_student)}")
    print(f"  is_cold_start_oe('{test_oe}')           : "
          f"{handler.is_cold_start_oe(test_oe)}")
