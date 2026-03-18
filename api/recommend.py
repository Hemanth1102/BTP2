"""
api/recommend.py
=================
FastAPI endpoint for OE recommendations.

Endpoints:
  GET  /recommend/{student_id}?semester=5
       → returns ranked list of all eligible OEs for the student

  GET  /health
       → returns model status and last checkpoint info

  GET  /student/{student_id}
       → returns student profile and OE history

Usage:
  pip install fastapi uvicorn
  uvicorn api.recommend:app --reload

  Then visit:
  http://localhost:8000/recommend/STU0001?semester=5
  http://localhost:8000/docs   ← auto-generated Swagger UI
"""

import os
import torch
import numpy as np
import pandas as pd
from fastapi         import FastAPI, HTTPException
from pydantic        import BaseModel
from typing          import List, Optional
from datetime        import datetime

from model.dataset  import OEDataset, STUDENT_FEATURE_COLS, OE_FEATURE_COLS
from model.neumf    import NeuMF
from model.train    import load_checkpoint, EMBEDDING_DIM, MLP_LAYERS, DROPOUT_RATE
from features.cold_start import ColdStartHandler

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

CHECKPOINT_DIR  = "checkpoints"
PROCESSED_DIR   = "data/processed"
RAW_DIR         = "data/raw"
OE_SEMESTERS    = [5, 6, 7]

app = FastAPI(
    title       = "OE Recommendation System",
    description = "Recommends Open Electives for students using NeuMF",
    version     = "1.0.0",
)


# ─────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────

class OERecommendation(BaseModel):
    rank            : int
    oe_id           : str
    offering_branch : str
    predicted_score : float
    is_new_oe       : int
    prof_id         : str

class RecommendationResponse(BaseModel):
    student_id      : str
    semester        : int
    branch          : str
    cgpa            : float
    total_eligible  : int
    is_cold_start   : bool
    recommendations : List[OERecommendation]
    generated_at    : str

class HealthResponse(BaseModel):
    status          : str
    checkpoint      : str
    model_params    : int
    device          : str
    last_updated    : str


# ─────────────────────────────────────────────────────────────
# MODEL LOADER — loads once at startup
# ─────────────────────────────────────────────────────────────

class ModelLoader:
    """Loads model and all lookup tables once at startup."""

    def __init__(self):
        self.model            = None
        self.device           = None
        self.student_lookup   = None
        self.oe_lookup        = None
        self.students_df      = None
        self.oe_info_df       = None
        self.interaction_df   = None
        self.cold_start       = None
        self.checkpoint_path  = None
        self.checkpoint_info  = {}

    def load(self):
        print("Loading model and data...")

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device : {self.device}")

        # Find best available checkpoint
        for sem in [7, 6, 5]:
            path = f"{CHECKPOINT_DIR}/model_sem{sem}.pt"
            if os.path.exists(path):
                self.checkpoint_path = path
                break

        if not self.checkpoint_path:
            raise FileNotFoundError(
                "No checkpoint found. Run model/train.py first."
            )

        # Load dataset for feature lookups
        # Use any split — we just need the lookup dicts
        ds = OEDataset(split="train")
        self.student_lookup = ds._student_lookup
        self.oe_lookup      = ds._oe_lookup

        # Build model
        self.model = NeuMF(
            student_dim   = ds.student_dim,
            oe_dim        = ds.oe_dim,
            embedding_dim = EMBEDDING_DIM,
            mlp_layers    = MLP_LAYERS,
            dropout       = DROPOUT_RATE,
        ).to(self.device)

        # Load checkpoint
        ckpt = load_checkpoint(self.checkpoint_path, self.model)
        self.model.eval()

        self.checkpoint_info = {
            "path"    : self.checkpoint_path,
            "epoch"   : ckpt.get("epoch", -1),
            "val_loss": ckpt.get("val_loss", -1),
            "ndcg"    : ckpt.get("ndcg@10", -1),
            "semester": ckpt.get("semester", -1),
        }

        # Load raw tables
        self.students_df    = pd.read_csv(f"{RAW_DIR}/students.csv")
        self.oe_info_df     = pd.read_csv(f"{RAW_DIR}/oe_info.csv")
        self.interaction_df = pd.read_csv(f"{PROCESSED_DIR}/interaction_matrix.csv")

        # Cold start handler
        self.cold_start = ColdStartHandler()
        self.cold_start.load()

        print(f"  Checkpoint : {self.checkpoint_path}")
        print(f"  NDCG@10    : {self.checkpoint_info['ndcg']:.4f}")
        print(f"  Model ready\n")


# Singleton loader — loads once when server starts
loader = ModelLoader()


@app.on_event("startup")
def startup_event():
    loader.load()


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def get_student(student_id: str) -> pd.Series:
    row = loader.students_df[loader.students_df["student_id"] == student_id]
    if row.empty:
        raise HTTPException(status_code=404,
                            detail=f"Student '{student_id}' not found")
    return row.iloc[0]


def get_eligible_oes(student_id: str,
                     branch    : str,
                     semester  : int) -> pd.DataFrame:
    """
    Returns eligible OEs for this student:
      - correct semester
      - not student's own branch
      - not already taken by this student
    """
    # OEs already taken by this student
    taken = set(
        loader.interaction_df[
            (loader.interaction_df["student_id"]  == student_id) &
            (loader.interaction_df["is_negative"] == 0)
        ]["oe_id"].tolist()
    )

    eligible = loader.oe_info_df[
        (loader.oe_info_df["available_semester"] == semester) &
        (loader.oe_info_df["offering_branch"]    != branch) &
        (~loader.oe_info_df["oe_id"].isin(taken))
    ].copy()

    return eligible


def score_oes(student_id   : str,
              eligible_oes : pd.DataFrame) -> List[dict]:
    """
    Score all eligible OEs for a student using NeuMF.
    Returns list of dicts sorted by predicted score descending.
    """
    if student_id not in loader.student_lookup:
        raise HTTPException(status_code=404,
                            detail=f"Student features not found for '{student_id}'")

    student_vec = torch.tensor(
        loader.student_lookup[student_id], dtype=torch.float32
    ).unsqueeze(0).to(loader.device)

    scored = []
    with torch.no_grad():
        for _, oe_row in eligible_oes.iterrows():
            oe_id = oe_row["oe_id"]

            if oe_id not in loader.oe_lookup:
                continue

            oe_vec = torch.tensor(
                loader.oe_lookup[oe_id], dtype=torch.float32
            ).unsqueeze(0).to(loader.device)

            score = loader.model(student_vec, oe_vec).item()

            scored.append({
                "oe_id"          : oe_id,
                "offering_branch": oe_row["offering_branch"],
                "predicted_score": round(score, 4),
                "is_new_oe"      : int(oe_row["is_new_oe"]),
                "prof_id"        : str(oe_row["prof_id"]),
            })

    # Sort by predicted score descending
    scored.sort(key=lambda x: x["predicted_score"], reverse=True)

    # Add rank
    for i, item in enumerate(scored):
        item["rank"] = i + 1

    return scored


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Check model status and checkpoint info."""
    return HealthResponse(
        status       = "ok",
        checkpoint   = loader.checkpoint_path or "none",
        model_params = loader.model.count_parameters() if loader.model else 0,
        device       = str(loader.device),
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


@app.get("/recommend/{student_id}", response_model=RecommendationResponse)
def recommend(student_id: str, semester: int = 5):
    """
    Get ranked OE recommendations for a student.

    Args:
      student_id : e.g. STU0001
      semester   : 5, 6, or 7

    Returns:
      Full ranked list of all eligible OEs (typically 32)
    """
    if semester not in OE_SEMESTERS:
        raise HTTPException(
            status_code=400,
            detail=f"Semester must be one of {OE_SEMESTERS}"
        )

    # Get student info
    student      = get_student(student_id)
    branch       = student["branch"]
    cgpa         = float(student["cgpa"])
    is_cold_start = loader.cold_start.is_cold_start_student(student_id)

    # Get eligible OEs
    eligible_oes = get_eligible_oes(student_id, branch, semester)

    if eligible_oes.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No eligible OEs found for student '{student_id}' "
                   f"in semester {semester}"
        )

    # Cold start → use content-based recommendations
    if is_cold_start:
        cold_recs = loader.cold_start.recommend_for_new_student(
            student_id, semester
        )
        if not cold_recs.empty:
            # Merge cold start scores with eligible OEs
            eligible_oes = eligible_oes.merge(
                cold_recs[["oe_id", "avg_similar_score"]],
                on="oe_id", how="left"
            )
            eligible_oes["avg_similar_score"] = \
                eligible_oes["avg_similar_score"].fillna(0.0)

    # Score with NeuMF
    scored = score_oes(student_id, eligible_oes)

    return RecommendationResponse(
        student_id      = student_id,
        semester        = semester,
        branch          = branch,
        cgpa            = cgpa,
        total_eligible  = len(scored),
        is_cold_start   = is_cold_start,
        recommendations = [OERecommendation(**item) for item in scored],
        generated_at    = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


@app.get("/student/{student_id}")
def student_profile(student_id: str):
    """Get student profile and their OE history."""
    student = get_student(student_id)

    history = loader.interaction_df[
        (loader.interaction_df["student_id"]  == student_id) &
        (loader.interaction_df["is_negative"] == 0)
    ][["oe_id", "grade", "score", "semester"]].to_dict(orient="records")

    return {
        "student_id" : student_id,
        "branch"     : student["branch"],
        "cgpa"       : float(student["cgpa"]),
        "batch_year" : int(student["batch_year"]),
        "oe_history" : history,
    }
