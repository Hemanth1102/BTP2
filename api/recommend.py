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

import joblib
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
        self.student_scaler   = None
        self.oe_scaler        = None

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

        # Load student scaler
        scaler_path = f"{PROCESSED_DIR}/student_scaler.pkl"
        if os.path.exists(scaler_path):
            self.student_scaler = joblib.load(scaler_path)
            print(f"  Student scaler loaded from {scaler_path}")
        else:
            print(f"  Warning: student_scaler.pkl not found — student features not normalized")

        # Load OE scaler
        oe_scaler_path = f"{PROCESSED_DIR}/oe_scaler.pkl"
        if os.path.exists(oe_scaler_path):
            self.oe_scaler = joblib.load(oe_scaler_path)
            print(f"  OE scaler loaded from {oe_scaler_path}")
        else:
            print(f"  Warning: oe_scaler.pkl not found — OE features not normalized")

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

    # Apply same normalization used during training
    raw_vec = loader.student_lookup[student_id].copy()
    if loader.student_scaler is not None:
        from model.dataset import STUDENT_FEATURE_COLS
        branch_cols = [c for c in STUDENT_FEATURE_COLS if c.startswith("branch_")]
        cont_cols   = [c for c in STUDENT_FEATURE_COLS if not c.startswith("branch_")]
        cont_idx    = [STUDENT_FEATURE_COLS.index(c) for c in cont_cols]
        import numpy as np
        raw_vec[cont_idx] = loader.student_scaler.transform(
            raw_vec[cont_idx].reshape(1, -1)
        ).flatten()

    student_vec = torch.tensor(raw_vec, dtype=torch.float32).unsqueeze(0).to(loader.device)

    from model.dataset import OE_FEATURE_COLS

    # Columns that were normalized during feature engineering
    # Exclude: branch OHE, sem OHE, binary flags — all already 0/1
    oe_cont_cols = [c for c in OE_FEATURE_COLS
                    if not c.startswith("branch_")
                    and c not in ("sem5", "sem6", "sem7",
                                  "is_new_oe", "is_new_prof")]
    oe_cont_idx  = [OE_FEATURE_COLS.index(c) for c in oe_cont_cols]

    scored = []
    with torch.no_grad():
        for _, oe_row in eligible_oes.iterrows():
            oe_id = oe_row["oe_id"]

            if oe_id not in loader.oe_lookup:
                continue

            raw_oe = loader.oe_lookup[oe_id].copy()
            if loader.oe_scaler is not None:
                import numpy as np
                raw_oe[oe_cont_idx] = loader.oe_scaler.transform(
                    raw_oe[oe_cont_idx].reshape(1, -1)
                ).flatten()

            oe_vec = torch.tensor(raw_oe, dtype=torch.float32).unsqueeze(0).to(loader.device)
            score  = loader.model(student_vec, oe_vec).item()

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


@app.get("/explain/{student_id}/{oe_id}")
def explain(student_id: str, oe_id: str, semester: int = 5):
    """
    Explain why a specific OE was ranked where it was for a student.

    Uses SHAP values to show the contribution of each feature
    to the predicted relevance score.

    Example:
      GET /explain/STU0001/ME_OE501_Robotics?semester=5

    Returns:
      predicted_score   : model's relevance score for this OE
      feature_contributions : each feature's SHAP contribution
      top_reasons       : top 3 human-readable reasons
      bottom_reasons    : bottom 3 factors pulling score down
    """
    import shap
    import numpy as np
    from model.dataset import STUDENT_FEATURE_COLS, OE_FEATURE_COLS

    # Validate student and OE exist
    student = get_student(student_id)
    branch  = student["branch"]

    oe_row  = loader.oe_info_df[loader.oe_info_df["oe_id"] == oe_id]
    if oe_row.empty:
        raise HTTPException(status_code=404,
                            detail=f"OE '{oe_id}' not found")

    if student_id not in loader.student_lookup:
        raise HTTPException(status_code=404,
                            detail=f"Student features not found for '{student_id}'")

    if oe_id not in loader.oe_lookup:
        raise HTTPException(status_code=404,
                            detail=f"OE features not found for '{oe_id}'")

    # Get feature vectors
    student_vec_np = loader.student_lookup[student_id]   # shape (11,)
    oe_vec_np      = loader.oe_lookup[oe_id]             # shape (14,)

    # Combined vector for SHAP: [student || oe]
    combined       = np.concatenate([student_vec_np, oe_vec_np]).reshape(1, -1)
    all_feat_names = STUDENT_FEATURE_COLS + OE_FEATURE_COLS

    # Predicted score
    student_tensor = torch.tensor(student_vec_np, dtype=torch.float32) \
                         .unsqueeze(0).to(loader.device)
    oe_tensor      = torch.tensor(oe_vec_np, dtype=torch.float32) \
                         .unsqueeze(0).to(loader.device)

    with torch.no_grad():
        predicted_score = loader.model(student_tensor, oe_tensor).item()

    # SHAP explanation
    # Build background dataset from a sample of students in the lookup
    sample_ids  = list(loader.student_lookup.keys())[:30]
    oe_ids      = list(loader.oe_lookup.keys())[:30]

    background = np.array([
        np.concatenate([loader.student_lookup[s], loader.oe_lookup[o]])
        for s, o in zip(sample_ids, oe_ids)
    ])

    # Model wrapper for SHAP
    def model_predict(x: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(x, dtype=torch.float32).to(loader.device)
        s_dim    = len(STUDENT_FEATURE_COLS)
        s_vec    = x_tensor[:, :s_dim]
        o_vec    = x_tensor[:, s_dim:]
        with torch.no_grad():
            scores = loader.model(s_vec, o_vec)
        return scores.cpu().numpy()

    explainer   = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(combined, nsamples=100)

    # shap_values shape can be (1, n_features) or (1, n_features, 1)
    if isinstance(shap_values, list):
        shap_vals = np.array(shap_values[0]).flatten()
    else:
        shap_vals = np.array(shap_values).flatten()

    # Build feature contribution dict
    contributions = {
        name: round(float(val), 4)
        for name, val in zip(all_feat_names, shap_vals)
    }

    # Sort by absolute contribution
    sorted_contribs = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Human readable labels
    readable_map = {
        "cgpa"                          : "Your CGPA",
        "avg_core_grade"                : "Your overall core course average",
        "sem1_avg"                      : "Your semester 1 average",
        "sem2_avg"                      : "Your semester 2 average",
        "sem3_avg"                      : "Your semester 3 average",
        "sem4_avg"                      : "Your semester 4 average",
        "branch_CSE"                    : "OE is offered by CSE branch",
        "branch_ECE"                    : "OE is offered by ECE branch",
        "branch_ME"                     : "OE is offered by ME branch",
        "branch_CE"                     : "OE is offered by CE branch",
        "branch_EEE"                    : "OE is offered by EEE branch",
        "oe_avg_teaching_clarity"       : "Professor teaching clarity",
        "oe_avg_course_organization"    : "Professor course organization",
        "oe_avg_overall_rating"         : "Professor overall rating",
        "oe_avg_interaction"            : "Professor student interaction",
        "oe_avg_assignment_usefulness"  : "Assignment usefulness",
        "sentiment_score"               : "Student sentiment about professor",
        "is_new_oe"                     : "OE is newly introduced",
        "is_new_prof"                   : "Professor is new",
        "available_semester"            : "Semester availability",
    }

    def make_readable(name: str, val: float) -> str:
        label     = readable_map.get(name, name)
        direction = "positively influences" if val >= 0 else "negatively influences"
        strength  = abs(val)
        if strength > 0.05:
            impact = "strongly"
        elif strength > 0.02:
            impact = "moderately"
        else:
            impact = "slightly"
        return f"{label} {impact} {direction} this recommendation ({val:+.4f})"

    positive_contribs = [(n, v) for n, v in sorted_contribs if v > 0.001]
    top_reasons       = [make_readable(n, v) for n, v in positive_contribs[:3]]
    negative_contribs = sorted(
        [(n, v) for n, v in contributions.items() if v < -0.001],
        key=lambda x: x[1]
    )
    bottom_reasons = [make_readable(n, v) for n, v in negative_contribs[:3]]
    if not top_reasons:
        top_reasons = ["No strong positive signals found for this recommendation"]
    if not bottom_reasons:
        bottom_reasons = ["No features are negatively influencing this recommendation"]

    return {
        "student_id"          : student_id,
        "oe_id"               : oe_id,
        "semester"            : semester,
        "predicted_score"     : round(predicted_score, 4),
        "feature_contributions": contributions,
        "top_reasons"         : top_reasons,
        "bottom_reasons"      : bottom_reasons,
        "generated_at"        : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
