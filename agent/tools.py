"""
agent/tools.py
===============
Four tools available to the agent loop.

Each tool:
  - does exactly one job
  - returns a result dict with status + key metrics
  - logs its action to results/agent_log.csv

Tools:
  retrain_model()      → retrains NeuMF on new semester data
  refresh_sentiment()  → reruns sentiment pipeline on new comments
  shap_eval()          → SHAP feature importance analysis
  cold_start_handler() → routes new student/OE to content-based fallback

Usage:
  from agent.tools import retrain_model, refresh_sentiment,
                          shap_eval, cold_start_handler
"""

import os
import torch
import shap
import numpy as np
import pandas as pd
from datetime  import datetime
from torch.utils.data import DataLoader

from model.dataset  import OEDataset
from model.neumf    import NeuMF
from model.train    import (train, load_checkpoint, save_checkpoint,
                             EMBEDDING_DIM, MLP_LAYERS, DROPOUT_RATE)
from model.evaluate import evaluate
from sentiment.preprocess          import preprocess
from sentiment.sentiment_model     import run_sentiment
from sentiment.aggregate_sentiment import aggregate
from features.cold_start           import ColdStartHandler

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"
LOG_PATH       = f"{RESULTS_DIR}/agent_log.csv"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# LOGGING HELPER
# ─────────────────────────────────────────────────────────────

def log_action(tool: str, semester: int, result: dict):
    """Append one row to agent_log.csv."""
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tool"     : tool,
        "semester" : semester,
        **result,
    }
    log_df = pd.DataFrame([row])

    if os.path.exists(LOG_PATH):
        log_df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(LOG_PATH, mode="w", header=True, index=False)

    print(f"  [log] {tool} @ sem {semester} → {result}")


# ─────────────────────────────────────────────────────────────
# TOOL 1 — RETRAIN MODEL
# ─────────────────────────────────────────────────────────────

def retrain_model(semester: int) -> dict:
    """
    Retrains NeuMF on data up to and including current semester.
    Compares new NDCG@10 against existing checkpoint.
    Only replaces checkpoint if new model is better.

    Returns:
      status       : "improved" | "not_improved" | "no_previous"
      old_ndcg     : NDCG@10 of previous checkpoint
      new_ndcg     : NDCG@10 of newly trained model
      replaced     : True if checkpoint was replaced
    """
    print(f"\n[Tool] retrain_model — semester {semester}")

    # Get old NDCG from existing checkpoint if it exists
    old_checkpoint_path = f"{CHECKPOINT_DIR}/model_sem{semester - 1}.pt"
    old_ndcg = 0.0

    if os.path.exists(old_checkpoint_path):
        old_ckpt = torch.load(old_checkpoint_path, weights_only=True)
        old_ndcg = old_ckpt.get("ndcg@10", 0.0)
        print(f"  Previous checkpoint NDCG@10 : {old_ndcg:.4f}")
    else:
        print(f"  No previous checkpoint found")

    # Retrain
    model, history = train(semester=semester)

    # Evaluate new model
    results_df = evaluate(semester=semester)
    new_ndcg   = results_df["ndcg@10"].mean()

    # Rollback decision
    if new_ndcg >= old_ndcg:
        status   = "improved" if old_ndcg > 0.0 else "no_previous"
        replaced = True
        print(f"  New model NDCG@10 {new_ndcg:.4f} >= {old_ndcg:.4f} → checkpoint kept")
    else:
        status   = "not_improved"
        replaced = False
        print(f"  New model NDCG@10 {new_ndcg:.4f} < {old_ndcg:.4f} → rollback, keeping old")

        # Restore old checkpoint
        if os.path.exists(old_checkpoint_path):
            import shutil
            shutil.copy(old_checkpoint_path,
                        f"{CHECKPOINT_DIR}/model_sem{semester}.pt")

    result = {
        "status"  : status,
        "old_ndcg": round(old_ndcg,  4),
        "new_ndcg": round(new_ndcg,  4),
        "replaced": replaced,
    }

    log_action("retrain_model", semester, result)
    return result


# ─────────────────────────────────────────────────────────────
# TOOL 2 — REFRESH SENTIMENT
# ─────────────────────────────────────────────────────────────

def refresh_sentiment(semester: int) -> dict:
    """
    Reruns the full sentiment pipeline on current comments.
    Updates prof_features.csv with new sentiment scores.

    Useful when:
      - New semester comments are added
      - A new professor joins with new feedback

    Returns:
      status           : "success" | "failed"
      profs_updated    : number of professors with updated scores
      new_profs_found  : professors who weren't in previous features
    """
    print(f"\n[Tool] refresh_sentiment — semester {semester}")

    try:
        # Load previous prof_features for comparison
        old_path = "data/processed/prof_features.csv"
        old_profs = set()
        if os.path.exists(old_path):
            old_df    = pd.read_csv(old_path)
            old_profs = set(old_df["prof_id"].tolist())

        # Rerun full sentiment pipeline
        preprocess()
        run_sentiment()
        new_prof_features = aggregate()

        # Compare
        new_profs       = set(new_prof_features["prof_id"].tolist())
        new_profs_found = len(new_profs - old_profs)
        profs_updated   = len(new_profs)

        result = {
            "status"         : "success",
            "profs_updated"  : profs_updated,
            "new_profs_found": new_profs_found,
        }

    except Exception as e:
        result = {
            "status"         : "failed",
            "error"          : str(e),
            "profs_updated"  : 0,
            "new_profs_found": 0,
        }
        print(f"  Error: {e}")

    log_action("refresh_sentiment", semester, result)
    return result


# ─────────────────────────────────────────────────────────────
# TOOL 3 — SHAP EVALUATION
# ─────────────────────────────────────────────────────────────

def shap_eval(semester: int) -> dict:
    """
    Runs SHAP on the trained model to find which features
    contribute most to predictions.

    Helps the agent decide:
      - Are professor ratings actually helping?
      - Is sentiment score adding value?
      - Should any features be dropped or added?

    Returns:
      status            : "success" | "failed"
      top_features      : top 5 most important features
      bottom_features   : bottom 5 least important features
      sentiment_rank    : rank of sentiment_score among all features
    """
    print(f"\n[Tool] shap_eval — semester {semester}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        checkpoint_path = f"{CHECKPOINT_DIR}/model_sem{semester}.pt"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = f"{CHECKPOINT_DIR}/model_sem5.pt"

        test_ds = OEDataset(split="test")
        model   = NeuMF(
            student_dim   = test_ds.student_dim,
            oe_dim        = test_ds.oe_dim,
            embedding_dim = EMBEDDING_DIM,
            mlp_layers    = MLP_LAYERS,
            dropout       = DROPOUT_RATE,
        ).to(device)

        load_checkpoint(checkpoint_path, model)
        model.eval()

        # Build combined feature matrix for SHAP
        # Sample 100 students for speed
        test_dl   = DataLoader(test_ds, batch_size=100, shuffle=True)
        s_batch, oe_batch, _, _ = next(iter(test_dl))

        # Combined input for SHAP: [student_vec || oe_vec]
        combined  = torch.cat([s_batch, oe_batch], dim=1).numpy()

        from model.dataset import STUDENT_FEATURE_COLS, OE_FEATURE_COLS
        all_feature_names = STUDENT_FEATURE_COLS + OE_FEATURE_COLS

        # Wrapper so SHAP can call the model with combined input
        def model_predict(x):
            x_tensor    = torch.tensor(x, dtype=torch.float32).to(device)
            s_dim       = len(STUDENT_FEATURE_COLS)
            s_vec       = x_tensor[:, :s_dim]
            oe_vec      = x_tensor[:, s_dim:]
            with torch.no_grad():
                scores  = model(s_vec, oe_vec)
            return scores.cpu().numpy()

        # KernelExplainer — model-agnostic, works with any PyTorch model
        background  = combined[:50]    # background dataset
        explainer   = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(combined[:50], nsamples=100)

        # Mean absolute SHAP value per feature
        # Handle all possible SHAP output shapes
        shap_arr = np.array(shap_values)

        # Shape can be: (n_samples, n_features) or (n_samples, n_features, 1)
        # or a list of arrays — flatten to (n_features,)
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[:, :, 0]       # drop output dim
        if shap_arr.ndim == 1:
            shap_arr = shap_arr.reshape(1, -1)  # ensure 2D

        mean_shap = np.abs(shap_arr).mean(axis=0).flatten()

        # Align feature names length with shap values length
        n_shap    = len(mean_shap)
        n_feat    = len(all_feature_names)
        feat_names = all_feature_names[:n_shap] if n_shap <= n_feat else all_feature_names
        if n_shap > n_feat:
            mean_shap = mean_shap[:n_feat]

        # Feature importance DataFrame
        importance_df = pd.DataFrame({
            "feature"   : feat_names,
            "importance": mean_shap,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        importance_df.to_csv(
            f"{RESULTS_DIR}/shap_importance_sem{semester}.csv", index=False
        )

        top_features    = importance_df["feature"].head(5).tolist()
        bottom_features = importance_df["feature"].tail(5).tolist()

        # Find sentiment rank
        sentiment_rank = importance_df[
            importance_df["feature"] == "sentiment_score"
        ].index[0] + 1  # 1-indexed

        print(f"  Top features    : {top_features}")
        print(f"  Sentiment rank  : {sentiment_rank} / {len(all_feature_names)}")

        result = {
            "status"          : "success",
            "top_features"    : str(top_features),
            "bottom_features" : str(bottom_features),
            "sentiment_rank"  : int(sentiment_rank),
            "total_features"  : len(all_feature_names),
        }

    except Exception as e:
        result = {
            "status"          : "failed",
            "error"           : str(e),
            "top_features"    : "",
            "bottom_features" : "",
            "sentiment_rank"  : -1,
            "total_features"  : -1,
        }
        print(f"  Error: {e}")

    log_action("shap_eval", semester, result)
    return result


# ─────────────────────────────────────────────────────────────
# TOOL 4 — COLD START HANDLER
# ─────────────────────────────────────────────────────────────

def cold_start_handler(student_id: str     = None,
                        oe_id     : str     = None,
                        semester  : int     = 5) -> dict:
    """
    Routes cold start cases to content-based fallback.

    If student_id provided → find similar students → recommend their OEs
    If oe_id provided      → find similar OEs      → warm-start embedding

    Returns:
      status       : "success" | "failed"
      case         : "new_student" | "new_oe"
      recommendations: list of recommended OE ids (new student case)
      similar_oes  : list of similar OE ids (new OE case)
    """
    print(f"\n[Tool] cold_start_handler — semester {semester}")

    try:
        handler = ColdStartHandler()
        handler.load()

        if student_id:
            print(f"  Cold start student : {student_id}")
            is_cold = handler.is_cold_start_student(student_id)

            if not is_cold:
                result = {
                    "status": "skipped",
                    "case"  : "new_student",
                    "reason": f"{student_id} already has OE history",
                }
            else:
                recs = handler.recommend_for_new_student(student_id, semester)
                result = {
                    "status"         : "success",
                    "case"           : "new_student",
                    "recommendations": recs["oe_id"].tolist() if not recs.empty else [],
                    "n_recommendations": len(recs),
                }

        elif oe_id:
            print(f"  Cold start OE      : {oe_id}")
            is_cold = handler.is_cold_start_oe(oe_id)

            if not is_cold:
                result = {
                    "status": "skipped",
                    "case"  : "new_oe",
                    "reason": f"{oe_id} already has interaction history",
                }
            else:
                similar = handler.find_similar_oes(oe_id)
                result  = {
                    "status"      : "success",
                    "case"        : "new_oe",
                    "similar_oes" : similar["oe_id"].tolist(),
                    "n_similar"   : len(similar),
                }
        else:
            result = {
                "status": "failed",
                "reason": "Must provide either student_id or oe_id",
            }

    except Exception as e:
        result = {"status": "failed", "error": str(e)}
        print(f"  Error: {e}")

    log_action("cold_start_handler", semester, result)
    return result
