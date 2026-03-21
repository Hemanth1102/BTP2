"""
model/train.py
===============
Training loop for NeuMF.

  Loss      : Binary Cross Entropy weighted by grade score
  Optimizer : Adam (lr=0.001)
  Schedule  : ReduceLROnPlateau — halves LR if val loss plateaus
  Early stop: patience=5 epochs on val loss
  Checkpoint: saved to checkpoints/model_sem{N}.pt on val loss improvement
  Rollback  : new checkpoint only replaces live model if NDCG@10 improves

Usage:
  python -m model.train
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data      import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.dataset import OEDataset, STUDENT_FEATURE_COLS, OE_FEATURE_COLS
from model.neumf   import NeuMF

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"

BATCH_SIZE     = 64
NUM_EPOCHS     = 50
LEARNING_RATE  = 0.001
EARLY_STOP_PAT = 5        # stop if val loss doesn't improve for N epochs
EMBEDDING_DIM  = 64
MLP_LAYERS     = [256, 128, 64]
DROPOUT_RATE   = 0.3

# Semester being trained on (change to 6 or 7 for later semesters)
TRAIN_SEMESTER = 5


# ─────────────────────────────────────────────────────────────
# WEIGHTED BCE LOSS
# ─────────────────────────────────────────────────────────────

def weighted_bce_loss(predictions : torch.Tensor,
                      labels      : torch.Tensor,
                      weights     : torch.Tensor) -> torch.Tensor:
    """
    BCE loss weighted by grade score.
    An O-grade positive (weight=1.0) contributes more than
    a C-grade positive (weight=0.5).

    Negatives always have weight=0.0 so they are treated equally
    regardless of the score placeholder.
    """
    bce     = nn.BCELoss(reduction="none")
    loss    = bce(predictions.squeeze(), labels)

    # For negatives, use weight=1.0 (equal treatment)
    # For positives, use the grade score as weight
    final_weights = torch.where(labels == 0,
                                torch.ones_like(weights),
                                weights)
    return (loss * final_weights).mean()


# ─────────────────────────────────────────────────────────────
# NDCG@K — used for rollback decision
# ─────────────────────────────────────────────────────────────

def ndcg_at_k(predicted_scores: list,
              actual_oe_idx   : int,
              k               : int = 10) -> float:
    """
    Compute NDCG@K for a single student.
    predicted_scores: list of (oe_id, score) sorted by score desc
    actual_oe_idx   : index of the student's actual OE in the ranked list
    """
    if actual_oe_idx >= k:
        return 0.0
    return 1.0 / np.log2(actual_oe_idx + 2)   # +2 because log2(1)=0


# ─────────────────────────────────────────────────────────────
# TRAINING EPOCH
# ─────────────────────────────────────────────────────────────

def train_epoch(model     : NeuMF,
                loader    : DataLoader,
                optimizer : torch.optim.Optimizer,
                device    : torch.device) -> float:

    model.train()
    total_loss = 0.0

    for student_vec, oe_vec, label, score in loader:
        student_vec = student_vec.to(device)
        oe_vec      = oe_vec.to(device)
        label       = label.to(device)
        score       = score.to(device)

        optimizer.zero_grad()
        prediction  = model(student_vec, oe_vec)
        loss        = weighted_bce_loss(prediction, label, score)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────
# VALIDATION EPOCH
# ─────────────────────────────────────────────────────────────

def val_epoch(model : NeuMF,
              loader: DataLoader,
              device: torch.device) -> float:

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for student_vec, oe_vec, label, score in loader:
            student_vec = student_vec.to(device)
            oe_vec      = oe_vec.to(device)
            label       = label.to(device)
            score       = score.to(device)

            prediction = model(student_vec, oe_vec)
            loss       = weighted_bce_loss(prediction, label, score)
            total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────
# SAVE / LOAD CHECKPOINT
# ─────────────────────────────────────────────────────────────

def save_checkpoint(model    : NeuMF,
                    optimizer: torch.optim.Optimizer,
                    epoch    : int,
                    val_loss : float,
                    ndcg     : float,
                    semester : int):

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = f"{CHECKPOINT_DIR}/model_sem{semester}.pt"

    torch.save({
        "epoch"      : epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "val_loss"   : val_loss,
        "ndcg@10"    : ndcg,
        "semester"   : semester,
    }, path)

    return path


def load_checkpoint(path  : str,
                    model : NeuMF,
                    optimizer: torch.optim.Optimizer = None):

    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_state"])

    return checkpoint


# ─────────────────────────────────────────────────────────────
# QUICK NDCG EVAL ON VAL SET — for rollback decision
# ─────────────────────────────────────────────────────────────

def quick_ndcg_eval(model     : NeuMF,
                    val_ds    : OEDataset,
                    device    : torch.device,
                    k         : int = 10) -> float:
    """
    For each student in val set, rank their 32 eligible OEs
    and compute NDCG@K based on their actual choice.
    Returns mean NDCG@K across all students.
    """
    model.eval()
    ndcg_scores = []

    # Group val interactions by student
    val_df = pd.read_csv("data/processed/interaction_matrix.csv")
    val_df = val_df[val_df["split"] == "val"]

    students = val_df["student_id"].unique()

    with torch.no_grad():
        for student_id in students:
            student_rows = val_df[val_df["student_id"] == student_id]
            actual_oe    = student_rows[student_rows["is_negative"] == 0]["oe_id"].values

            if len(actual_oe) == 0:
                continue

            actual_oe = actual_oe[0]

            # Get all OEs for this student (positives + negatives)
            all_oe_ids = student_rows["oe_id"].values

            # Score each OE
            student_vec = torch.tensor(
                val_ds._student_lookup[student_id], dtype=torch.float32
            ).unsqueeze(0).to(device)

            scores = []
            for oe_id in all_oe_ids:
                oe_vec = torch.tensor(
                    val_ds._oe_lookup[oe_id], dtype=torch.float32
                ).unsqueeze(0).to(device)
                score  = model(student_vec, oe_vec).item()
                scores.append((oe_id, score))

            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)
            ranked_oes = [oe for oe, _ in scores]

            if actual_oe in ranked_oes:
                rank  = ranked_oes.index(actual_oe)
                ndcg  = ndcg_at_k(scores, rank, k)
                ndcg_scores.append(ndcg)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


# ─────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train(semester: int = TRAIN_SEMESTER):

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Datasets
    train_ds = OEDataset(split="train")
    val_ds   = OEDataset(split="val")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = NeuMF(
        student_dim   = train_ds.student_dim,
        oe_dim        = train_ds.oe_dim,
        embedding_dim = EMBEDDING_DIM,
        mlp_layers    = MLP_LAYERS,
        dropout       = DROPOUT_RATE,
    ).to(device)

    print(f"Model parameters : {model.count_parameters():,}")

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=3, verbose=True)

    # Training state
    best_val_loss  = float("inf")
    best_ndcg      = 0.0
    patience_count = 0
    history        = []

    print(f"\nTraining NeuMF — semester {semester}")
    print(f"{'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} "
          f"{'NDCG@10':<12} {'LR':<10} {'Status'}")
    print("─" * 72)

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss = train_epoch(model, train_dl, optimizer, device)
        val_loss   = val_epoch(model, val_dl, device)

        # Compute NDCG every 5 epochs (expensive, don't do every epoch)
        if epoch % 5 == 0 or epoch == 1:
            ndcg = quick_ndcg_eval(model, val_ds, device, k=10)
        else:
            ndcg = history[-1]["ndcg"] if history else 0.0

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        status = ""

        # Save checkpoint if val loss improved AND NDCG is better or equal
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0

            if ndcg >= best_ndcg:
                best_ndcg = ndcg
                path      = save_checkpoint(model, optimizer,
                                            epoch, val_loss, ndcg, semester)
                status = "saved"
            else:
                status = "val improved, ndcg did not"
        else:
            patience_count += 1
            status = f"patience {patience_count}/{EARLY_STOP_PAT}"

        history.append({
            "epoch"     : epoch,
            "train_loss": round(train_loss, 6),
            "val_loss"  : round(val_loss,   6),
            "ndcg"      : round(ndcg,       6),
            "lr"        : current_lr,
        })

        print(f"{epoch:<8} {train_loss:<14.6f} {val_loss:<14.6f} "
              f"{ndcg:<12.4f} {current_lr:<10.6f} {status}")

        # Early stopping
        if patience_count >= EARLY_STOP_PAT:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PAT} epochs)")
            break

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(f"{RESULTS_DIR}/training_history_sem{semester}.csv",
                      index=False)

    print(f"\n✓ Training complete")
    print(f"  Best val loss : {best_val_loss:.6f}")
    print(f"  Best NDCG@10  : {best_ndcg:.4f}")
    print(f"  History saved : {RESULTS_DIR}/training_history_sem{semester}.csv")

    # Load best checkpoint for return
    best_path = f"{CHECKPOINT_DIR}/model_sem{semester}.pt"
    if os.path.exists(best_path):
        load_checkpoint(best_path, model)
        print(f"  Best model loaded from {best_path}")

    # Re-evaluate on full 32-OE ranking and update checkpoint NDCG
    # This ensures rollback comparisons use honest metrics not inflated ones
    print(f"Computing honest NDCG@10 on full 32-OE ranking...")
    from model.evaluate import evaluate
    eval_semester = min(semester + 2, 7)   # cap at 7, highest test semester
    results_df    = evaluate(semester=eval_semester)
    honest_ndcg  = round(float(results_df["ndcg@10"].mean()), 4)
    print(f"  Honest NDCG@10 (full 32-OE) : {honest_ndcg}")

    # Update the stored NDCG in checkpoint
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, weights_only=True)
        ckpt["ndcg@10"] = honest_ndcg
        torch.save(ckpt, best_path)
        print(f"  Checkpoint NDCG updated to {honest_ndcg}")

    return model, history_df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, history = train(semester=TRAIN_SEMESTER)
