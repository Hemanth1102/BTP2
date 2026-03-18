"""
model/dataset.py
=================
PyTorch Dataset class for the NeuMF recommender.

Each sample:
  student_vector → float tensor of student features
  oe_vector      → float tensor of OE features
  label          → binary int   (0 or 1, BCE loss target)
  score          → float        (0.0–1.0, used as sample weight)

Split mapping (locked):
  train → sem 5 interactions
  val   → sem 6 interactions
  test  → sem 7 interactions

Usage:
  from model.dataset import OEDataset
  from torch.utils.data import DataLoader

  train_ds = OEDataset(split="train")
  train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

  for student_vec, oe_vec, label, score in train_dl:
      ...
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

PROCESSED_DIR = "data/processed"

# Student feature columns — must match feature_engineering.py output
STUDENT_FEATURE_COLS = (
    ["branch_CSE", "branch_ECE", "branch_ME", "branch_CE", "branch_EEE"] +
    ["cgpa", "avg_core_grade",
     "sem1_avg", "sem2_avg", "sem3_avg", "sem4_avg"]
)

# OE feature columns — must match feature_engineering.py output
OE_FEATURE_COLS = (
    ["branch_CSE", "branch_ECE", "branch_ME", "branch_CE", "branch_EEE"] +
    ["available_semester",
     "oe_avg_teaching_clarity", "oe_avg_course_organization",
     "oe_avg_overall_rating",   "oe_avg_interaction",
     "oe_avg_assignment_usefulness",
     "sentiment_score", "is_new_oe", "is_new_prof"]
)


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

class OEDataset(Dataset):
    """
    Loads interaction_matrix, student_features, oe_features
    and returns (student_vec, oe_vec, label, score) per sample.

    Args:
        split       : "train" | "val" | "test"
        processed_dir: path to data/processed/
    """

    def __init__(self,
                 split: str        = "train",
                 processed_dir: str = PROCESSED_DIR):

        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test'. Got '{split}'"

        self.split = split

        # Load tables
        interactions   = pd.read_csv(f"{processed_dir}/interaction_matrix.csv")
        student_feats  = pd.read_csv(f"{processed_dir}/student_features.csv")
        oe_feats       = pd.read_csv(f"{processed_dir}/oe_features.csv")

        # Filter to requested split
        interactions = interactions[interactions["split"] == split].reset_index(drop=True)

        # Build lookup dicts: id → feature vector
        self._student_lookup = self._build_lookup(student_feats,
                                                   "student_id",
                                                   STUDENT_FEATURE_COLS)
        self._oe_lookup      = self._build_lookup(oe_feats,
                                                   "oe_id",
                                                   OE_FEATURE_COLS)

        # Store interaction rows as aligned arrays for fast __getitem__
        self._student_ids = interactions["student_id"].values
        self._oe_ids      = interactions["oe_id"].values
        self._labels      = interactions["label"].astype(float).values
        self._scores      = interactions["score"].astype(float).values

        print(f"OEDataset [{split}] loaded:")
        print(f"  Samples      : {len(self._labels)}")
        print(f"  Positives    : {int(self._labels.sum())}")
        print(f"  Negatives    : {int((self._labels == 0).sum())}")
        print(f"  Student dims : {len(STUDENT_FEATURE_COLS)}")
        print(f"  OE dims      : {len(OE_FEATURE_COLS)}")

    # ─────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_lookup(df: pd.DataFrame,
                      id_col: str,
                      feature_cols: list) -> dict:
        """
        Build {id → np.array of features} for fast O(1) access
        during training.
        """
        lookup = {}
        for _, row in df.iterrows():
            lookup[row[id_col]] = row[feature_cols].values.astype(np.float32)
        return lookup

    # ─────────────────────────────────────────────────────────
    # DATASET INTERFACE
    # ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        student_id = self._student_ids[idx]
        oe_id      = self._oe_ids[idx]

        student_vec = torch.tensor(self._student_lookup[student_id], dtype=torch.float32)
        oe_vec      = torch.tensor(self._oe_lookup[oe_id],           dtype=torch.float32)
        label       = torch.tensor(self._labels[idx],                dtype=torch.float32)
        score       = torch.tensor(self._scores[idx],                dtype=torch.float32)

        return student_vec, oe_vec, label, score

    # ─────────────────────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────────────────────

    @property
    def student_dim(self) -> int:
        return len(STUDENT_FEATURE_COLS)

    @property
    def oe_dim(self) -> int:
        return len(OE_FEATURE_COLS)


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    for split in ["train", "val", "test"]:
        print(f"\n{'='*40}")
        ds = OEDataset(split=split)

        # Test single sample fetch
        student_vec, oe_vec, label, score = ds[0]
        print(f"  Sample[0] student_vec shape : {student_vec.shape}")
        print(f"  Sample[0] oe_vec shape      : {oe_vec.shape}")
        print(f"  Sample[0] label             : {label.item()}")
        print(f"  Sample[0] score             : {score.item():.2f}")

    # Test DataLoader
    print(f"\n{'='*40}")
    print("DataLoader test (train, batch_size=64):")
    train_ds = OEDataset(split="train")
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

    student_batch, oe_batch, label_batch, score_batch = next(iter(train_dl))
    print(f"  student_batch shape : {student_batch.shape}")
    print(f"  oe_batch shape      : {oe_batch.shape}")
    print(f"  label_batch shape   : {label_batch.shape}")
    print(f"  score_batch shape   : {score_batch.shape}")
    print(f"  label distribution  : "
          f"{label_batch.sum().int().item()} pos / "
          f"{(label_batch==0).sum().int().item()} neg in this batch")
