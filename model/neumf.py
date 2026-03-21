"""
model/neumf.py
===============
NeuMF (Neural Matrix Factorization) — He et al., 2017
Combines GMF (linear) + MLP (non-linear) interaction modeling.

Architecture:
  Student input  → student embedding layer
  OE input       → oe embedding layer

  GMF path:
    student_emb ⊙ oe_emb  (element-wise product)
    → captures linear interactions

  MLP path:
    [student_emb || oe_emb]  (concatenation)
    → FC(256) → ReLU → Dropout
    → FC(128) → ReLU → Dropout
    → FC(64)  → ReLU → Dropout
    → captures non-linear interactions

  NeuMF:
    [GMF_output || MLP_output]
    → FC(1) → Sigmoid
    → relevance score in [0, 1]

Usage:
  from model.neumf import NeuMF

  model = NeuMF(student_dim=11, oe_dim=14)
  score = model(student_vec, oe_vec)   # shape: (batch, 1)
"""

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────
# CONFIG — locked MLP layer sizes
# ─────────────────────────────────────────────────────────────

EMBEDDING_DIM = 64      # size of both student and OE embeddings
MLP_LAYERS    = [256, 128, 64]   # hidden layer sizes for MLP path
DROPOUT_RATE  = 0.3


# ─────────────────────────────────────────────────────────────
# NeuMF
# ─────────────────────────────────────────────────────────────

class NeuMF(nn.Module):
    """
    Args:
        student_dim  : dimension of student feature vector
        oe_dim       : dimension of OE feature vector
        embedding_dim: size of learned embeddings (default 64)
        mlp_layers   : hidden layer sizes for MLP path
        dropout      : dropout rate applied after each MLP layer
    """

    def __init__(self,
                 student_dim  : int,
                 oe_dim       : int,
                 embedding_dim: int       = EMBEDDING_DIM,
                 mlp_layers   : list      = MLP_LAYERS,
                 dropout      : float     = DROPOUT_RATE):

        super(NeuMF, self).__init__()

        self.embedding_dim = embedding_dim

        # ── Embedding layers ──────────────────────────────────
        # Project raw feature vectors into dense embedding space
        self.student_embedding = nn.Sequential(
            nn.Linear(student_dim, embedding_dim),
            nn.ReLU(),
        )
        self.oe_embedding = nn.Sequential(
            nn.Linear(oe_dim, embedding_dim),
            nn.ReLU(),
        )

        # ── MLP path ──────────────────────────────────────────
        # Input = concatenation of both embeddings = 2 * embedding_dim
        mlp_input_dim = embedding_dim * 2
        mlp_modules   = []

        in_dim = mlp_input_dim
        for out_dim in mlp_layers:
            mlp_modules.append(nn.Linear(in_dim, out_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*mlp_modules)

        # ── Output layer ──────────────────────────────────────
        # GMF output dim  = embedding_dim  (element-wise product)
        # MLP output dim  = mlp_layers[-1] (last hidden layer)
        # Combined input  = embedding_dim + mlp_layers[-1]
        output_input_dim = embedding_dim + mlp_layers[-1]

        self.output_layer = nn.Sequential(
            nn.Linear(output_input_dim, 1),
            nn.Sigmoid(),
        )

        # Initialize weights
        self._init_weights()

    # ─────────────────────────────────────────────────────────
    # WEIGHT INITIALIZATION
    # ─────────────────────────────────────────────────────────

    def _init_weights(self):
        """
        Xavier uniform for linear layers — standard for deep networks.
        Prevents vanishing/exploding gradients at initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # ─────────────────────────────────────────────────────────
    # FORWARD PASS
    # ─────────────────────────────────────────────────────────

    def forward(self,
                student_vec: torch.Tensor,
                oe_vec     : torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_vec : (batch_size, student_dim)
            oe_vec      : (batch_size, oe_dim)

        Returns:
            score       : (batch_size, 1) — relevance score in [0, 1]
        """

        # Shared embeddings used by both paths
        student_emb = self.student_embedding(student_vec)   # (batch, embedding_dim)
        oe_emb      = self.oe_embedding(oe_vec)             # (batch, embedding_dim)

        # GMF path — element-wise product captures linear interactions
        gmf_output = student_emb * oe_emb                  # (batch, embedding_dim)

        # MLP path — concatenation + FC layers captures non-linear interactions
        mlp_input  = torch.cat([student_emb, oe_emb], dim=1)  # (batch, 2*embedding_dim)
        mlp_output = self.mlp(mlp_input)                       # (batch, mlp_layers[-1])

        # Combine GMF + MLP outputs
        combined = torch.cat([gmf_output, mlp_output], dim=1)  # (batch, emb_dim + mlp[-1])

        # Final relevance score
        score = self.output_layer(combined)                    # (batch, 1)

        return score

    # ─────────────────────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self, student_dim: int, oe_dim: int):
        print("NeuMF Architecture")
        print("==================")
        print(f"  Student input dim    : {student_dim}")
        print(f"  OE input dim         : {oe_dim}")
        print(f"  Embedding dim        : {self.embedding_dim}")
        print(f"  GMF output dim       : {self.embedding_dim}")
        print(f"  MLP layers           : {MLP_LAYERS}")
        print(f"  MLP output dim       : {MLP_LAYERS[-1]}")
        print(f"  Combined dim         : {self.embedding_dim + MLP_LAYERS[-1]}")
        print(f"  Output               : 1 (sigmoid)")
        print(f"  Dropout rate         : {DROPOUT_RATE}")
        print(f"  Total parameters     : {self.count_parameters():,}")
        print("==================")


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Dimensions from dataset.py
    STUDENT_DIM = 11    # 5 branch OHE + cgpa + avg_core + sem1-4
    OE_DIM      = 14    # 5 branch OHE + sem + 5 ratings + sentiment + 2 flags

    model = NeuMF(student_dim=STUDENT_DIM, oe_dim=OE_DIM)
    model.summary(STUDENT_DIM, OE_DIM)

    # Test forward pass with dummy batch
    batch_size  = 64
    student_vec = torch.randn(batch_size, STUDENT_DIM)
    oe_vec      = torch.randn(batch_size, OE_DIM)

    model.eval()
    with torch.no_grad():
        scores = model(student_vec, oe_vec)

    print(f"\nForward pass test:")
    print(f"  Input  student_vec : {student_vec.shape}")
    print(f"  Input  oe_vec      : {oe_vec.shape}")
    print(f"  Output scores      : {scores.shape}")
    print(f"  Score range        : [{scores.min().item():.4f}, "
          f"{scores.max().item():.4f}]  (expected within [0, 1])")
    print(f"  Score mean         : {scores.mean().item():.4f}")

    # Test GPU if available
    if torch.cuda.is_available():
        print(f"\nGPU test (RTX 3050):")
        model      = model.cuda()
        student_vec = student_vec.cuda()
        oe_vec      = oe_vec.cuda()
        with torch.no_grad():
            scores = model(student_vec, oe_vec)
        print(f"  GPU forward pass   : OK")
        print(f"  Output device      : {scores.device}")
    else:
        print("\nNo GPU detected — running on CPU")
