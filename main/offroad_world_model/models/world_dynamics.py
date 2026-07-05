"""
Terrain-Aware World Dynamics Prediction (Section 3.2)

Takes the unified latent sequence Z and terrain physical embedding E_g,
predicts future scene latent states Z_pred using a Transformer-based
dynamic network f_dyn.

Key equation (3):  Z_pred = f_dyn(Z̄, E_g; θ_dyn)

Also implements latent temporal consistency regularization (Eq. 4):
  L_reg = (1/(T'-1)) * Σ ||z_{T+k+1} - z_{T+k}||²₂
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TerrainPhysicalEmbedding(nn.Module):
    """
    Encodes terrain physical attributes into a fixed-dimensional embedding E_g.
    Attributes include: friction coefficient, slope angle, terrain type,
    surface loose degree, elevation, and roughness.
    """

    def __init__(self, phys_dim: int = 8, embed_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(phys_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim),
        )

    def forward(self, terrain_phys: torch.Tensor) -> torch.Tensor:
        """
        Args:
            terrain_phys: [B, 8] terrain physical parameters
        Returns:
            E_g: [B, 1, D] terrain embedding
        """
        e = self.net(terrain_phys)          # [B, D]
        return e.unsqueeze(1)               # [B, 1, D] for broadcasting


class TerrainAwareWorldDynamics(nn.Module):
    """
    Transformer-based dynamic network that predicts future latent states
    conditioned on terrain physical embedding.

    Architecture:
      - Terrain embedding E_g is prepended as a condition token
      - Input sequence: [E_g, z̄_1, z̄_2, ..., z̄_T]
      - Transformer encoder processes the sequence
      - Output head projects to future latent states Z_pred
    """

    def __init__(
        self,
        latent_dim: int = 512,
        terrain_phys_dim: int = 8,
        transformer_num_layers: int = 6,
        transformer_num_heads: int = 8,
        dropout: float = 0.1,
        prediction_horizon: int = 16,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.prediction_horizon = prediction_horizon

        # Terrain physical embedding
        self.terrain_embed = TerrainPhysicalEmbedding(terrain_phys_dim, latent_dim)

        # Learnable condition token (marks the terrain condition)
        self.cond_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=transformer_num_heads,
                dim_feedforward=latent_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=transformer_num_layers,
        )

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # Output projection to prediction_horizon frames
        self.temporal_proj = nn.Linear(latent_dim, latent_dim * prediction_horizon)

        self.norm_out = nn.LayerNorm(latent_dim)

    def forward(
        self,
        z: torch.Tensor,                  # [B, T, D] input latent sequence
        terrain_phys: torch.Tensor,       # [B, 8] terrain physical params
    ) -> dict:
        """
        Returns:
            dict with:
              - 'z_pred': [B, T', D] predicted future latent states
              - 'latent_reg': scalar, temporal smoothness regularization
        """
        B, T, D = z.shape

        # 1. Encode terrain physics
        e_g = self.terrain_embed(terrain_phys)       # [B, 1, D]

        # 2. Prepend terrain condition to latent sequence
        cond = self.cond_token.expand(B, -1, -1)     # [B, 1, D]
        input_seq = torch.cat([cond, e_g, z], dim=1) # [B, 2+T, D]

        # 3. Transformer forward
        encoded = self.transformer(input_seq)         # [B, 2+T, D]

        # 4. Extract future representation (from the latent portion)
        future_repr = encoded[:, 2:, :]               # [B, T, D]

        # 5. Predict future states
        z_pred_feat = self.pred_head(future_repr)     # [B, T, D]

        # 6. Temporal projection to prediction horizon
        #    Take the last frame's prediction and expand
        last_pred = z_pred_feat[:, -1:, :]            # [B, 1, D]
        z_pred_flat = self.temporal_proj(last_pred)   # [B, 1, D*T']
        z_pred = z_pred_flat.view(B, self.prediction_horizon, D)
        z_pred = self.norm_out(z_pred)

        # 7. Latent temporal consistency regularization (Eq. 4)
        diffs = (z_pred[:, 1:] - z_pred[:, :-1]).pow(2).sum(dim=-1).sqrt()
        latent_reg = diffs.mean()

        return {
            "z_pred": z_pred,
            "latent_reg": latent_reg,
        }


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("Testing TerrainAwareWorldDynamics...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TerrainAwareWorldDynamics(
        latent_dim=512,
        terrain_phys_dim=8,
        transformer_num_layers=6,
        transformer_num_heads=8,
        prediction_horizon=16,
    ).to(device)

    B, T, D = 2, 16, 512
    dummy_z = torch.randn(B, T, D).to(device)
    dummy_terrain = torch.randn(B, 8).to(device)

    with torch.no_grad():
        out = model(dummy_z, dummy_terrain)

    print(f"Input latent:    {dummy_z.shape}")
    print(f"Predicted future: {out['z_pred'].shape}")
    print(f"Latent reg loss:  {out['latent_reg'].item():.4f}")
    print(f"#Params: {sum(p.numel() for p in model.parameters()):,}")
    print("Test passed!")
