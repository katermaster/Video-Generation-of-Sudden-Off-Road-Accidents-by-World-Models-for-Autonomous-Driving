"""
Sudden Accident Condition Controller (Section 3.3)

Implements controllable triggering and physical evolution of four typical
off-road sudden accidents:

  1. Animal Intrusion  - Dynamic obstacle suddenly enters scene
  2. Rockfall           - Rocks falling from slopes onto vehicle path
  3. Terrain Collapse   - Ground collapses beneath/ahead of vehicle
  4. Sudden Skidding    - Friction mutation causes loss of traction

Key equations:
  (5)  z_t^cond = z_t + W_c · C + b_c,    ∀ t ≥ t_trigger
  (6)  μ_t = μ_0 - μ_rate · I(p_ego ∈ S_μ)   [skidding physics]

Accident control vector C ∈ R^6:
  [accident_type, trigger_frame, spatial_range, friction_atten, collapse_slope, obstacle_size]
"""

import torch
import torch.nn as nn


class AccidentControlEncoder(nn.Module):
    """Encodes the accident control vector C into latent-compatible representation."""

    def __init__(self, control_dim: int = 6, latent_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(control_dim, latent_dim, bias=True)
        # Initialize with small weights for stable training
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, control_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            control_vector: [B, 6] — (type, trigger_frame, spatial_range,
                                     friction_atten, collapse_slope, obstacle_size)
        Returns:
            [B, D] projected control embedding
        """
        return self.proj(control_vector)


class FrictionMutationModel(nn.Module):
    """
    Physical model for sudden skidding accident.
    Computes the real-time dynamic friction coefficient (Eq. 6):

      μ_t = μ_0 - μ_rate · I(p_ego ∈ S_μ)

    where:
      μ_0 ∈ [0.2, 0.8]   — original terrain friction coefficient
      μ_rate ∈ [0.1, 0.5] — friction attenuation ratio
      S_μ                  — low-friction accident region
      I(·)                 — indicator function
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        mu0: torch.Tensor,        # [B] original friction
        mu_rate: torch.Tensor,    # [B] attenuation ratio
        ego_position: torch.Tensor,  # [B, 2] (x, y) in scene coordinates
        accident_region: torch.Tensor,  # [B, 4] (cx, cy, w, h) low-friction zone
    ) -> torch.Tensor:
        """
        Returns:
            mu_t: [B] real-time friction coefficient after mutation
        """
        # Check if ego vehicle is inside the accident region S_μ
        cx, cy, w, h = accident_region.unbind(dim=-1)
        px, py = ego_position.unbind(dim=-1)

        in_region = (
            (px >= cx - w / 2) &
            (px <= cx + w / 2) &
            (py >= cy - h / 2) &
            (py <= cy + h / 2)
        ).float()  # [B], 1.0 if inside, 0.0 otherwise

        # Apply friction mutation: μ_t = μ_0 - μ_rate * I(p_ego ∈ S_μ)
        mu_t = mu0 - mu_rate * in_region
        # Clamp to physically plausible range
        mu_t = mu_t.clamp(min=0.05, max=1.0)

        return mu_t


class TerrainCollapseModel(nn.Module):
    """
    Physical model for terrain collapse accident.
    Models sudden changes in terrain elevation and slope.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        elevation: torch.Tensor,      # [B, T, H, W] terrain elevation map
        collapse_slope: torch.Tensor, # [B] slope angle of collapse (radians)
        collapse_mask: torch.Tensor,  # [B, H, W] region to collapse
        trigger_idx: int,             # When collapse starts
    ) -> torch.Tensor:
        """
        Returns modified elevation map with collapse applied.
        For pre-trained mode, applies a smooth deformation.
        """
        B, T, H, W = elevation.shape
        collapse_slope = collapse_slope.view(B, 1, 1, 1)

        # Smooth depth decrease in collapse region
        collapse_depth = torch.tan(collapse_slope) * 2.0  # approximate drop
        collapse_mask_expanded = collapse_mask.unsqueeze(1)  # [B, 1, H, W]

        # Apply collapse progressively after trigger
        for t in range(trigger_idx, min(trigger_idx + 5, T)):
            progress = (t - trigger_idx + 1) / 5.0
            elevation[:, t] = elevation[:, t] - collapse_depth.squeeze(-1) * collapse_mask * progress

        return elevation


class AccidentController(nn.Module):
    """
    Conditional accident controller that:
      1. Encodes the accident control vector C
      2. Injects it into latent states at the trigger time (Eq. 5)
      3. Applies physics-based evolution rules per accident type
    """

    def __init__(
        self,
        control_vector_dim: int = 6,
        latent_dim: int = 512,
        num_accident_types: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_accident_types = num_accident_types

        # Control vector encoder + projection (Eq. 5: W_c, b_c)
        self.control_encoder = AccidentControlEncoder(control_vector_dim, latent_dim)

        # Per-accident-type learnable embeddings for finer control
        self.type_embeddings = nn.Embedding(num_accident_types, latent_dim)

        # Physical evolution models
        self.friction_model = FrictionMutationModel()
        self.collapse_model = TerrainCollapseModel()

        # Post-injection refinement
        self.refine = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def inject_accident(
        self,
        z: torch.Tensor,              # [B, T, D] latent sequence
        control_vector: torch.Tensor, # [B, 6] accident control
    ) -> torch.Tensor:
        """
        Injects accident condition into latent states at trigger time (Eq. 5).

        Args:
            z: original latent sequence
            control_vector: (type, trigger_frame, spatial_range,
                            friction_atten, collapse_slope, obstacle_size)
        Returns:
            z_cond: [B, T, D] conditional latent sequence
        """
        B, T, D = z.shape
        device = z.device

        # Parse control vector
        accident_type = control_vector[:, 0].long()           # [B]
        trigger_frame = control_vector[:, 1].clamp(0, T - 1).long()  # [B]

        # Encode control vector (W_c · C + b_c)
        control_embed = self.control_encoder(control_vector)  # [B, D]

        # Get accident type embedding
        type_embed = self.type_embeddings(accident_type)      # [B, D]

        # Combine control + type embeddings
        combined = self.refine(
            torch.cat([control_embed, type_embed], dim=-1)
        )  # [B, D]

        # Create injection mask: 1 after trigger, 0 before
        trigger_mask = (
            torch.arange(T, device=device).unsqueeze(0) >= trigger_frame.unsqueeze(1)
        ).float()  # [B, T]

        # Inject condition (Eq. 5): z_t^cond = z_t + (W_c·C + b_c) * mask
        z_cond = z + combined.unsqueeze(1) * trigger_mask.unsqueeze(-1)

        return z_cond

    def compute_physical_params(
        self,
        control_vector: torch.Tensor,
        z_cond: torch.Tensor,
    ) -> dict:
        """
        Compute physics-based parameters for accident evolution.

        Returns a dict with physical parameters for each accident type.
        """
        B = control_vector.size(0)
        accident_type = control_vector[:, 0].long()

        # Extract relevant parameters from control vector
        mu0 = control_vector[:, 2].clamp(0.2, 0.8)       # friction coefficient
        mu_rate = control_vector[:, 3].clamp(0.1, 0.5)    # attenuation ratio
        collapse_slope = control_vector[:, 4].clamp(0.1, 1.0)  # slope (radians)
        obstacle_size = control_vector[:, 5].clamp(0.1, 5.0)   # obstacle size (m)

        # Default ego position (center of scene) for skidding model
        ego_pos = torch.zeros(B, 2, device=control_vector.device)
        # Default accident region (center of scene with variable size)
        accident_region = torch.stack([
            torch.zeros(B, device=control_vector.device),  # cx
            torch.zeros(B, device=control_vector.device),  # cy
            obstacle_size,                                  # w
            obstacle_size,                                  # h
        ], dim=-1)

        # Compute friction mutation
        mu_t = self.friction_model(mu0, mu_rate, ego_pos, accident_region)

        return {
            "accident_type": accident_type,
            "friction_mu0": mu0,
            "friction_mu_t": mu_t,
            "friction_atten_ratio": mu_rate,
            "collapse_slope": collapse_slope,
            "obstacle_size": obstacle_size,
        }

    def forward(
        self,
        z: torch.Tensor,
        control_vector: torch.Tensor,
    ) -> dict:
        """
        Full accident control pipeline.

        Args:
            z: [B, T, D] original latent states
            control_vector: [B, 6] accident control parameters
        Returns:
            dict with:
              - 'z_cond': [B, T, D] conditional latent states
              - 'physical_params': dict of computed physical parameters
        """
        # Inject accident condition
        z_cond = self.inject_accident(z, control_vector)

        # Compute physical evolution parameters
        physical_params = self.compute_physical_params(control_vector, z_cond)

        return {
            "z_cond": z_cond,
            "physical_params": physical_params,
        }


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("Testing AccidentController...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    controller = AccidentController(
        control_vector_dim=6,
        latent_dim=512,
        num_accident_types=4,
    ).to(device)

    B, T, D = 2, 16, 512
    dummy_z = torch.randn(B, T, D).to(device)

    # Control vector: [type=3 (skidding), trigger=8, friction=0.4, attenuation=0.3,
    #                   collapse_slope=0.5, obstacle_size=2.0]
    dummy_control = torch.tensor([
        [3.0, 8.0, 0.4, 0.3, 0.5, 2.0],
        [1.0, 6.0, 0.6, 0.2, 0.3, 1.5],
    ]).to(device)

    with torch.no_grad():
        out = controller(dummy_z, dummy_control)

    print(f"Input latent:     {dummy_z.shape}")
    print(f"Conditional latent: {out['z_cond'].shape}")
    print(f"Physical params:")
    for k, v in out["physical_params"].items():
        print(f"  {k}: {v}")
    print(f"#Params: {sum(p.numel() for p in controller.parameters()):,}")
    print("Test passed!")
