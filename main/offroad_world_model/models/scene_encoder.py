"""
Multi-Modal Off-Road Scene Encoder (Section 3.2)

Components:
  1. ResNet-18 backbone (pretrained) for per-frame visual feature extraction
  2. Temporal Transformer for spatiotemporal modeling
  3. MLP + Conv for vehicle state & terrain semantic feature embedding
  4. Cross-attention fusion to produce unified latent representation z_t
  5. Temporal position encoding for long-range dependencies
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class TemporalPositionEncoding(nn.Module):
    """Sinusoidal temporal position encoding to enhance long-range dependency."""

    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] -> [B, T, D]"""
        return x + self.pe[:, : x.size(1)]


class VisualEncoder(nn.Module):
    """
    Per-frame visual encoder using pretrained ResNet-18.
    Handles multi-view (6 cameras) by processing each view independently.
    """

    def __init__(self, feature_dim: int = 512, freeze_backbone: bool = False):
        super().__init__()
        # Load pretrained ResNet-18
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Remove final FC and pooling
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        # Adaptive pooling to fixed spatial size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Project to feature dimension
        self.proj = nn.Linear(512, feature_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*V, 3, H, W] where B=batch, V=num_views
        Returns:
            [B*V, D] visual features
        """
        feat = self.backbone(x)          # [B*V, 512, H', W']
        feat = self.pool(feat)           # [B*V, 512, 1, 1]
        feat = feat.flatten(1)           # [B*V, 512]
        feat = self.proj(feat)           # [B*V, D]
        return feat


class VehicleStateEncoder(nn.Module):
    """MLP encoder for ego-vehicle state vector s_t ∈ R^8."""

    def __init__(self, state_dim: int = 8, hidden_dim: int = 256, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: [B, T, 8] -> [B, T, D]"""
        return self.net(s)


class TerrainSemanticEncoder(nn.Module):
    """
    Conv + MLP encoder for terrain semantic features.
    Input includes terrain labels, slope maps, friction maps.
    """

    def __init__(self, terrain_dim: int = 16, hidden_dim: int = 256, out_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(terrain_dim, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, terrain: torch.Tensor) -> torch.Tensor:
        """
        Args:
            terrain: [B*T, C_terrain, H, W]
        Returns:
            [B, T, D]
        """
        B_T = terrain.size(0)
        feat = self.conv(terrain)        # [B*T, 128, 1, 1]
        feat = feat.flatten(1)           # [B*T, 128]
        feat = self.proj(feat)           # [B*T, D]
        return feat


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion: Q from visual features, K/V from
    vehicle state + terrain semantic features. (Equation 2)
    Attn(Q, K, V) = softmax(Q·K^T / √d_k) · V
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        visual_feat: torch.Tensor,       # Q: [B, T, D]
        vehicle_terrain_feat: torch.Tensor,  # K/V: [B, T, D]
    ) -> torch.Tensor:
        attn_out, _ = self.attention(
            query=visual_feat,
            key=vehicle_terrain_feat,
            value=vehicle_terrain_feat,
        )
        out = self.norm(visual_feat + self.dropout(attn_out))
        return out


class OffRoadSceneEncoder(nn.Module):
    """
    Unified multi-modal latent scene representation.
    Integrates visual, vehicle state, and terrain semantics into z_t.

    Pipeline:
      Images (6-view) -> ResNet-18 -> Visual Feats [Q]
      Vehicle State    -> MLP      -> State Feats  [K/V]
      Terrain Semantic -> Conv+MLP -> Terrain Feats [K/V]
      Concat[State, Terrain] -> Cross-Attn with Visual -> z_t
      z_t + Temporal PE -> Temporal Transformer -> Latent Sequence Z
    """

    def __init__(
        self,
        visual_feat_dim: int = 512,
        temporal_hidden_dim: int = 256,
        temporal_num_heads: int = 8,
        temporal_num_layers: int = 4,
        temporal_dropout: float = 0.1,
        vehicle_state_dim: int = 8,
        terrain_semantic_dim: int = 16,
        unified_latent_dim: int = 512,
        num_views: int = 6,
        freeze_visual_backbone: bool = False,
    ):
        super().__init__()
        self.num_views = num_views
        self.unified_latent_dim = unified_latent_dim

        # Sub-encoders
        self.visual_encoder = VisualEncoder(visual_feat_dim, freeze_visual_backbone)
        self.vehicle_encoder = VehicleStateEncoder(vehicle_state_dim, 256, unified_latent_dim)
        self.terrain_encoder = TerrainSemanticEncoder(terrain_semantic_dim, 256, unified_latent_dim)

        # Cross-attention fusion
        self.cross_attn = CrossAttentionFusion(unified_latent_dim, temporal_num_heads, temporal_dropout)

        # Temporal modeling
        self.temporal_pe = TemporalPositionEncoding(unified_latent_dim)
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=unified_latent_dim,
                nhead=temporal_num_heads,
                dim_feedforward=temporal_hidden_dim * 4,
                dropout=temporal_dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=temporal_num_layers,
        )
        self.temporal_norm = nn.LayerNorm(unified_latent_dim)

    def _reshape_multi_view(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Reshape [B, V, T, C, H, W] -> [B*V*T, C, H, W]"""
        return x.permute(0, 2, 1, 3, 4, 5).reshape(-1, *x.shape[3:])

    def forward(
        self,
        images: torch.Tensor,            # [B, T, V, 3, H, W]
        vehicle_state: torch.Tensor,     # [B, T, 8]
        terrain_semantic: torch.Tensor,  # [B, T, C_terrain, H_t, W_t]
    ) -> torch.Tensor:
        """
        Returns:
            Z: [B, T, D] unified latent scene representation
        """
        B, T, V = images.shape[:3]

        # 1. Visual encoding: process all frames x views
        images_flat = images.reshape(B * T * V, *images.shape[3:])  # [B*T*V, 3, H, W]
        visual_feat = self.visual_encoder(images_flat)               # [B*T*V, D]
        visual_feat = visual_feat.view(B, T, V, -1).mean(dim=2)     # [B, T, D] — mean over views

        # 2. Vehicle state encoding
        vehicle_feat = self.vehicle_encoder(vehicle_state)           # [B, T, D]

        # 3. Terrain semantic encoding
        terrain_flat = terrain_semantic.reshape(B * T, *terrain_semantic.shape[2:])
        terrain_feat = self.terrain_encoder(terrain_flat)           # [B*T, D]
        terrain_feat = terrain_feat.view(B, T, -1)                  # [B, T, D]

        # 4. Concatenate vehicle + terrain as K/V source
        kv_feat = vehicle_feat + terrain_feat                       # [B, T, D]

        # 5. Cross-attention fusion (Eq. 2)
        z = self.cross_attn(visual_feat, kv_feat)                   # [B, T, D]

        # 6. Temporal position encoding
        z = self.temporal_pe(z)                                     # [B, T, D]

        # 7. Temporal Transformer
        z = self.temporal_transformer(z)                            # [B, T, D]
        z = self.temporal_norm(z)

        return z  # [B, T, D] unified latent sequence


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("Testing OffRoadSceneEncoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OffRoadSceneEncoder(
        visual_feat_dim=512,
        temporal_hidden_dim=256,
        temporal_num_heads=8,
        temporal_num_layers=4,
        vehicle_state_dim=8,
        terrain_semantic_dim=16,
        unified_latent_dim=512,
        num_views=6,
        freeze_visual_backbone=True,
    ).to(device)

    # Dummy inputs
    B, T, V = 2, 16, 6
    dummy_images = torch.randn(B, T, V, 3, 224, 224).to(device)
    dummy_state = torch.randn(B, T, 8).to(device)
    dummy_terrain = torch.randn(B, T, 16, 56, 56).to(device)

    with torch.no_grad():
        z = model(dummy_images, dummy_state, dummy_terrain)

    print(f"Input images:  {dummy_images.shape}")
    print(f"Output latent: {z.shape}")
    print(f"#Params: {sum(p.numel() for p in model.parameters()):,}")
    print("Test passed!")
