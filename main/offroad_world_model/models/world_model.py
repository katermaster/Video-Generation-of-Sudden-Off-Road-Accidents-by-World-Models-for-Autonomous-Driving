"""
Off-Road World Model for Sudden Accident Video Generation

Main model combining all four core modules:
  1. OffRoadSceneEncoder        — Multi-modal scene representation
  2. TerrainAwareWorldDynamics  — Future latent state prediction
  3. AccidentController         — Conditional accident triggering
  4. OpticalFlowDiffusionDecoder — Video generation from latents

Paper: Video Generation of Sudden Off-Road Accidents by World Models
       for Autonomous Driving (ACCV 2026, Paper #41)
"""

from typing import Optional, Dict
import torch
import torch.nn as nn

from .scene_encoder import OffRoadSceneEncoder
from .world_dynamics import TerrainAwareWorldDynamics
from .accident_controller import AccidentController
from .diffusion_decoder import OpticalFlowDiffusionDecoder


class OffRoadWorldModel(nn.Module):
    """
    Terrain-aware safety-centric world model for off-road autonomous driving.

    Complete pipeline:
      Input: Multi-view images, vehicle state, terrain semantics, accident control
      └─> SceneEncoder: images + state + terrain → latent Z
      └─> WorldDynamics: Z + terrain_phys → future latent Z_pred
      └─> AccidentController: Z_pred + control_vector → conditional Z_cond
      └─> DiffusionDecoder: Z_cond + conditioning_frame → accident video
      Output: Generated accident video + metadata
    """

    def __init__(
        self,
        # Scene encoder
        visual_feat_dim: int = 512,
        temporal_hidden_dim: int = 256,
        temporal_num_heads: int = 8,
        temporal_num_layers: int = 4,
        vehicle_state_dim: int = 8,
        terrain_semantic_dim: int = 16,
        unified_latent_dim: int = 512,
        num_views: int = 6,
        freeze_visual_backbone: bool = True,
        # World dynamics
        terrain_phys_dim: int = 8,
        dynamics_num_layers: int = 6,
        prediction_horizon: int = 16,
        # Accident controller
        control_vector_dim: int = 6,
        num_accident_types: int = 4,
        # Diffusion decoder
        diffusion_model_id: str = "stabilityai/stable-video-diffusion-img2vid",
        output_fps: int = 10,
        use_fp16: bool = True,
    ):
        super().__init__()
        self.unified_latent_dim = unified_latent_dim
        self.prediction_horizon = prediction_horizon
        self.num_views = num_views
        self.num_accident_types = num_accident_types

        # 1. Scene Encoder
        self.scene_encoder = OffRoadSceneEncoder(
            visual_feat_dim=visual_feat_dim,
            temporal_hidden_dim=temporal_hidden_dim,
            temporal_num_heads=temporal_num_heads,
            temporal_num_layers=temporal_num_layers,
            vehicle_state_dim=vehicle_state_dim,
            terrain_semantic_dim=terrain_semantic_dim,
            unified_latent_dim=unified_latent_dim,
            num_views=num_views,
            freeze_visual_backbone=freeze_visual_backbone,
        )

        # 2. World Dynamics (predicts future states from past latents)
        self.world_dynamics = TerrainAwareWorldDynamics(
            latent_dim=unified_latent_dim,
            terrain_phys_dim=terrain_phys_dim,
            transformer_num_layers=dynamics_num_layers,
            transformer_num_heads=temporal_num_heads,
            prediction_horizon=prediction_horizon,
        )

        # 3. Accident Controller
        self.accident_controller = AccidentController(
            control_vector_dim=control_vector_dim,
            latent_dim=unified_latent_dim,
            num_accident_types=num_accident_types,
        )

        # 4. Diffusion Decoder
        self.diffusion_decoder = OpticalFlowDiffusionDecoder(
            model_id=diffusion_model_id,
            latent_dim=unified_latent_dim,
            num_frames=prediction_horizon,
            output_fps=output_fps,
            use_fp16=use_fp16,
        )

    def load_pretrained_decoder(self, device: torch.device):
        """Load pretrained SVD for the diffusion decoder."""
        self.diffusion_decoder.load_pretrained(device)

    def encode_scene(
        self,
        images: torch.Tensor,
        vehicle_state: torch.Tensor,
        terrain_semantic: torch.Tensor,
    ) -> torch.Tensor:
        """Encode multi-modal observations into latent representations."""
        return self.scene_encoder(images, vehicle_state, terrain_semantic)

    def predict_future(
        self,
        z: torch.Tensor,
        terrain_phys: torch.Tensor,
    ) -> Dict:
        """Predict future latent states with terrain-aware dynamics."""
        return self.world_dynamics(z, terrain_phys)

    def apply_accident(
        self,
        z_pred: torch.Tensor,
        control_vector: torch.Tensor,
    ) -> Dict:
        """Inject accident conditions into predicted future latent states."""
        return self.accident_controller(z_pred, control_vector)

    def generate_video(
        self,
        z_cond: torch.Tensor,
        conditioning_frame: torch.Tensor,
        physical_params: Optional[Dict] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        optical_flow_weight: float = 0.3,
    ) -> torch.Tensor:
        """Generate accident video from conditional latent states."""
        return self.diffusion_decoder(
            z_cond=z_cond,
            conditioning_frame=conditioning_frame,
            physical_params=physical_params,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            optical_flow_weight=optical_flow_weight,
        )

    def forward(
        self,
        images: torch.Tensor,                # [B, T, V, 3, H, W]
        vehicle_state: torch.Tensor,         # [B, T, 8]
        terrain_semantic: torch.Tensor,      # [B, T, C_terrain, H_t, W_t]
        terrain_phys: torch.Tensor,          # [B, 8]
        control_vector: torch.Tensor,        # [B, 6]
        conditioning_frame: torch.Tensor,    # [B, 3, H_img, W_img]
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        optical_flow_weight: float = 0.3,
    ) -> Dict:
        """
        Full pipeline: encode → predict → control → generate.

        Returns:
            dict with:
              - 'video': [B, T', 3, H, W] generated accident video
              - 'latent_z': [B, T, D] scene latents
              - 'z_pred': [B, T', D] predicted future latents
              - 'z_cond': [B, T', D] conditional accident latents
              - 'physical_params': dict
              - 'latent_reg': scalar regularization loss
        """
        # Stage 1: Encode scene
        z = self.encode_scene(images, vehicle_state, terrain_semantic)

        # Stage 2: Predict future latent states
        dynamics_out = self.predict_future(z, terrain_phys)
        z_pred = dynamics_out["z_pred"]
        latent_reg = dynamics_out["latent_reg"]

        # Stage 3: Inject accident conditions
        accident_out = self.apply_accident(z_pred, control_vector)
        z_cond = accident_out["z_cond"]
        physical_params = accident_out["physical_params"]

        # Stage 4: Generate accident video
        video = self.generate_video(
            z_cond=z_cond,
            conditioning_frame=conditioning_frame,
            physical_params=physical_params,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            optical_flow_weight=optical_flow_weight,
        )

        return {
            "video": video,
            "latent_z": z,
            "z_pred": z_pred,
            "z_cond": z_cond,
            "physical_params": physical_params,
            "latent_reg": latent_reg,
        }


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("OffRoadWorldModel — Full Pipeline Test")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = OffRoadWorldModel(
        visual_feat_dim=512,
        temporal_hidden_dim=256,
        temporal_num_heads=8,
        temporal_num_layers=2,         # Small for testing
        vehicle_state_dim=8,
        terrain_semantic_dim=16,
        unified_latent_dim=512,
        num_views=6,
        freeze_visual_backbone=True,
        terrain_phys_dim=8,
        dynamics_num_layers=3,         # Small for testing
        prediction_horizon=16,
        control_vector_dim=6,
        num_accident_types=4,
        diffusion_model_id="stabilityai/stable-video-diffusion-img2vid",
        output_fps=10,
        use_fp16=False,
    ).to(device)

    # Create dummy inputs matching paper specifications
    B, T, V = 1, 16, 6
    H_img, W_img = 256, 256
    H_terrain, W_terrain = 56, 56

    dummy_images = torch.randn(B, T, V, 3, H_img, W_img).to(device)
    dummy_vehicle_state = torch.randn(B, T, 8).to(device)
    dummy_terrain_semantic = torch.randn(B, T, 16, H_terrain, W_terrain).to(device)
    dummy_terrain_phys = torch.randn(B, 8).to(device)

    # Accident: sudden skidding (type=3), trigger at frame 8
    dummy_control = torch.tensor([[3.0, 8.0, 0.4, 0.3, 0.5, 2.0]]).to(device)
    dummy_cond_frame = torch.randn(B, 3, H_img, W_img).to(device)

    print("\nInput shapes:")
    print(f"  images:           {dummy_images.shape}")
    print(f"  vehicle_state:    {dummy_vehicle_state.shape}")
    print(f"  terrain_semantic: {dummy_terrain_semantic.shape}")
    print(f"  terrain_phys:     {dummy_terrain_phys.shape}")
    print(f"  control_vector:   {dummy_control.shape}")
    print(f"  cond_frame:       {dummy_cond_frame.shape}")

    print("\nRunning forward pass...")
    with torch.no_grad():
        out = model(
            images=dummy_images,
            vehicle_state=dummy_vehicle_state,
            terrain_semantic=dummy_terrain_semantic,
            terrain_phys=dummy_terrain_phys,
            control_vector=dummy_control,
            conditioning_frame=dummy_cond_frame,
            num_inference_steps=5,  # Minimal for quick test
        )

    print("\nOutput shapes:")
    print(f"  video:          {out['video'].shape}")
    print(f"  latent_z:       {out['latent_z'].shape}")
    print(f"  z_pred:         {out['z_pred'].shape}")
    print(f"  z_cond:         {out['z_cond'].shape}")
    print(f"  latent_reg:     {out['latent_reg'].item():.6f}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel stats:")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Frozen params:    {total_params - trainable_params:,}")

    print("\n" + "=" * 60)
    print("Full pipeline test passed!")
    print("=" * 60)
