#!/usr/bin/env python3
"""
Inference script for Off-Road Sudden Accident Video Generation.

Usage:
    # Quick demo with synthetic scene
    python inference.py --demo

    # Generate specific accident type
    python inference.py --demo --accident_type sudden_skidding
    python inference.py --demo --accident_type animal_intrusion
    python inference.py --demo --accident_type rockfall
    python inference.py --demo --accident_type terrain_collapse

    # Use a real image as conditioning frame
    python inference.py --conditioning_image path/to/offroad_scene.jpg

    # Load checkpoint
    python inference.py --checkpoint path/to/checkpoint.pth --demo

Paper: Video Generation of Sudden Off-Road Accidents by World Models
       for Autonomous Driving (ACCV 2026, Paper #41)
"""

import os
import sys
import argparse
import warnings
from typing import Dict

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.world_model import OffRoadWorldModel
from data.demo_utils import create_demo_input, load_demo_image, list_accident_types
from utils.visualization import save_video, make_comparison_grid

warnings.filterwarnings("ignore")


# Accident type name to ID mapping
ACCIDENT_NAME_TO_ID = {
    "animal_intrusion": 0,
    "rockfall": 1,
    "terrain_collapse": 2,
    "sudden_skidding": 3,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Off-Road Sudden Accident Video Generation"
    )

    # Input options
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with synthetic demo scene (no input file needed)"
    )
    parser.add_argument(
        "--conditioning_image", type=str, default=None,
        help="Path to conditioning image (off-road scene)"
    )
    parser.add_argument(
        "--input_video", type=str, default=None,
        help="Path to input video for scene encoding (optional)"
    )

    # Accident control
    parser.add_argument(
        "--accident_type", type=str, default="sudden_skidding",
        choices=list(ACCIDENT_NAME_TO_ID.keys()),
        help="Type of accident to generate"
    )
    parser.add_argument(
        "--trigger_frame", type=int, default=8,
        help="Frame index at which accident is triggered"
    )
    parser.add_argument(
        "--friction_atten", type=float, default=0.3,
        help="Friction attenuation ratio for skidding (0.1-0.5)"
    )

    # Model options
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--num_frames", type=int, default=16,
        help="Number of video frames to generate"
    )
    parser.add_argument(
        "--image_size", type=int, nargs=2, default=[256, 256],
        help="Output image size (H W)"
    )

    # Diffusion options
    parser.add_argument(
        "--num_inference_steps", type=int, default=50,
        help="Diffusion denoising steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--optical_flow_weight", type=float, default=0.3,
        help="Weight for optical flow consistency loss"
    )

    # Output options
    parser.add_argument(
        "--output", type=str, default="./outputs/accident_video.mp4",
        help="Output video path"
    )
    parser.add_argument(
        "--output_fps", type=int, default=10,
        help="Output video FPS"
    )
    parser.add_argument(
        "--save_latents", action="store_true",
        help="Also save latent representations"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on"
    )

    # Display options
    parser.add_argument(
        "--list_accidents", action="store_true",
        help="List all available accident types and exit"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed information"
    )

    return parser.parse_args()


def get_device(preferred: str = "auto") -> torch.device:
    """Determine the best available device."""
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(
    num_frames: int = 16,
    num_views: int = 6,
    freeze_backbone: bool = True,
    use_fp16: bool = True,
    checkpoint_path: str = None,
    device: torch.device = torch.device("cpu"),
) -> OffRoadWorldModel:
    """
    Build and optionally load checkpoint for the world model.
    """
    print("[INFO] Building OffRoadWorldModel...")

    model = OffRoadWorldModel(
        visual_feat_dim=512,
        temporal_hidden_dim=256,
        temporal_num_heads=8,
        temporal_num_layers=4,
        vehicle_state_dim=8,
        terrain_semantic_dim=16,
        unified_latent_dim=512,
        num_views=num_views,
        freeze_visual_backbone=freeze_backbone,
        terrain_phys_dim=8,
        dynamics_num_layers=6,
        prediction_horizon=num_frames,
        control_vector_dim=6,
        num_accident_types=4,
        diffusion_model_id="stabilityai/stable-video-diffusion-img2vid",
        output_fps=10,
        use_fp16=use_fp16 and device.type == "cuda",
    )

    # Load checkpoint if provided
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print("[INFO] Checkpoint loaded (strict=False)")
    else:
        print("[INFO] No checkpoint loaded — using randomly initialized weights")
        print("[INFO] Scene encoder ResNet-18 uses pretrained ImageNet weights")
        print("[INFO] Accident controller and dynamics are randomly initialized")

    model.to(device)
    model.eval()

    # Load pretrained SVD decoder if on CUDA
    if device.type == "cuda":
        try:
            model.load_pretrained_decoder(device)
        except Exception as e:
            print(f"[WARN] Could not load SVD decoder: {e}")
            print("[INFO] Will use built-in fallback decoder")

    return model


def run_inference(
    model: OffRoadWorldModel,
    inputs: Dict,
    args,
    device: torch.device,
) -> Dict:
    """Run full inference pipeline."""
    print("\n" + "=" * 60)
    print("Running inference...")
    print("=" * 60)

    with torch.inference_mode():
        outputs = model(
            images=inputs["images"],
            vehicle_state=inputs["vehicle_state"],
            terrain_semantic=inputs["terrain_semantic"],
            terrain_phys=inputs["terrain_phys"],
            control_vector=inputs["control_vector"],
            conditioning_frame=inputs["conditioning_frame"],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            optical_flow_weight=args.optical_flow_weight,
        )

    return outputs


def print_results(outputs: Dict, model: OffRoadWorldModel):
    """Print inference results summary."""
    print("\n" + "=" * 60)
    print("Inference Results")
    print("=" * 60)

    video = outputs["video"]
    print(f"  Generated video:     {video.shape}")
    print(f"  Value range:         [{video.min().item():.3f}, {video.max().item():.3f}]")
    print(f"  Latent Z shape:      {outputs['latent_z'].shape}")
    print(f"  Predicted Z:         {outputs['z_pred'].shape}")
    print(f"  Conditional Z:       {outputs['z_cond'].shape}")
    print(f"  Latent reg loss:     {outputs['latent_reg'].item():.6f}")

    params = outputs["physical_params"]
    print(f"\n  Accident type:       {params['accident_type'].item()}")
    print(f"  Friction mu_0:       {params['friction_mu0'].item():.3f}")
    print(f"  Friction mu_t:       {params['friction_mu_t'].item():.3f}")
    print(f"  Friction atten:      {params['friction_atten_ratio'].item():.3f}")
    print(f"  Collapse slope:      {params['collapse_slope'].item():.3f}")
    print(f"  Obstacle size:       {params['obstacle_size'].item():.3f}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model params:        {total_params:,} total, {trainable:,} trainable")


def main():
    args = parse_args()

    # List accident types
    if args.list_accidents:
        print("\nAvailable accident types:")
        print("-" * 40)
        for type_id, info in list_accident_types().items():
            print(f"  [{type_id}] {info['name']}")
            print(f"       {info['description']}")
        return

    # Setup device
    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")

    # Build model
    model = build_model(
        num_frames=args.num_frames,
        num_views=6,
        freeze_backbone=True,
        use_fp16=(device.type == "cuda"),
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # Prepare inputs
    accident_type_id = ACCIDENT_NAME_TO_ID[args.accident_type]

    if args.demo:
        print(f"[INFO] Creating synthetic demo scene...")
        print(f"[INFO] Accident type: {args.accident_type} (ID={accident_type_id})")

        inputs = create_demo_input(
            num_frames=args.num_frames,
            num_views=6,
            image_size=tuple(args.image_size),
            accident_type_id=accident_type_id,
            device=device,
        )

        # Override control vector with user-specified params
        inputs["control_vector"][0, 1] = args.trigger_frame
        inputs["control_vector"][0, 3] = args.friction_atten

    elif args.conditioning_image is not None:
        print(f"[INFO] Loading conditioning image: {args.conditioning_image}")

        cond_frame = load_demo_image(args.conditioning_image, tuple(args.image_size))
        cond_frame = cond_frame.unsqueeze(0).to(device)

        # Use demo inputs but override conditioning frame
        inputs = create_demo_input(
            num_frames=args.num_frames,
            num_views=6,
            image_size=tuple(args.image_size),
            accident_type_id=accident_type_id,
            device=device,
        )
        inputs["conditioning_frame"] = cond_frame
        inputs["control_vector"][0, 1] = args.trigger_frame
        inputs["control_vector"][0, 3] = args.friction_atten

    else:
        print("[ERROR] Please specify --demo or --conditioning_image")
        print("Example: python inference.py --demo --accident_type sudden_skidding")
        sys.exit(1)

    # Print input shapes
    if args.verbose:
        print("\nInput shapes:")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")

    # Run inference
    outputs = run_inference(model, inputs, args, device)

    # Print results
    print_results(outputs, model)

    # Save video
    video = outputs["video"]
    if video.dim() == 5:
        video = video[0]  # [T, 3, H, W]

    save_video(
        video.cpu(),
        output_path=args.output,
        fps=args.output_fps,
    )

    print(f"\n[SUCCESS] Accident video saved to: {args.output}")
    print(f"          Accident type: {args.accident_type}")
    print(f"          Frames: {args.num_frames} @ {args.output_fps} FPS")
    print("=" * 60)


if __name__ == "__main__":
    main()
