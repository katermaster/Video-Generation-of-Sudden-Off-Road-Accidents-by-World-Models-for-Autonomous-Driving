"""
Demo utilities for creating test inputs without a real dataset.
Generates synthetic multi-view driving scenes for pipeline testing.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Optional, Tuple, List

# Accident type definitions (matching paper Section 4.1)
ACCIDENT_TYPES = {
    0: {
        "name": "animal_intrusion",
        "description": "Animal suddenly enters driving path",
        "control": [0.0, 8.0, 0.3, 0.1, 0.0, 1.5],  # type, trigger, range, friction, slope, size
    },
    1: {
        "name": "rockfall",
        "description": "Rocks falling from slope onto vehicle path",
        "control": [1.0, 10.0, 0.5, 0.1, 0.8, 3.0],
    },
    2: {
        "name": "terrain_collapse",
        "description": "Ground collapses beneath or ahead of vehicle",
        "control": [2.0, 7.0, 0.8, 0.1, 0.6, 4.0],
    },
    3: {
        "name": "sudden_skidding",
        "description": "Sudden loss of traction due to friction change",
        "control": [3.0, 8.0, 0.4, 0.3, 0.1, 2.0],
    },
}


def list_accident_types() -> Dict:
    """Return available accident types and their descriptions."""
    return ACCIDENT_TYPES


def load_demo_image(path: Optional[str] = None, size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """
    Load a demo image or create a synthetic off-road scene.

    Args:
        path: path to image file, or None for synthetic
        size: (H, W) output size
    Returns:
        image tensor [3, H, W] in range [-1, 1]
    """
    if path is not None:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((size[1], size[0]), Image.LANCZOS)
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
            return img
        except Exception as e:
            print(f"[WARN] Could not load image '{path}': {e}")
            print("[INFO] Falling back to synthetic demo image.")

    # Create a synthetic off-road scene
    H, W = size
    img = torch.zeros(3, H, W)

    # Sky gradient (top 40%)
    for y in range(int(H * 0.4)):
        t = y / (H * 0.4)
        img[0, y] = 0.3 + 0.2 * t  # R: darker at top
        img[1, y] = 0.4 + 0.3 * t  # G
        img[2, y] = 0.6 + 0.2 * t  # B: bluer at top

    # Ground (brownish, bottom 60%)
    for y in range(int(H * 0.4), H):
        img[0, y] = 0.5 + 0.1 * torch.randn(1).item()
        img[1, y] = 0.35 + 0.05 * torch.randn(1).item()
        img[2, y] = 0.1 + 0.05 * torch.randn(1).item()

    # Rocky texture on ground
    for _ in range(20):
        rx = np.random.randint(0, W)
        ry = np.random.randint(int(H * 0.45), H)
        rr = np.random.randint(3, 15)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij',
        )
        dist = (x_grid - rx) ** 2 + (y_grid - ry) ** 2
        mask = dist < rr ** 2
        img[0][mask] *= 0.7
        img[1][mask] *= 0.7
        img[2][mask] *= 0.7

    # Dirt road (center, slightly winding)
    road_center = W // 2
    for y in range(H):
        road_wobble = int(10 * np.sin(y / 30.0))
        road_left = road_center - 25 + road_wobble
        road_right = road_center + 25 + road_wobble
        if 0 < road_left < road_right < W:
            img[0, y, road_left:road_right] += 0.1
            img[1, y, road_left:road_right] += 0.05

    img = img.clamp(-1, 1)
    return img


def create_demo_input(
    num_frames: int = 16,
    num_views: int = 6,
    image_size: Tuple[int, int] = (256, 256),
    accident_type_id: int = 3,  # sudden_skidding
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """
    Create a complete demo input dictionary for the world model.

    Args:
        num_frames: temporal window T
        num_views: number of camera views (default 6)
        image_size: (H, W) of each frame
        accident_type_id: 0=animal, 1=rockfall, 2=terrain_collapse, 3=skidding
        device: torch device
    Returns:
        dict with all required inputs for OffRoadWorldModel.forward()
    """
    H, W = image_size
    B = 1  # batch size for demo

    # Create base off-road scene
    base_img = load_demo_image(None, image_size)

    # Create multi-view video: [B, T, V, 3, H, W]
    # Simulate slight camera motion over time
    images = []
    for t in range(num_frames):
        frame_views = []
        for v in range(num_views):
            # Add slight variation per view and time
            frame = base_img.clone()
            noise = torch.randn(3, H, W) * 0.02
            frame = frame + noise
            # Simulate forward motion: slight zoom
            if t > 0:
                shift = t * 0.01
                frame = frame * (1.0 + shift * 0.1)
            frame = frame.clamp(-1, 1)
            frame_views.append(frame)
        images.append(torch.stack(frame_views))

    images = torch.stack(images).unsqueeze(0)  # [1, T, V, 3, H, W]

    # Vehicle state: [B, T, 8] (speed, accel_3d(3), yaw, roll, slip_rate)
    vehicle_state = torch.zeros(B, num_frames, 8)
    vehicle_state[:, :, 0] = torch.linspace(0.3, 0.5, num_frames)  # speed 30-50 km/h
    vehicle_state[:, :, 1] = 0.01  # slight forward acceleration
    vehicle_state[:, :, 4] = torch.sin(torch.linspace(0, 0.5, num_frames)) * 0.1  # yaw

    # Terrain semantic: [B, T, 16, 56, 56]
    terrain_semantic = torch.randn(B, num_frames, 16, 56, 56) * 0.5

    # Terrain physical params: [B, 8] (friction, slope, loose_degree, elevation, roughness, ...)
    terrain_phys = torch.tensor([[0.6, 0.2, 0.3, 0.1, 0.4, 0.1, 0.5, 0.2]])

    # Accident control vector: [B, 6]
    accident_info = ACCIDENT_TYPES[accident_type_id]
    control_vector = torch.tensor([accident_info["control"]])

    # Conditioning frame: [B, 3, H, W] (first frame, front view)
    conditioning_frame = images[0, 0, 0].clone()  # [3, H, W]
    conditioning_frame = conditioning_frame.unsqueeze(0)  # [1, 3, H, W]

    # Move to device
    inputs = {
        "images": images.to(device),
        "vehicle_state": vehicle_state.to(device),
        "terrain_semantic": terrain_semantic.to(device),
        "terrain_phys": terrain_phys.to(device),
        "control_vector": control_vector.to(device),
        "conditioning_frame": conditioning_frame.to(device),
        "accident_type": ACCIDENT_TYPES[accident_type_id]["name"],
    }

    return inputs


if __name__ == "__main__":
    print("Testing demo data creation...")
    inputs = create_demo_input(
        num_frames=16,
        num_views=6,
        image_size=(256, 256),
        accident_type_id=3,
    )
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {v}")
    print("Demo data created successfully!")
