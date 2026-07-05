"""
Visualization utilities for generated accident videos.
"""

import os
import torch
import numpy as np
from typing import Optional, List


def tensor_to_numpy(video: torch.Tensor) -> np.ndarray:
    """
    Convert video tensor to numpy array for saving.

    Args:
        video: [T, 3, H, W] or [B, T, 3, H, W] in range [-1, 1] or [0, 1]
    Returns:
        numpy array [T, H, W, 3] in range [0, 255] uint8
    """
    if video.dim() == 5:
        video = video[0]  # Take first batch

    # [T, 3, H, W] -> [T, H, W, 3]
    video = video.permute(0, 2, 3, 1)

    # Normalize to [0, 1]
    if video.min() < 0:
        video = (video + 1.0) / 2.0

    video = video.clamp(0, 1)
    video = (video * 255).to(torch.uint8)
    return video.cpu().numpy()


def save_video(
    video: torch.Tensor,
    output_path: str,
    fps: int = 10,
) -> None:
    """
    Save video tensor to MP4 file.

    Args:
        video: [T, 3, H, W] tensor
        output_path: output file path (.mp4)
        fps: frames per second
    """
    frames = tensor_to_numpy(video)

    try:
        import imageio
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"[INFO] Video saved to: {output_path}")
    except ImportError:
        # Fallback: save as individual frames
        output_dir = output_path.replace(".mp4", "_frames")
        os.makedirs(output_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            from PIL import Image
            img = Image.fromarray(frame)
            img.save(os.path.join(output_dir, f"frame_{i:04d}.png"))
        print(f"[INFO] Frames saved to: {output_dir}/")
    except Exception as e:
        print(f"[WARN] Could not save video: {e}")
        # Save frames as fallback
        output_dir = output_path.replace(".mp4", "_frames")
        os.makedirs(output_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            from PIL import Image
            img = Image.fromarray(frame)
            img.save(os.path.join(output_dir, f"frame_{i:04d}.png"))
        print(f"[INFO] Frames saved to: {output_dir}/")


def make_comparison_grid(
    original: torch.Tensor,
    generated: torch.Tensor,
    num_frames: int = 4,
) -> torch.Tensor:
    """
    Create a comparison grid of original vs generated frames.

    Args:
        original: [T, 3, H, W] original video
        generated: [T, 3, H, W] generated video
        num_frames: number of frames to show
    Returns:
        grid image [3, H*2, W*num_frames]
    """
    T = original.size(0)
    indices = torch.linspace(0, T - 1, num_frames).long()

    rows = []
    for idx in indices:
        rows.append(torch.cat([original[idx], generated[idx]], dim=-1))  # side by side

    grid = torch.cat(rows, dim=-1)  # horizontal strip
    return grid


def overlay_accident_labels(
    video: torch.Tensor,
    labels: List[str],
    accident_type: str = "unknown",
) -> torch.Tensor:
    """
    Add text overlay to video indicating accident type and frame info.

    This is a placeholder — for actual text rendering, use PIL or OpenCV.
    """
    # Returns video unchanged for now; PIL/OpenCV text rendering
    # would go here in production
    return video
