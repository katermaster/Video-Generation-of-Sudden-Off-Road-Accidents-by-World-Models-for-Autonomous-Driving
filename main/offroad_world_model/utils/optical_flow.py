"""Optical flow utility functions for temporal consistency."""

import torch
import torch.nn.functional as F


class OpticalFlowUtils:
    """Utilities for optical flow computation and visualization."""

    @staticmethod
    def compute_farneback_flow(frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """
        Simple gradient-based optical flow (pure PyTorch, no OpenCV needed).

        Args:
            frame1: [B, 3, H, W] or [3, H, W]
            frame2: same shape
        Returns:
            flow: [B, 2, H, W] or [2, H, W]
        """
        squeeze_batch = frame1.dim() == 3
        if squeeze_batch:
            frame1 = frame1.unsqueeze(0)
            frame2 = frame2.unsqueeze(0)

        B = frame1.size(0)
        H, W = frame1.shape[-2:]

        # Convert to grayscale
        def to_gray(img):
            return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

        gray1 = to_gray(frame1)
        gray2 = to_gray(frame2)

        # Sobel kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device=frame1.device
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device=frame1.device
        ).view(1, 1, 3, 3)

        # Spatial gradients
        Ix = F.conv2d(gray1, sobel_x, padding=1) / 8.0
        Iy = F.conv2d(gray1, sobel_y, padding=1) / 8.0

        # Temporal gradient
        It = gray2 - gray1

        # Lucas-Kanade approximation
        eps = 1e-6
        denom = Ix.pow(2) + Iy.pow(2) + eps
        u = -Ix * It / denom
        v = -Iy * It / denom

        flow = torch.cat([u, v], dim=1)

        if squeeze_batch:
            flow = flow.squeeze(0)
        return flow

    @staticmethod
    def flow_to_rgb(flow: torch.Tensor) -> torch.Tensor:
        """Convert optical flow to RGB for visualization."""
        if flow.dim() == 3:
            flow = flow.unsqueeze(0)

        B, _, H, W = flow.shape
        u, v = flow[:, 0], flow[:, 1]

        # Magnitude and angle
        mag = torch.sqrt(u.pow(2) + v.pow(2))
        ang = torch.atan2(v, u)

        # HSV-like coloring
        h = ang / (2 * torch.pi) % 1.0
        s = torch.ones_like(h)
        v_mag = mag / (mag.max() + 1e-6)

        # HSV to RGB (simplified)
        h6 = h * 6.0
        i = h6.long().clamp(0, 5)
        f = h6 - i.float()
        p = v_mag * (1 - s)
        q = v_mag * (1 - f * s)
        t = v_mag * (1 - (1 - f) * s)

        rgb = torch.zeros(B, 3, H, W, device=flow.device)
        # This is a simplification; for production use matplotlib/torchvision
        rgb[:, 0] = mag / (mag.max() + 1e-6)  # Red channel = magnitude
        rgb[:, 1] = (ang + torch.pi) / (2 * torch.pi)  # Green channel = direction

        return rgb.clamp(0, 1)

    @staticmethod
    def compute_video_flow_error(video: torch.Tensor) -> float:
        """
        Compute mean optical flow error for a video.
        Lower values = smoother motion (Eq. 4 application).

        Args:
            video: [T, 3, H, W] or [B, T, 3, H, W]
        Returns:
            mean flow error
        """
        if video.dim() == 4:
            video = video.unsqueeze(0)

        B, T = video.shape[:2]
        errors = []
        for t in range(T - 1):
            flow = OpticalFlowUtils.compute_farneback_flow(
                video[0, t], video[0, t + 1]
            )
            errors.append(flow.pow(2).mean().item())

        return sum(errors) / len(errors) if errors else 0.0
