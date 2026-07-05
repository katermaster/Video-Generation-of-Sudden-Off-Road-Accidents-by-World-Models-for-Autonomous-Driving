"""
Evaluation metrics for accident video generation (Section 4.3).

Metrics categories:
  1. Generation quality: PSNR, SSIM, LPIPS
  2. Temporal consistency: optical flow error, temporal consistency score
  3. Physical plausibility: trajectory plausibility score
"""

import torch
import torch.nn.functional as F


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio.
    Higher = better reconstruction quality.
    Paper target: 32.74 dB
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(1.0 / mse).item()


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """
    Structural Similarity Index.
    Higher = better structural similarity.
    Paper target: 0.961
    """
    # Simple SSIM implementation
    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device)
    kernel = kernel / window_size ** 2

    mu1 = F.conv2d(pred, kernel, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(target, kernel, padding=window_size // 2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def compute_optical_flow_error(
    pred_video: torch.Tensor,
    gt_video: torch.Tensor,
) -> float:
    """
    Mean End-Point Error (EPE) of optical flow.
    Lower = better temporal consistency.
    Paper target: 0.049
    """
    from .optical_flow import OpticalFlowUtils

    flow_pred = OpticalFlowUtils.compute_optical_flow_sequence(pred_video)
    flow_gt = OpticalFlowUtils.compute_optical_flow_sequence(gt_video)

    error = torch.abs(flow_pred - flow_gt).mean().item()
    return error


def compute_trajectory_score(
    pred_video: torch.Tensor,
    gt_trajectory: torch.Tensor,
) -> float:
    """
    Trajectory plausibility score.
    Higher = more physically plausible motion.
    Paper target: 0.937

    Simplified: measures consistency of object motion across frames.
    """
    B, T, C, H, W = pred_video.shape

    # Extract motion vectors from optical flow
    from .optical_flow import OpticalFlowUtils

    motions_pred = []
    motions_gt = []
    for b in range(B):
        for t in range(T - 1):
            flow_p = OpticalFlowUtils.compute_farneback_flow(
                pred_video[b, t], pred_video[b, t + 1]
            )
            flow_g = OpticalFlowUtils.compute_farneback_flow(
                gt_video[b, t], gt_video[b, t + 1]
            )
            motions_pred.append(flow_p.mean().item())
            motions_gt.append(flow_g.mean().item())

    # Correlation between predicted and ground-truth motion patterns
    pred_tensor = torch.tensor(motions_pred)
    gt_tensor = torch.tensor(motions_gt)

    # Pearson correlation as trajectory score
    pred_centered = pred_tensor - pred_tensor.mean()
    gt_centered = gt_tensor - gt_tensor.mean()

    corr = (pred_centered * gt_centered).sum() / (
        torch.sqrt((pred_centered ** 2).sum()) * torch.sqrt((gt_centered ** 2).sum()) + 1e-8
    )

    return corr.item()
