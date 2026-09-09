"""
Density map construction and two-component training loss.

Density map: (T,) vector where density[t] = counting rate at frame t.
    sum(density) = total count. Peaks at repetition cycle boundaries.

Two construction modes:
    Synthetic  (Countix — count only):   uniform Gaussian placement.
    Accurate   (RepCount — cycle times): Gaussians at real cycle centers.

Per-cycle kernel shape is selectable (kernel="gaussian"|"triangle"|"sawtooth",
see _kernel_values below) so the bump placed at each cycle center need not be
a Gaussian.

Training loss (two components, matching TransRAC):
    L = KL(density_prob, gt_density_prob)     — shape of the density curve
      + alpha * SmoothL1(count_pred, count)   — total count accuracy
"""

import math

import torch
import torch.nn.functional as F

KERNELS = ("gaussian", "triangle", "sawtooth")


# ── density map construction ───────────────────────────────────────────────────

def _kernel_values(t_vals: torch.Tensor, center: float, width: float, kernel: str) -> torch.Tensor:
    """
    Per-cycle bump shape, evaluated at every t_vals position.

    All kernels are centered on `center` and use `width` as their characteristic
    scale (the same role sigma plays for the Gaussian) -- final normalisation
    (sum -> count) happens in the caller, so absolute kernel height doesn't matter.

        gaussian : exp(-0.5*(d/width)^2)                        — unbounded tails
        triangle : linear taper to 0 at |d| == width             — compact, sharp peak
        sawtooth : linear ramp 0 -> 1 over [-width, 0], instant drop after center
    """
    d = t_vals - center
    if kernel == "gaussian":
        return torch.exp(-0.5 * (d / width) ** 2)
    if kernel == "triangle":
        return torch.clamp(1.0 - d.abs() / width, min=0.0)
    if kernel == "sawtooth":
        v = (d + width) / width
        mask = (d >= -width) & (d <= 0.0)
        return torch.where(mask, v, torch.zeros_like(v))
    raise ValueError(f"Unknown density kernel {kernel!r}, expected one of {KERNELS}.")


def make_density_map_uniform(
    count: int, T: int, sigma_factor: float = 0.4, kernel: str = "gaussian",
) -> torch.Tensor:
    """
    Synthetic density map for Countix (count only, no cycle timing).
    Places count kernel bumps at equal intervals over T frames.
    Normalised to sum to count.
    """
    if count <= 0:
        return torch.zeros(T)

    density = torch.zeros(T)
    period  = T / count
    t_vals  = torch.arange(T, dtype=torch.float32)

    for k in range(int(count)):
        center = (k + 0.5) * period
        width  = max(1.0, period * sigma_factor)
        density += _kernel_values(t_vals, center, width, kernel)

    total = density.sum()
    if total > 0:
        density = density / total * count
    return density


def make_density_map_from_frames(
    cycle_starts:  list,
    cycle_ends:    list,
    T:             int,
    window_frames: int,
    sigma_factor:  float = 0.4,
    kernel:        str = "gaussian",
) -> torch.Tensor:
    """
    Density map from frame-indexed cycle annotations (RepCount format).

    Args:
        cycle_starts  : cycle start frame indices, relative to trimmed window start (i.e. already - f_start)
        cycle_ends    : cycle end frame indices, relative to trimmed window start
        T             : number of uniformly sampled output frames
        window_frames : total frames in the trimmed window (f_end - f_start + 1)
        sigma_factor  : kernel width as fraction of cycle duration
        kernel        : bump shape per cycle -- one of KERNELS (see _kernel_values)

    Returns:
        density: (T,) float32, sums to len(cycle_starts)
    """
    count = len(cycle_starts)
    if count == 0 or window_frames <= 0:
        return torch.zeros(T)

    t_vals  = torch.arange(T, dtype=torch.float32)
    density = torch.zeros(T)

    for s, e in zip(cycle_starts, cycle_ends):
        # convert relative frame index → output frame index (scale to T)
        center_frame = ((s + e) / 2) / window_frames * T
        width_frame  = max(((e - s) * sigma_factor) / window_frames * T, 0.5)
        density += _kernel_values(t_vals, center_frame, width_frame, kernel)

    total = density.sum()
    if total > 0:
        density = density / total * count
    return density


def make_density_map_from_cycles(
    cycle_starts:  list,
    cycle_ends:    list,
    T:             int,
    window_start:  float,
    window_end:    float,
    sigma_factor:  float = 0.4,
    kernel:        str = "gaussian",
) -> torch.Tensor:
    """
    Accurate density map from RepCount cycle boundary annotations.
    Places one kernel bump at the center of each annotated cycle.
    Normalised to sum to len(cycle_starts).
    """
    count = len(cycle_starts)
    if count == 0:
        return torch.zeros(T)

    duration = max(window_end - window_start, 1e-6)
    t_vals   = torch.arange(T, dtype=torch.float32)
    density  = torch.zeros(T)

    for s, e in zip(cycle_starts, cycle_ends):
        center_frame = (((s + e) / 2) - window_start) / duration * T
        width_frame  = max(((e - s) * sigma_factor) / duration * T, 0.5)
        density += _kernel_values(t_vals, center_frame, width_frame, kernel)

    total = density.sum()
    if total > 0:
        density = density / total * count
    return density


# ── phase-in-cycle targets (auxiliary task) ────────────────────────────────────

def make_phase_targets(
    cycle_starts:  list,
    cycle_ends:    list,
    frame_indices: list,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-frame progress-through-repetition targets, from the same RepCount cycle
    boundary annotations the density maps use -- no extra annotation files.

    For every requested RAW frame index (not rescaled to any output length --
    each entry of `frame_indices` is a real frame number in the original video,
    e.g. from clip_frame_indices()/range(0, num_frames, stride)):
        if it falls inside an annotated cycle [s, e]:
            phase = (frame - s) / (e - s)              in [0, 1)
            theta = 2*pi*phase
            target_sin, target_cos = sin(theta), cos(theta)
        else:
            target_sin = target_cos = 0, and the frame is marked invalid.

    Regressing (sin, cos) instead of phase directly avoids the phase=1 -> 0
    discontinuity at every cycle boundary, which a direct regression target
    would otherwise have to jump across.

    Assumes cycles are sorted and non-overlapping (RepCount's format); the last
    matching cycle wins for any (annotation-error) overlap.

    Args:
        cycle_starts, cycle_ends : cycle boundary frame indices, same
                                    coordinate space as frame_indices (both
                                    either original-video or window-relative).
        frame_indices             : raw frame index sampled at each output
                                    position, len = T.

    Returns:
        phase_sin, phase_cos, phase_mask : each (T,) float32. phase_mask is
        1.0 inside an annotated cycle, 0.0 outside -- used as the 0/1
        classification target for loss.InRepetitionLoss.
    """
    idx = torch.as_tensor(list(frame_indices), dtype=torch.float32)
    n = idx.shape[0]
    phase_sin  = torch.zeros(n)
    phase_cos  = torch.zeros(n)
    phase_mask = torch.zeros(n)

    for s, e in zip(cycle_starts, cycle_ends):
        if e <= s:
            continue
        inside = (idx >= s) & (idx <= e)
        theta  = 2 * math.pi * (idx - s) / (e - s)
        phase_sin  = torch.where(inside, torch.sin(theta), phase_sin)
        phase_cos  = torch.where(inside, torch.cos(theta), phase_cos)
        phase_mask = torch.where(inside, torch.ones(n), phase_mask)

    return phase_sin, phase_cos, phase_mask


# ── training loss ──────────────────────────────────────────────────────────────

def repetition_loss(
    pred_density: torch.Tensor,
    gt_density:   torch.Tensor,
    gt_count:     torch.Tensor,
    alpha:        float = 1,
) -> tuple[torch.Tensor, dict]:
    """
    Two-component density regression loss: MSE (per-frame values) + SmoothL1 (count).

    Args:
        pred_density : (B, T) predicted density (may be negative -- no output
                       activation on the current DensityHead)
        gt_density   : (B, T) target density map, sums to gt_count, non-negative
        gt_count     : (B,)   total count ground truth
        alpha        : weight for count regression term (kept moderate; MSE
                       already carries most of the count signal since
                       count = sum(density))

    Returns:
        loss   : scalar
        losses : dict of component values for logging
    """
    mse_loss = F.mse_loss(pred_density, gt_density)

    pred_count = pred_density.sum(dim=-1)
    count_loss = torch.sum(torch.div(torch.abs(pred_count - gt_count), gt_count + 1e-1)) / \
                            pred_count.flatten().shape[0] # mae

    total = mse_loss + alpha * count_loss

    return total, {
        "mse":   mse_loss.item(),
        "count": count_loss.item(),
        "total": total.item(),
    }