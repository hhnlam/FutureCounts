"""
Losses for density-map repetition counting.

    loss = MSE(pred_density, gt_density) + lambda_count * MAE(pred_count, gt_count)

where pred_count = pred_density.sum(-1). The density term shapes the per-frame
curve; the count term keeps the integral (the actual prediction) accurate.

An optional second density-side term penalizes sign/support mismatches the
MSE doesn't explicitly target -- predicting density where there should be
none, or none where there should be some:

    loss += lambda_support * DensitySupportLoss(pred_density, gt_density)

A third, also purely auxiliary and independent term classifies whether each
frame is currently inside a repetition at all, reusing phase_mask itself as
the 0/1 target (see model.InRepetitionHead):

    loss += lambda_in_repetition * InRepetitionLoss(pred_in_rep, phase_mask)

A fourth, also purely auxiliary term encourages the shared temporal
bottleneck to be predictive of the frozen encoder's OWN future embeddings
(see model.FuturePredictionHead):

    loss += lambda_future * FuturePredictionLoss(pred_future, encoder_feats)

A fifth, also purely auxiliary term hinge-penalizes FALSE-POSITIVE counts
on windows whose ground-truth count is (near) zero -- CountLoss treats those
examples like any other, so this term exists to stamp out that specific
failure mode directly:

    loss += lambda_zero * ZeroCountLoss(pred_count, gt_count)

A sixth, also purely auxiliary term regresses the count DIRECTLY (no
reference-length rescaling) from a small pooled/per-frame MLP (see
model.CountHead) -- follows up on the P-vs-2P periodicity investigation,
testing whether the shared bottleneck already carries enough signal to
recover the count without going through the unsupervised density map:

    loss += lambda_count_head * CountHeadLoss(pred_count_head, gt_count)

Modules (all return (scalar_loss, components_dict) so training can log parts):
    DensityLoss           : MSE(pred_density, gt_density)
    DensitySupportLoss    : symmetric hinge penalizing pred_density>0 where gt==0
                            and pred_density<=0 where gt>0
    CountLoss             : MAE(pred_count,   gt_count)   [relative, epsilon-stabilised]
    InRepetitionLoss      : BCE(pred_in_rep, phase_mask) -- in-repetition classifier;
                            'focal' (default) | 'balanced' | 'bce' (see mode)
    FuturePredictionLoss  : 1 - cosine_similarity(pred_embedding_{t}, encoder_embedding_{t+k}),
                            averaged over valid t (see class docstring for the shift)
    ZeroCountLoss         : relu(pred_count - zero_tolerance)^2, averaged over just the
                            examples with gt_count<=zero_count_threshold -- false-positive
                            hinge penalty on true zero-count windows
    CountHeadLoss         : smooth_l1 (default) | l1 | mse between pred_count_head and
                            gt_count directly -- no reference-length rescaling
    CombinedLoss          : DensityLoss + lambda_count * CountLoss + lambda_support * DensitySupportLoss
                            [+ lambda_in_repetition * InRepetitionLoss, only when use_in_repetition=True]
                            [+ lambda_future * FuturePredictionLoss, only when use_future_pred=True]
                            [+ lambda_zero * ZeroCountLoss, only when use_zero_penalty=True]
                            [+ lambda_count_head * CountHeadLoss, only when use_count_head=True]
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityLoss(nn.Module):
    """Per-frame density MSE. pred/gt: (B, T)."""

    def forward(self, pred_density: torch.Tensor, gt_density: torch.Tensor, alpha=50) -> torch.Tensor:
        
        density_error = pred_density - gt_density

        weight = 1 + alpha*gt_density

        density_loss = (
            weight * density_error**2
        ).mean()

        # return F.mse_loss(pred_density, gt_density)
        return density_loss


class DensitySupportLoss(nn.Module):
    """
    Penalizes support/sign mismatches DensityLoss's MSE doesn't explicitly
    target: pred_density > 0 where gt_density == 0 (a phantom prediction),
    and pred_density <= 0 where gt_density > 0 (a missed prediction).
    Symmetric hinge -- penalty scales with how far pred_density sits on the
    wrong side of zero; zero once it's on the correct side (no reward for
    being "more correct" than necessary). pred/gt: (B, T).
    """

    def forward(self, pred_density: torch.Tensor, gt_density: torch.Tensor) -> torch.Tensor:
        is_zero = gt_density == 0
        is_pos  = gt_density > 0
        false_pos = F.relu(pred_density) * is_zero.float()
        false_neg = F.relu(-pred_density) * is_pos.float()
        return (false_pos + false_neg).mean()


class CountLoss(nn.Module):
    """
    Count regression on the density integral.

    mode='relative' (default): mean |pred-gt| / (gt+eps) -- matches the parent
        project's repetition_loss and normalises across the wide count range so
        high-count videos don't dominate. mode='l1': plain mean |pred-gt|.
    mode='fast_rep_weighted': plain L1, per-sample UPWEIGHTED when repetitions
        are "fast" relative to the clip -- few temporal feature steps
        available to resolve each one:

            steps_per_rep = clip_length / gt_count

        clip_length is the fixed, padded length every FrameClipDataset clip
        has (an approximation of the real per-repetition frame budget, not
        the true non-padded frame count -- see dataset.FrameClipDataset).
        Fewer steps/rep -> harder for the TCN's fixed temporal receptive
        field to resolve -> upweighted (checked in order, so e.g.
        steps_per_rep=5 gets 2.00x, not 1.25x*1.50x*2.00x stacked):

            steps_per_rep <= 20 -> 1.25x
            steps_per_rep <= 12 -> 1.50x
            steps_per_rep <=  8 -> 2.00x
            otherwise           -> 1.00x

        High-gt_count samples have the fewest steps/rep in a fixed-length
        clip, so this directly targets the "count head squashed toward the
        pool's low/mid-count mean" failure mode (see overfit_subset.py's
        gradient-similarity diagnostic). Requires `clip_length` at
        construction time.
    """

    def __init__(self, mode: str = "relative", eps: float = 0.1, clip_length: Optional[int] = None):
        super().__init__()
        if mode not in ("relative", "l1", "fast_rep_weighted", "l2", "mixl2", "countix", "repcount"):
            raise ValueError("mode must be 'relative', 'l1', or 'fast_rep_weighted'.")
        if mode == "fast_rep_weighted" and not clip_length:
            raise ValueError("mode='fast_rep_weighted' requires clip_length (frames per padded clip).")
        self.mode = mode
        self.eps  = eps
        self.clip_length = clip_length

    def forward(self, pred_count: torch.Tensor, gt_count: torch.Tensor) -> torch.Tensor:
        abs_err = (pred_count - gt_count).abs()
        if self.mode == "relative":
            return (abs_err / (gt_count + self.eps)).mean()
        if self.mode == "fast_rep_weighted":
            steps_per_rep = self.clip_length / gt_count.float().clamp_min(1.0)
            weights = torch.ones_like(abs_err)
            weights = torch.where(steps_per_rep <= 20, 1.25, weights)
            weights = torch.where(steps_per_rep <= 12, 1.50, weights)
            weights = torch.where(steps_per_rep <= 8, 2.00, weights)

            mse = abs_err.square()

            count_loss = abs_err + 0.02 * mse
            return (weights * count_loss).mean()
        if self.mode == 'l2':
            return (abs_err.square()).mean()
        if self.mode == 'countix':
            l2 = abs_err.square()
            relative = abs_err / (gt_count + self.eps)
            # return (0.5*l2 + 1*abs_err).mean()
            return (l2 + 0.7*relative).mean()
        if self.mode == 'repcount':
                    l2 = abs_err.square()
                    relative = abs_err / (gt_count + self.eps)
                    return (0.5*l2 + 1*abs_err).mean()
                    # return (l2 + 0.5*abs_err).mean()
        return abs_err.mean()

    def weight(self, gt_count):
        return 1 + (gt_count / 10)  # adjust divisor to tune


class InRepetitionLoss(nn.Module):
    """
    Auxiliary per-frame binary-classification loss for model.InRepetitionHead:
    is this frame inside a repetition or not? Target is density.
    make_phase_targets' own phase_mask (1.0 = inside an annotated cycle, 0.0
    = not) -- the SAME target MaskedPhaseLoss uses as a mask, reused here
    directly as the classification label instead.

    Repetitions typically cover a minority of frames, so a plain BCE over
    every position is dominated by the (easy, majority) negative class.
    `mode` selects how this is addressed:

      'bce'      -- plain, unweighted BCEWithLogits mean over all frames.
                    No class balancing at all (the original baseline).
      'balanced' -- per-frame BCE is averaged separately within the positive
                    and negative subsets (across the whole flattened batch),
                    then combined alpha/(1-alpha) -- so each class
                    contributes to the loss according to `alpha` regardless
                    of how imbalanced the frame counts are. Falls back to
                    the plain BCE mean when a batch has no positives or no
                    negatives at all (nothing to balance against).
      'focal'    -- (default) binary focal loss (Lin et al., 2017): each
                    per-frame BCE term is scaled by
                    alpha_t * (1 - p_t)^gamma, where p_t is the model's
                    predicted probability for the TRUE class. Easy,
                    already-well-classified frames (p_t close to 1) get
                    down-weighted toward zero, so the loss automatically
                    concentrates on hard/misclassified frames instead of
                    being swamped by the easy majority negatives; `alpha`
                    additionally reweights the positive class (repetition
                    frames) up and the negative class down, same as
                    'balanced'. `gamma` is ignored by the other two modes.

    forward(pred_in_rep, gt_in_rep) -> scalar
        pred_in_rep: (B, T) raw logit (InRepetitionHead has no output
            activation -- use binary_cross_entropy_with_logits, not a
            separate sigmoid + BCE, for numerical stability).
        gt_in_rep: (B, T) in {0, 1} (phase_mask).
    """

    def __init__(self, mode: str = "focal", alpha: float = 0.7, gamma: float = 2.0):
        super().__init__()
        if mode not in ("focal", "balanced", "bce"):
            raise ValueError("mode must be 'focal', 'balanced', or 'bce'.")
        self.mode = mode
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_in_rep: torch.Tensor, gt_in_rep: torch.Tensor) -> torch.Tensor:
        inside_logits = pred_in_rep.reshape(-1)
        inside_targets = gt_in_rep.reshape(-1).float()

        bce = F.binary_cross_entropy_with_logits(
            inside_logits,
            inside_targets,
            reduction="none",
        )

        if self.mode == "bce":
            return bce.mean()

        if self.mode == "balanced":
            positive_mask = inside_targets == 1
            negative_mask = inside_targets == 0
            if not positive_mask.any() or not negative_mask.any():
                return bce.mean()
            positive_loss = bce[positive_mask].mean()
            negative_loss = bce[negative_mask].mean()
            return self.alpha * positive_loss + (1 - self.alpha) * negative_loss

        p = torch.sigmoid(inside_logits)
        p_t = p * inside_targets + (1 - p) * (1 - inside_targets)
        alpha_t = self.alpha * inside_targets + (1 - self.alpha) * (1 - inside_targets)

        focal_loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        return focal_loss.mean()


class FuturePredictionLoss(nn.Module):
    """
    Auxiliary future-embedding prediction loss for the shared temporal
    bottleneck (see model.FuturePredictionHead). NOT sequence generation, NOT
    autoregressive, NOT reconstruction -- just a per-timestep regression that
    shapes the upstream temporal aggregator.

    Given the encoder's own embeddings e_1..e_T and the head's predictions
    pred_1..pred_T, the shift is applied internally:

        pred[:, :-horizon]  vs  target[:, horizon:]

    and the last `horizon` predictions (which have no future target inside
    the clip) are dropped.

    residual=False (default): pred_t is trained to approximate the ABSOLUTE
    future embedding e_{t+horizon} directly -- target_{t+horizon}.

    residual=True: pred_t is instead trained to approximate the RESIDUAL
    change Delta z_t = e_{t+horizon} - e_t -- i.e. FuturePredictionHead
    predicts a delta to add to the current embedding rather than the future
    embedding itself. This reframes the regression target as motion in
    latent space, which is typically smaller-magnitude and more stationary
    than the absolute embedding, so it can be an easier quantity to predict.
    The head's raw output shape/semantics don't change -- only what it's
    supervised to approximate.

    For each remaining valid timestep, mode='cosine' (default) computes:

        loss_t = 1 - cosine_similarity(pred_t, target_t)

    mode='smooth_l1' instead computes elementwise Huber/smooth-L1 loss
    (see `smooth_l1_beta`) between pred_t and target_t, averaged over the
    embedding dimension (target_t is e_{t+horizon} or Delta z_t depending on
    `residual`). Both modes are then averaged over all valid (batch,
    timestep) pairs.

    forward(pred_embeddings, target_embeddings) -> scalar
        Both args: (B, T, D). target_embeddings is always the encoder's own
        (unshifted) embeddings -- the horizon shift (and, when residual=True,
        the e_{t+horizon} - e_t subtraction) both happen internally. Returns
        0 (still graph-connected, so .backward() is safe) if T <= horizon (no
        valid timestep to compare).
    """

    def __init__(self, horizon: int = 4, mode: str = "cosine", smooth_l1_beta: float = 1.0,
                 residual: bool = False):
        super().__init__()
        if mode not in ("cosine", "smooth_l1"):
            raise ValueError("mode must be 'cosine' or 'smooth_l1'.")
        self.horizon = horizon
        self.mode = mode
        self.smooth_l1_beta = smooth_l1_beta
        self.residual = residual

    def _shift(self, pred_embeddings: torch.Tensor, target_embeddings: torch.Tensor):
        """
        Shared by forward() and zero_baseline(): applies the horizon shift
        (pred[:, :-k] vs target[:, k:]) and, when residual=True, turns the
        target into Delta z_t = e_{t+horizon} - e_t. Returns (None, None) if
        T <= horizon (no valid timestep).
        """
        k = self.horizon
        T = pred_embeddings.shape[1]
        if T <= k:
            return None, None
        pred   = pred_embeddings[:, :-k]     # (B, T-k, D)
        target = target_embeddings[:, k:]    # (B, T-k, D), e_{t+horizon}
        if self.residual:
            target = target - target_embeddings[:, :-k]   # Delta z_t = e_{t+horizon} - e_t
        return pred, target

    def forward(self, pred_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
        pred, target = self._shift(pred_embeddings, target_embeddings)
        if pred is None:
            return pred_embeddings.sum() * 0.0
        if self.mode == "smooth_l1":
            return F.smooth_l1_loss(pred, target, beta=self.smooth_l1_beta)
        cos_sim = F.cosine_similarity(pred, target, dim=-1)  # (B, T-k)
        return (1.0 - cos_sim).mean()

    @torch.no_grad()
    def zero_baseline(self, pred_embeddings: torch.Tensor, target_embeddings: torch.Tensor,
                       beta: float = 0.05):
        """
        Is the head learning anything beyond predicting zero?

        Residual targets (residual=True) are typically small and centred near
        zero (see analyze_future_residual.py), so a trivial all-zeros
        prediction can already score a deceptively low loss -- a shrinking
        `future` loss component doesn't by itself mean the head learned
        useful motion; it may just have collapsed toward outputting ~0. This
        compares the head's actual prediction against that zero baseline
        directly, on the SAME (shifted, residual-if-applicable) target
        forward() trains against.

        Deliberately always smooth-L1 at a fixed beta (not this module's own
        `mode`/`smooth_l1_beta`) regardless of what mode training uses:
        cosine similarity of a near-zero vector is degenerate and won't
        surface a magnitude collapse the way an elementwise regression loss
        will, so this is a fixed, mode-independent magnitude check.

        Returns (zero_loss, model_loss, improvement) as plain floats, where
        improvement = 1 - model_loss / zero_loss:
            ~0 or negative -> the head is (close to) a trivial zero predictor;
                               turning up lambda_future will not fix that.
            closer to 1     -> the head is capturing real predictive signal.
        Returns None if T <= horizon (no valid timestep).
        """
        pred, target = self._shift(pred_embeddings, target_embeddings)
        if pred is None:
            return None
        zero_pred  = torch.zeros_like(target)
        zero_loss  = F.smooth_l1_loss(zero_pred, target, beta=beta).item()
        model_loss = F.smooth_l1_loss(pred, target, beta=beta).item()
        improvement = 1.0 - model_loss / zero_loss if zero_loss > 0 else float("nan")
        return zero_loss, model_loss, improvement


class ZeroCountLoss(nn.Module):
    """
    Auxiliary hinge penalty targeted specifically at ZERO-count windows:
    penalizes pred_count for sitting above `zero_tolerance` on examples whose
    ground-truth count is (near) zero, and is exactly 0 for every other
    example -- a dedicated false-positive penalty, distinct from CountLoss's
    per-example MAE/relative-error. CountLoss doesn't single these out:
    count_mode='relative' divides by (gt_count + eps), so on a true zero-count
    example even a small absolute overcount is inflated into a large relative
    error (already implicitly punished, but entangled with every other
    example's relative scaling); count_mode='l1' gives zero-count examples no
    special weight at all. This term exists to stamp out that specific
    failure mode (predicting repetitions on a clip that has none) directly,
    independent of `count_mode`.

    zero_mask = gt_count <= zero_count_threshold (default 0.0 -- exact
        zero-count windows only; raise it slightly to also cover
        near-zero/noisy annotations).
    loss = relu(pred_count - zero_tolerance)^2, averaged over just the masked
        (zero-count) examples -- squared (not linear) hinge so larger
        false-positive overcounts are penalized more than proportionally.
        `zero_tolerance` is the amount of overcount allowed before any
        penalty kicks in (0 would penalize even tiny numerical noise around
        a perfect 0 prediction).
    Returns 0 (still graph-connected, so .backward() is safe) if the batch
        has no zero-count examples at all.

    forward(pred_count, gt_count) -> scalar
        pred_count, gt_count: (B,)
    """

    def __init__(self, zero_tolerance: float = 0.1, zero_count_threshold: float = 0.0):
        super().__init__()
        self.zero_tolerance = zero_tolerance
        self.zero_count_threshold = zero_count_threshold

    def forward(self, pred_count: torch.Tensor, gt_count: torch.Tensor) -> torch.Tensor:
        zero_mask = gt_count <= self.zero_count_threshold
        if zero_mask.any():
            return F.relu(pred_count[zero_mask] - self.zero_tolerance).square().mean()
        return pred_count.sum() * 0.0


class CountHeadLoss(nn.Module):
    """
    Auxiliary count-regression loss for model.CountHead.

    The target is gt_count itself, regardless of the clip's T. CountHead exists to test
    directly whether the shared bottleneck (pooled over T, or per-frame then
    summed -- see CountHead's `mode`) already carries enough signal to
    regress the count with a small MLP, independent of the unsupervised
    per-frame density map.

    mode='smooth_l1' (default): Huber loss between pred_count and gt_count
    (see `beta`). mode='l1': plain MAE. mode='mse': plain MSE.

    forward(pred_count_head, gt_count) -> scalar
        pred_count_head: (B,) CountHead's raw prediction.
        gt_count:        (B,) ground-truth repetition count for the clip.
    """

    def __init__(self, mode: str = "smooth_l1", beta: float = 1.0, eps=0.1):
        super().__init__()
        if mode not in ("smooth_l1", "l1", "mse", "relative"):
            raise ValueError("mode must be 'smooth_l1', 'l1', or 'mse'.")
        self.mode = mode
        self.beta = beta
        self.eps = eps

    def forward(self, pred_count_head: torch.Tensor, gt_count: torch.Tensor) -> torch.Tensor:
        if self.mode == "smooth_l1":
            return F.smooth_l1_loss(pred_count_head, gt_count, beta=self.beta)
        if self.mode == "l1":
            return F.l1_loss(pred_count_head, gt_count)
        if self.mode == "relative":
            abs_err = (pred_count_head - gt_count).abs()
            return (abs_err / (gt_count + self.eps)).mean()
        return F.mse_loss(pred_count_head, gt_count)


class CombinedLoss(nn.Module):
    """
    L = DensityLoss + lambda_count * CountLoss + lambda_support * DensitySupportLoss
        [+ lambda_in_repetition * InRepetitionLoss, only when use_in_repetition=True]
        [+ lambda_future * FuturePredictionLoss, only when use_future_pred=True]
        [+ lambda_zero * ZeroCountLoss, only when use_zero_penalty=True]
        [+ lambda_count_head * CountHeadLoss, only when use_count_head=True]

    forward(pred_density, gt_density, gt_count, pred_count=None,
            phase_mask=None, pred_in_rep=None, pred_future=None,
            target_future=None, pred_count_head=None)
        -> (loss, components)
        `components` holds float (.item()'d, no grad) values for logging under
        "density"/"count"/"support"/"phase"/"in_repetition"/"future"/"outside"/
        "zero"/"rate"/"total", PLUS "count_term" and (when the future head is active)
        "future_term" --
        the weighted (lambda_count * count_loss, lambda_future * future_loss)
        tensors STILL graph-connected, for callers that want each term's own
        contribution to the gradient (e.g. train.py's per-term grad-norm
        logging: `components["count_term"].backward(retain_graph=True)`).
        pred_density : (B, T) or None. None when the density head is disabled
                       (RepetitionCounter(use_density=False), phase-only
                       inference) -- the density term is then skipped
                       entirely (omitted from `components`, not zeroed).
                       Otherwise: predicted density (may be negative -- no
                       output activation on DensityHead).
        gt_density   : (B, T)   target density, non-negative, sums to gt_count.
                       Still required even when pred_density is None (only
                       used if pred_density is given).
        gt_count     : (B,)     ground-truth count
        pred_count   : (B,) predicted count, e.g. from model.unpack_output()
                       (which is the model's own count, however it was
                       produced -- density.sum() or CountHead's own prediction).
                       REQUIRED when pred_density is None (nothing else to
                       derive a count from). If omitted while pred_density is
                       given, falls back to pred_density.sum(-1) -- the
                       original behaviour.
        pred_in_rep  : (B, T) raw logit, from RepetitionCounter's
                       InRepetitionHead (see model.unpack_in_repetition) --
                       ignored unless use_in_repetition=True. Trained against
                       phase_mask itself (reused directly as the 0/1
                       classification target, see InRepetitionLoss).
        phase_mask   : (B, T), from dataset.py / density.make_phase_targets --
                       the in-repetition classification target (see
                       pred_in_rep above).
        pred_future  : (B, T, D), from RepetitionCounter's FuturePredictionHead
                       (see model.unpack_future) -- ignored unless
                       use_future_pred=True. Interpreted as an absolute
                       future embedding or a residual Delta z depending on
                       `future_residual` (see FuturePredictionLoss).
        target_future: (B, T, D), the frozen encoder's own embeddings
                       (model.unpack_future's `encoder_feats`) -- the
                       prediction target, shifted (and, when
                       future_residual=True, differenced) internally by
                       FuturePredictionLoss.
        pred_count_head : (B,), from RepetitionCounter's CountHead (see
                       model.unpack_count_head) -- ignored unless
                       use_count_head=True. Regressed directly against
                       gt_count.

    The density head, when present, is untouched by any of this -- in-
    repetition, future-prediction, and count_head are purely additive. When
    use_future_pred=False and use_count_head=False (defaults) and
    pred_density is given, behaviour and the `components` dict are IDENTICAL
    to before future/count_head/density-optional support existed.
    """

    def __init__(
        self,
        lambda_count:   float = 5.0,
        lambda_density: float = 0,
        lambda_support: float = 0.0,
        count_mode:     str = "relative",
        count_eps:      float = 0.1,
        clip_length:    Optional[int] = None,
        use_in_repetition: bool = False,
        lambda_in_repetition: float = 1.0,
        in_rep_loss_mode: str = "focal",
        in_rep_focal_alpha: float = 0.7,
        in_rep_focal_gamma: float = 2.0,
        use_future_pred: bool = False,
        lambda_future:  float = 10,
        future_horizon: int = 12,
        future_mode:    str = "smooth_l1",
        future_smooth_l1_beta: float = 0.05,
        future_residual: bool = True,
        use_zero_penalty: bool = False,
        lambda_zero:    float = 0.5,
        zero_tolerance: float = 0.1,
        zero_count_threshold: float = 0.0,
        use_count_head: bool = False,
        lambda_count_head: float = 1.0,
        count_head_loss_mode: str = "smooth_l1",
        count_head_loss_beta: float = 1.0,
    ):
        super().__init__()
        self.density_loss  = DensityLoss()
        self.support_loss  = DensitySupportLoss()
        self.count_loss    = CountLoss(mode=count_mode, eps=count_eps, clip_length=clip_length)
        self.in_rep_loss   = (
            InRepetitionLoss(mode=in_rep_loss_mode, alpha=in_rep_focal_alpha, gamma=in_rep_focal_gamma)
            if use_in_repetition else None
        )
        self.future_loss   = (
            FuturePredictionLoss(horizon=future_horizon, mode=future_mode,
                                  smooth_l1_beta=future_smooth_l1_beta,
                                  residual=future_residual)
            if use_future_pred else None
        )
        self.zero_loss     = (
            ZeroCountLoss(zero_tolerance=zero_tolerance, zero_count_threshold=zero_count_threshold)
            if use_zero_penalty else None
        )
        self.count_head_loss = (
            CountHeadLoss(mode=count_head_loss_mode, beta=count_head_loss_beta)
            if use_count_head else None
        )
        self.lambda_count_head = lambda_count_head
        self.lambda_count  = lambda_count
        self.lambda_density = lambda_density
        self.lambda_support = lambda_support
        self.lambda_in_repetition = lambda_in_repetition
        self.lambda_future = lambda_future
        self.lambda_zero   = lambda_zero

    def forward(
        self,
        pred_density:  Optional[torch.Tensor],
        gt_density:    torch.Tensor,
        gt_count:      torch.Tensor,
        pred_count:    Optional[torch.Tensor] = None,
        phase_mask:    Optional[torch.Tensor] = None,
        pred_in_rep:   Optional[torch.Tensor] = None,
        pred_future:   Optional[torch.Tensor] = None,
        target_future: Optional[torch.Tensor] = None,
        pred_count_head: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        components: dict = {}
        total: Optional[torch.Tensor] = None

        if pred_density is not None:
            d_loss = self.density_loss(pred_density, gt_density)
            components["density"] = d_loss.item()
            total = self.lambda_density * d_loss

            s_loss = self.support_loss(pred_density, gt_density)
            components["support"] = s_loss.item()
            total = total + self.lambda_support * s_loss

            if pred_count is None:
                pred_count = pred_density.sum(dim=-1)
        elif pred_count is None:
            raise ValueError(
                "CombinedLoss needs pred_count when pred_density is None -- the "
                "density head is disabled, so pass the model's own count "
                "(e.g. unpack_output(model_out)[1]) explicitly."
            )

        c_loss = self.count_loss(pred_count, gt_count)
        components["count"] = c_loss.item()
        weighted_count = self.lambda_count * c_loss
        total = (total + weighted_count) if total is not None else weighted_count
        # weighted, graph-connected (not .item()'d) -- callers (e.g. train.py) can
        # .backward(retain_graph=True) on these individually to measure each term's
        # own contribution to the gradient, separately from the float `components`
        # values above (which are for logging/aggregation only, no grad).
        components["count_term"] = weighted_count

        if self.in_rep_loss is not None and pred_in_rep is not None and phase_mask is not None:
            # phase_mask reused directly as the 0/1 classification target --
            # see InRepetitionLoss docstring.
            ir_loss = self.in_rep_loss(pred_in_rep, phase_mask)
            total = total + self.lambda_in_repetition * ir_loss
            components["in_repetition"] = ir_loss.item()

        if self.future_loss is not None and pred_future is not None and target_future is not None:
            f_loss = self.future_loss(pred_future, target_future)
            weighted_future = self.lambda_future * f_loss
            total = total + weighted_future
            components["future"] = f_loss.item()
            components["future_term"] = weighted_future

        if self.zero_loss is not None:
            z_loss = self.zero_loss(pred_count, gt_count)
            total = total + self.lambda_zero * z_loss
            components["zero"] = z_loss.item()

        if self.count_head_loss is not None and pred_count_head is not None:
            ch_loss = self.count_head_loss(pred_count_head, gt_count)
            total = total + self.lambda_count_head * ch_loss
            components["count_head"] = ch_loss.item()

        components["total"] = total.item()
        return total, components
