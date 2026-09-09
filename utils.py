"""
Shared utilities: seeds, metrics, checkpoints, config loading, sliding-window
inference, and density visualisation.
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Sampler

from model import unpack_output, unpack_in_repetition, unpack_count_head


# ── reproducibility ─────────────────────────────────────────────────────────────

def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python / numpy / torch (all devices). deterministic=True is slower."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker_init_fn for reproducible augmentation across workers.

    Also caps each worker to a single intra-op thread: worker __getitem__ work
    (PIL decode/resize, building a density map for one sample) is inherently
    serial and doesn't benefit from torch's thread pool, so on many-core boxes
    leaving the default (one thread per core, per worker) badly oversubscribes
    the CPU across num_workers processes.
    """
    torch.set_num_threads(1)
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ── config ───────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load a YAML config into a plain dict."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_config(config: dict, path: str) -> None:
    """Dump a config dict to YAML, e.g. alongside eval/train outputs for provenance."""
    import yaml
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def merge_overrides(config: dict, overrides: dict) -> dict:
    """Override config values with any non-None entries in `overrides` (CLI wins)."""
    out = dict(config)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out


# ── metrics ──────────────────────────────────────────────────────────────────────

def compute_metrics(gts, preds, alpha: float = 0.1) -> dict:
    """
    Standard repetition-counting metrics.

        MAE      : mean |pred-gt| / (gt+alpha)   (alpha=0 RepNet, alpha=0.1 TransRAC)
        RMSE     : sqrt(mean (pred-gt)^2)
        OBO      : fraction with |pred-gt| <= 1  (off-by-one accuracy)
        OBO10    : fraction with |pred-gt| <= 0.10 * max(gt, eps)  (off-by-10% accuracy --
                   relative-error tolerance, unlike OBO's fixed absolute margin of 1.
                   Uses a fixed tiny eps, independent of `alpha` above, purely to keep
                   the gt=0 case finite -- gt=0 then effectively requires pred≈0.)
        Accuracy : fraction with round(pred)==round(gt)
    """
    gts   = np.asarray(gts, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    abs_err = np.abs(preds - gts)
    return {
        "n":        int(len(gts)),
        "MAE":      float(np.mean(abs_err / (gts + alpha))) if len(gts) else 0.0,
        "RMSE":     float(np.sqrt(np.mean((preds - gts) ** 2))) if len(gts) else 0.0,
        "OBO":      float(np.mean(abs_err <= 1)) if len(gts) else 0.0,
        "OBO10":    float(np.mean(abs_err <= np.maximum(1, 0.10 * np.maximum(gts, 1e-6)))) if len(gts) else 0.0,
        "Accuracy": float(np.mean(np.round(preds) == np.round(gts))) if len(gts) else 0.0,
    }


def compute_in_repetition_metrics(logits: torch.Tensor, targets: torch.Tensor,
                                   threshold: float = 0.5) -> dict:
    """
    Binary classification metrics for the auxiliary InRepetitionHead: is this
    frame inside a repetition or not? logits/targets: any shape, flattened
    together -- pass the concatenation across the whole validation set (a
    single video can easily have zero frames of one class, which would make
    precision/recall/specificity undefined for it alone).

        positive_ratio: fraction of GT frames actually inside a repetition
            (class balance -- precision/recall only mean something next to
            this: e.g. 90% recall is unremarkable if 90% of frames are
            positive).
        precision:      of frames predicted "inside", fraction actually inside.
        recall:         of frames actually inside, fraction predicted so.
        specificity:    of frames actually outside, fraction predicted so.

    logits are raw (pre-sigmoid); a sigmoid + threshold is applied here.
    """
    logits  = logits.reshape(-1)
    targets = targets.reshape(-1)

    positive_ratio = targets.float().mean()
    prob = torch.sigmoid(logits)
    pred = (prob > threshold).float()

    tp = ((pred == 1) & (targets == 1)).sum()
    tn = ((pred == 0) & (targets == 0)).sum()
    fp = ((pred == 1) & (targets == 0)).sum()
    fn = ((pred == 0) & (targets == 1)).sum()

    precision   = tp / (tp + fp).clamp_min(1)
    recall      = tp / (tp + fn).clamp_min(1)
    specificity = tn / (tn + fp).clamp_min(1)

    return {
        "positive_ratio": float(positive_ratio.item()),
        "precision":      float(precision.item()),
        "recall":         float(recall.item()),
        "specificity":    float(specificity.item()),
    }


# ── sliding-window inference ─────────────────────────────────────────────────────

def compute_window_starts(N: int, clip_length: int, stride: int) -> list[int]:
    """
    Sliding-window start positions tiling [0, N) with windows of length
    `clip_length` spaced `stride` apart, in whatever index space `N` is
    expressed in (raw frames or temporal-stride-subsampled frames -- the
    caller decides). Always guarantees the tail is covered: the last window
    starts at max(0, N - clip_length), even when that isn't a multiple of
    `stride`. Shared by sliding_window_inference (eval) and
    sliding_window_train_count (the full-video training path) so the two
    can't drift apart.
    """
    if N <= 0:
        return [0]
    starts = list(range(0, N, stride))
    tail = max(0, N - clip_length)
    if starts[-1] != tail:
        starts.append(tail)
    return starts


@torch.no_grad()
@torch.no_grad()
def _count_head_tiled_sum(encoder, model, frames: torch.Tensor, clip_length: int,
                          device: torch.device, use_amp: bool) -> Optional[torch.Tensor]:
    """
    Whole-video count purely from CountHead: tiles `frames` into NON-
    overlapping clip_length windows (batched into one forward pass) and sums
    each window's predicted count directly.

    CountHead predicts an already-integrated per-clip count, not a per-frame
    rate like density -- overlapping windows would double-count a repetition
    that falls in the overlap, so unlike density/phase's overlap-and-average
    reconstruction, this ALWAYS uses non-overlapping tiling, independent of
    whatever window_stride the caller is using for the density/phase
    ensembling in sliding_window_inference above.

    Returns None if the model has no count_head.
    """
    N = frames.shape[0]
    starts = compute_window_starts(N, clip_length, clip_length)
    windows = []
    for start in starts:
        valid = min(clip_length, N - start)
        window = frames[start:start + valid]
        if valid < clip_length:
            pad = window[-1:].expand(clip_length - valid, *window.shape[1:])
            window = torch.cat([window, pad], dim=0)
        windows.append(window)
    batch = torch.stack(windows, dim=0).to(device, non_blocking=True)  # (W,L,3,H,W)

    with torch.autocast(device_type=device.type, enabled=use_amp):
        out = model.forward_from_features(encoder(batch))
    count_head_out = unpack_count_head(out)
    if count_head_out is None:
        return None
    return count_head_out.float().cpu().sum()


def _stitch_median(windows: list, N_out: int, extra_shape: tuple = ()) -> torch.Tensor:
    """
    Combine per-window contributions onto one length-N_out timeline by taking
    the MEDIAN, at each position, over every window that covers it -- the
    aggregation="median" counterpart to sliding_window_inference's default
    scatter-add-then-mean. `windows` is a list of (token_start, token_valid,
    tensor) tuples, tensor shaped (token_valid, *extra_shape). Positions
    outside every window's span are excluded via NaN-masking (compute_window_starts
    guarantees every position in [0, N_out) is covered by at least one window,
    so no position ends up all-NaN). Returns an all-zero tensor if `windows`
    is empty (mirrors the mean path's all-zero fallback when a head was
    requested but never actually present in any window's output).
    """
    if not windows:
        return torch.zeros(N_out, *extra_shape, dtype=torch.float32)
    stacked = torch.full((len(windows), N_out, *extra_shape), float("nan"), dtype=torch.float32)
    for i, (start, valid, values) in enumerate(windows):
        stacked[i, start:start + valid] = values
    # nanquantile(0.5), NOT nanmedian: torch.nanmedian returns the LOWER of
    # the two middle elements on an even-length input rather than averaging
    # them -- for positions covered by exactly 2 windows (the common case
    # right at typical overlap factors), that would deterministically pick
    # the smaller of the two predictions every time, silently reintroducing
    # a systematic downward (undercounting) bias -- the opposite of what
    # this aggregation mode is for. nanquantile interpolates instead,
    # matching the usual (numpy-style) median definition.
    return torch.nanquantile(stacked, 0.5, dim=0)


@torch.no_grad()
def sliding_window_inference(
    encoder,
    model,
    frames: torch.Tensor,
    clip_length: int,
    window_stride: int,
    device: torch.device,
    use_amp: bool = False,
    return_in_rep: bool = False,
    return_count_head: bool = False,
    temporal_downsample: int = 1,
    aggregation: str = "mean",
):
    """
    Reconstruct a full-video density map (and, optionally, in-repetition
    prediction and/or a CountHead-based count) from overlapping clip
    predictions.

    frames: (N, 3, H, W) normalised, on CPU or GPU.
    For each window [start, start+clip_length):
        encode -> model -> per-frame density (clip_length,)
                            [+ in_rep logit (clip_length,)]
    The valid portion is scatter-added onto a length-N accumulator, and
    overlapping positions are combined per `aggregation`. The in-repetition
    logit is combined the same way (then a sigmoid gives a per-frame
    probability).

    aggregation: how overlapping windows' predictions at the SAME raw-frame
        position get combined into one value.
        "mean" (default): arithmetic mean -- IDENTICAL to this function's
            behaviour before this parameter existed.
        "median": per-position median across every window that covers that
            position, instead of the mean. Motivation: when nearby windows
            disagree about exactly where a peak sits (more likely when
            repetitions are packed densely relative to clip_length -- see
            Countix vs RepCount steps_per_rep analysis), the mean blends the
            disagreement into a single wider/lower blob, which can merge two
            true peaks into one and worsen undercounting. The median instead
            picks the "majority" value at each position without smearing
            outlier windows into their neighbours. Costs extra memory
            (stacks every window's full contribution instead of one running
            accumulator) -- fine at the video lengths/window counts this
            pipeline sees, not intended for very long videos or tiny
            window_stride. CountHead is unaffected either way (see below --
            it has no per-token resolution to aggregate).

    CountHead is handled differently (see _count_head_tiled_sum): it predicts
    an already-integrated per-clip count, not a per-frame quantity, so it
    can't be scatter-averaged like density/in_rep -- when window_stride >=
    clip_length (the windows used for density above are already
    non-overlapping), each window's count_head prediction is summed directly
    with no extra forward passes; when window_stride < clip_length
    (overlapping, e.g. for density ensembling), count_head is reconstructed
    via one extra non-overlapping pass instead, independent of the stride
    used for everything else.

    return_in_rep=False, return_count_head=False (default): returns density
        (N,) on CPU. Raises if the model has no density head
        (use_density=False) -- there is then nothing to return; call with
        return_count_head=True instead (see below).
    return_in_rep=True: returns (density, in_rep).
    return_count_head=True: appends `count_head` (a 0-dim tensor, or None if
        the model has no count_head) as the LAST element of whatever tuple
        shape the flags above would otherwise produce -- (density,
        count_head) if return_in_rep is False, else (density, in_rep,
        count_head). `density`/`in_rep`/`count_head` are each (N,), (N,),
        0-dim or None depending on which heads the model has (unpack_output /
        unpack_in_repetition / unpack_count_head already reflect that) --
        callers must handle None in any position. With this flag, the
        RuntimeError above is skipped even if the model has no density head
        (use_density=False, use_count_head=True) -- count_head is then the
        only valid count source.

    pred_count = returned_density.sum() when a density head is present;
    else returned_count_head directly (if requested).

    temporal_downsample: 1 (default) -- encoder honours the normal per-frame
        contract, every returned tensor is length N as documented above.
        > 1 -- pass the encoder's own `encoder.temporal_downsample` (e.g.
        VideoMAEClipEncoder(interp_time="downsample")): each window then
        emits clip_length // temporal_downsample tokens instead of
        clip_length, so density/in_rep are reconstructed on an
        N // temporal_downsample accumulator instead of N -- returned
        tensors are correspondingly shorter. Requires clip_length %
        temporal_downsample == 0. Window start/valid-length bookkeeping
        stays in raw-frame space (unaffected); only the accumulator and the
        scatter-add indices move to token space (floor-divided, so an
        odd leftover raw frame at a window's tail is dropped from that
        window's contribution -- bounded, same spirit as the last-frame-
        repeat padding above). count_head is unaffected either way (see
        _count_head_tiled_sum -- it's a per-window scalar, no per-token
        resolution to speak of).
    """
    if clip_length % temporal_downsample != 0:
        raise ValueError(
            f"clip_length ({clip_length}) must be a multiple of temporal_downsample "
            f"({temporal_downsample}) so every window's token count divides evenly."
        )
    if aggregation not in ("mean", "median"):
        raise ValueError(f"aggregation must be 'mean' or 'median', got {aggregation!r}")
    N = frames.shape[0]
    if N == 0:
        zero_count_head = torch.zeros(()) if return_count_head else None
        if return_in_rep:
            result = (torch.zeros(0), torch.zeros(0))
        else:
            result = torch.zeros(0)
        if return_count_head:
            result = (result if isinstance(result, tuple) else (result,)) + (zero_count_head,)
        return result

    N_out   = N // temporal_downsample
    win_out = clip_length // temporal_downsample

    accum = torch.zeros(N_out, dtype=torch.float32)
    counts = torch.zeros(N_out, dtype=torch.float32)
    in_rep_accum = torch.zeros(N_out, dtype=torch.float32) if return_in_rep else None
    density_used = False

    # aggregation="median" needs every window's own contribution kept apart
    # (not summed into `accum`) until every window has run, so the per-position
    # median can be taken across them -- see _stitch_median below.
    density_windows = [] if aggregation == "median" else None
    in_rep_windows = [] if aggregation == "median" and return_in_rep else None

    non_overlapping = window_stride >= clip_length
    count_head_accum = 0.0
    count_head_used = False

    starts = compute_window_starts(N, clip_length, window_stride)

    for start in starts:
        valid = min(clip_length, N - start)
        window = frames[start:start + valid]                       # (valid,3,H,W)
        if valid < clip_length:                                    # pad by repeating last frame
            pad = window[-1:].expand(clip_length - valid, *window.shape[1:])
            window = torch.cat([window, pad], dim=0)
        window = window.unsqueeze(0).to(device, non_blocking=True)  # (1,L,3,H,W)

        # token-space window position/length -- see temporal_downsample above.
        token_start = start // temporal_downsample
        token_valid = min(win_out, valid // temporal_downsample)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model.forward_from_features(encoder(window))     # (1,win_out,D) or (1,win_out,S,D) -> out
            density, _, _ = unpack_output(out)
            in_rep = unpack_in_repetition(out) if return_in_rep else None
            # only usable directly (summed per non-overlapping window) when the
            # main loop itself is already non-overlapping -- see non_overlapping above.
            count_head_out = (
                unpack_count_head(out) if return_count_head and non_overlapping else None
            )

        if density is not None:
            density_used = True
            density = density.squeeze(0).float().cpu()[:token_valid]
            if aggregation == "median":
                density_windows.append((token_start, token_valid, density))
            else:
                accum[token_start:token_start + token_valid] += density
        counts[token_start:token_start + token_valid] += 1.0
        if return_in_rep and in_rep is not None:
            in_rep = in_rep.squeeze(0).float().cpu()[:token_valid]
            if aggregation == "median":
                in_rep_windows.append((token_start, token_valid, in_rep))
            else:
                in_rep_accum[token_start:token_start + token_valid] += in_rep
        if count_head_out is not None:
            count_head_used = True
            count_head_accum += count_head_out.squeeze(0).float().cpu().item()

    counts = counts.clamp(min=1.0)
    if aggregation == "median":
        full_density = _stitch_median(density_windows, N_out) if density_used else None
        full_in_rep = _stitch_median(in_rep_windows, N_out) if in_rep_windows is not None else None
    else:
        full_density = (accum / counts) if density_used else None
        full_in_rep = (in_rep_accum / counts) if in_rep_accum is not None else None

    full_count_head = None
    if return_count_head:
        if non_overlapping:
            full_count_head = torch.tensor(count_head_accum) if count_head_used else None
        else:
            full_count_head = _count_head_tiled_sum(encoder, model, frames, clip_length, device, use_amp)

    if return_in_rep:
        result = (full_density, full_in_rep)
    else:
        if full_density is None and not return_count_head:
            raise RuntimeError(
                "sliding_window_inference: this model has no density head "
                "(use_density=False) -- there is no density to reconstruct. "
                "Call with return_count_head=True to read CountHead's count "
                "directly instead."
            )
        result = full_density

    if return_count_head:
        result = (result if isinstance(result, tuple) else (result,)) + (full_count_head,)
    return result


def sliding_window_train_count(
    encoder,
    model,
    frames: torch.Tensor,
    clip_length: int,
    window_stride: int,
    device: torch.device,
    use_amp: bool = False,
    temporal_downsample: int = 1,
) -> torch.Tensor:
    """
    Whole-video predicted count, WITH gradients -- the full-video training
    counterpart to sliding_window_inference (see full_video_sampling in
    config.yaml / train.py). Tiles `frames` into clip_length windows (same
    tail-guaranteed compute_window_starts scheme), batches every window into
    ONE encoder+model forward pass, then stitches the per-window density
    predictions into one full-length curve via the same scatter-add +
    overlap-averaging sliding_window_inference uses -- except the accumulator
    stays graph-connected (no @torch.no_grad(), no .cpu() detach), since
    density requires grad. Returns a 0-dim tensor (full_density.sum()),
    backward()-able.

    Falls back to CountHead when the model has no density head
    (use_density=False): CountHead predicts an already-integrated per-clip
    count, so when window_stride >= clip_length (the windows already tiled
    above are non-overlapping -- the documented full_video_sampling default),
    its per-window predictions from that SAME batched forward pass are just
    summed directly, no extra compute. When window_stride < clip_length
    (overlapping -- summing would double-count), one extra non-overlapping
    grad-connected forward pass reconstructs it instead. Raises only if the
    model has neither a density head nor a count_head -- mirroring
    sliding_window_inference's own requirement.

    temporal_downsample: see sliding_window_inference -- pass the encoder's
    own `encoder.temporal_downsample`. The density accumulator (and the
    count_head fallback, which is resolution-agnostic already) both stay
    consistent with that function's token-space reconstruction.
    """
    if clip_length % temporal_downsample != 0:
        raise ValueError(
            f"clip_length ({clip_length}) must be a multiple of temporal_downsample "
            f"({temporal_downsample}) so every window's token count divides evenly."
        )
    N = frames.shape[0]
    N_out   = N // temporal_downsample
    win_out = clip_length // temporal_downsample
    starts = compute_window_starts(N, clip_length, window_stride)

    windows, valids = [], []
    for start in starts:
        valid = min(clip_length, N - start)
        window = frames[start:start + valid]
        if valid < clip_length:
            pad = window[-1:].expand(clip_length - valid, *window.shape[1:])
            window = torch.cat([window, pad], dim=0)
        windows.append(window)
        valids.append(valid)

    batch = torch.stack(windows, dim=0).to(device, non_blocking=True)  # (W,L,3,H,W)

    with torch.autocast(device_type=device.type, enabled=use_amp):
        feats = encoder(batch)                                  # (W,win_out,D) or (W,win_out,S,D)
        out = model.forward_from_features(feats)
        density, _, _ = unpack_output(out)

    if density is not None:
        density = density.float()

        accum  = torch.zeros(N_out, dtype=density.dtype, device=density.device)
        counts = torch.zeros(N_out, dtype=torch.float32, device=density.device)
        for w, (start, valid) in enumerate(zip(starts, valids)):
            token_start = start // temporal_downsample
            token_valid = min(win_out, valid // temporal_downsample)
            accum[token_start:token_start + token_valid]  = accum[token_start:token_start + token_valid] + density[w, :token_valid]
            counts[token_start:token_start + token_valid] = counts[token_start:token_start + token_valid] + 1.0

        full_density = accum / counts.clamp(min=1.0)
        return full_density.sum()

    count_head_out = unpack_count_head(out)
    if count_head_out is not None:
        if window_stride >= clip_length:
            return count_head_out.float().sum()
        # overlapping windows would double-count a CountHead prediction (it's
        # already an integrated per-clip count, not a per-frame rate) -- redo
        # with non-overlapping tiling instead, still grad-connected.
        starts2 = compute_window_starts(N, clip_length, clip_length)
        windows2 = []
        for start in starts2:
            valid = min(clip_length, N - start)
            window = frames[start:start + valid]
            if valid < clip_length:
                pad = window[-1:].expand(clip_length - valid, *window.shape[1:])
                window = torch.cat([window, pad], dim=0)
            windows2.append(window)
        batch2 = torch.stack(windows2, dim=0).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            count_head_out2 = unpack_count_head(model.forward_from_features(encoder(batch2)))
        return count_head_out2.float().sum()

    raise RuntimeError(
        "sliding_window_train_count: this model has no density head "
        "(use_density=False) and no count_head (count_head.enabled=False) -- "
        "there is no signal to reconstruct a whole-video count from."
    )


# ── checkpoints ──────────────────────────────────────────────────────────────────

def save_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None,
                    epoch: int = 0, best_metric: float = float("inf"), extra: Optional[dict] = None) -> None:
    """Save a resumable checkpoint. Handles DDP-wrapped models transparently."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "model":       raw_model.state_dict(),
        "optimizer":   optimizer.state_dict() if optimizer is not None else None,
        "scheduler":   scheduler.state_dict() if scheduler is not None else None,
        "scaler":      scaler.state_dict() if scaler is not None else None,
        "epoch":       epoch,
        "best_metric": best_metric,
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None,
                    map_location="cpu") -> dict:
    """
    Restore model (+ optionally optimizer/scheduler/scaler). Returns the raw dict.

    strict=False: older checkpoints (e.g. anything saved before the
    TemporalSelfSimilarity/TSMConvStack/TemporalTransformerAggregator dead-code
    removal) carry extra tsm.*/conv_stack.*/transformer.* weights that no
    longer exist on the model -- those are harmless to drop (that branch was
    never in the forward pass). Missing/unexpected keys are still printed
    rather than silently swallowed, since an unexpected mismatch can also
    mean a genuine architecture mismatch (e.g. use_in_repetition/use_phase
    not matching between the checkpoint and this model) that you DO want to
    notice.
    """
    ckpt = torch.load(path, map_location=map_location)
    raw_model = model.module if hasattr(model, "module") else model
    result = raw_model.load_state_dict(ckpt["model"], strict=False)
    if result.missing_keys:
        print(f"[load_checkpoint] {len(result.missing_keys)} missing key(s) "
              f"(randomly initialized instead): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[load_checkpoint] {len(result.unexpected_keys)} unexpected key(s) "
              f"in checkpoint (dropped): {result.unexpected_keys}")
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


# ── weighted sampler (rare high-count upsampling) ───────────────────────────────

def _rarity_weights(dataset, n_bins: int = 10, bin_strategy: str = "width") -> torch.Tensor:
    """
    Per-sample weight inversely proportional to its count-bin's population.

    bin_strategy="width" (default, original behaviour): n_bins equal-WIDTH
        bins over the raw count VALUE range. Badly miscalibrated on
        right-skewed count distributions (e.g. Countix: counts 2-73, but
        77% of videos have count in [2,10)) -- that entire majority lands in
        ONE bin and gets crushed to a tiny fraction of total sampling
        weight, while a couple of extreme outliers each get a per-sample
        weight rivaling the whole majority combined. Verified on Countix:
        the [2,10) majority (76.9% of the data) gets just 11.1% of total
        sampling weight under this scheme.
    bin_strategy="quantile": n_bins equal-POPULATION bins (via quantiles of
        the count distribution) -- every bin gets a comparable share of
        total sampling weight regardless of how skewed the raw values are.
        Same Countix majority gets 72.7% of total weight instead of 11.1%
        -- still upsamples the genuine tail, without starving the bulk of
        the data of training exposure.
    """
    if bin_strategy not in ("width", "quantile"):
        raise ValueError(f"bin_strategy must be 'width' or 'quantile', got {bin_strategy!r}")
    counts = torch.tensor([c for _, c, _ in dataset.samples], dtype=torch.float32)
    if bin_strategy == "quantile":
        qs = torch.linspace(0, 1, n_bins + 1)
        edges = torch.unique(torch.quantile(counts, qs))
        n_bins = max(1, len(edges) - 1)
    else:
        lo, hi = counts.min(), counts.max() + 1e-6
        edges  = torch.linspace(lo, hi, n_bins + 1)
    bin_ids = torch.bucketize(counts, edges[1:-1]).clamp(max=n_bins - 1)
    bin_counts = torch.zeros(n_bins)
    for b in bin_ids:
        bin_counts[b] += 1
    return (1.0 / bin_counts.clamp(min=1))[bin_ids]


def make_count_sampler(dataset, n_bins: int = 10, bin_strategy: str = "width"):
    """WeightedRandomSampler that upsamples rare counts. Needs dataset.samples. See _rarity_weights."""
    from torch.utils.data import WeightedRandomSampler
    weights = _rarity_weights(dataset, n_bins, bin_strategy=bin_strategy)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


def make_zero_upweight_sampler(dataset, zero_weight: float = 5.0,
                                rarity_weighted: bool = False, n_bins: int = 10,
                                bin_strategy: str = "width"):
    """
    WeightedRandomSampler that multiplies the sampling weight of every
    ZERO-count sample by `zero_weight`, leaving every other sample at its
    base weight. Needs dataset.samples ([(video_id, count, num_frames), ...]).

    Zero-count videos (no repetition at all) are typically extremely scarce
    in RepCount-style datasets (often <1%) yet important: without enough
    sampling pressure toward them, the model rarely sees a true "no
    repetition" example during training, which encourages exactly the
    false-positive counting behaviour loss.ZeroCountLoss / config's
    zero_count_penalty penalizes directly. This attacks the same problem
    from the DATA side instead -- see zero-count examples more often per
    epoch -- and composes with that loss term (use either, both, or neither).

    rarity_weighted=False (default): every non-zero sample has base weight
        1.0 -- zero-count samples end up EXACTLY `zero_weight`x as likely to
        be drawn as a typical non-zero sample. True: start from
        make_count_sampler's rarity weights instead of 1.0 (composes with
        weighted_sampler's rare-high-count upsampling rather than replacing
        it) -- zero-count samples still end up `zero_weight`x their own
        rarity weight, but no longer directly comparable 1:1 to non-zero
        samples' raw draw probability.
    """
    from torch.utils.data import WeightedRandomSampler
    counts = torch.tensor([c for _, c, _ in dataset.samples], dtype=torch.float32)
    base = _rarity_weights(dataset, n_bins, bin_strategy=bin_strategy) if rarity_weighted else torch.ones_like(counts)
    weights = torch.where(counts <= 0, base * zero_weight, base)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


class FullVideoSplitSampler(Sampler[int]):
    """
    Per-epoch 80/20 (configurable) video-level split for full_video_sampling
    (see config.yaml / train.py). Each epoch, freshly selects
    round(fraction * num_videos) video indices to be that epoch's "full
    video" set -- excluded from what this sampler yields, and processed
    separately by train.py's full-video count step (see
    sliding_window_train_count). The shuffled COMPLEMENT (the other videos)
    is yielded here, feeding FrameClipDataset's normal one-random-crop path
    completely unchanged.

    Selection is a pure function of (seed, epoch) via
    random.Random(seed + epoch) -- deterministic and reproducible across
    checkpoint resume, with no dependency on prior random state. Mirrors
    DistributedSampler.set_epoch's pattern: this Sampler lives in the main
    process and is re-iterated fresh every epoch, so it works correctly even
    with DataLoader(persistent_workers=True) -- only plain integer indices
    get dispatched to the (already-forked) workers.

    Single-process only: does no rank/world_size partitioning. Combining
    full_video_sampling.enabled with DDP is rejected early in train.py.
    """

    def __init__(self, num_videos: int, fraction: float, seed: int):
        if not (0.0 <= fraction <= 1.0):
            raise ValueError(f"full_video_sampling.fraction must be in [0,1], got {fraction}")
        self.num_videos = num_videos
        self.fraction   = fraction
        self.seed       = seed
        self.n_full     = round(fraction * num_videos)
        self.epoch      = 0
        self.full_video_indices: list[int] = []
        self._normal_indices: list[int] = list(range(num_videos))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        rng = random.Random(self.seed + epoch)
        full = set(rng.sample(range(self.num_videos), self.n_full)) if self.n_full > 0 else set()
        self.full_video_indices = sorted(full)
        normal = [i for i in range(self.num_videos) if i not in full]
        rng.shuffle(normal)
        self._normal_indices = normal

    def __iter__(self):
        return iter(self._normal_indices)

    def __len__(self) -> int:
        return len(self._normal_indices)


class FullVideoPrefetcher:
    """
    Background-thread double-buffering for the full_video_sampling training
    step (see train.py: `for sample in prefetcher:` replaces
    `for vi in full_video_sampler.full_video_indices: sample = full_video_ds[vi]`).

    FrameVideoDataset.__getitem__ synchronously JPEG-decodes EVERY frame of a
    whole video (no DataLoader/num_workers involved for this path, unlike the
    main per-clip train_loader) -- at a low full_video_sampling.fraction this
    is hidden by the batched loop's own I/O, but at a high fraction it starts
    to dominate: the GPU sits idle between videos while the next one decodes
    on the main thread. This class submits dataset[idx] for the NEXT video to
    a single background thread while the caller is busy with the CURRENT
    video's forward/backward/optimizer step, so decode and GPU compute
    overlap. PIL's JPEG decoder releases the GIL for the bulk of its work, so
    a single thread (no multiprocessing, no pickling of the dataset) already
    gets real overlap.

    Usage: `for sample in FullVideoPrefetcher(full_video_ds, indices): ...`
    -- yields exactly what `dataset[idx]` would, for each idx in `indices`,
    in order.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pos = 0
        self._future = None
        self._submit_next()

    def _submit_next(self) -> None:
        if self._pos < len(self.indices):
            idx = self.indices[self._pos]
            self._future = self._executor.submit(self.dataset.__getitem__, idx)
            self._pos += 1
        else:
            self._future = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._future is None:
            self._executor.shutdown(wait=False)
            raise StopIteration
        sample = self._future.result()
        self._submit_next()
        return sample


# ── visualisation ────────────────────────────────────────────────────────────────

def kl_divergence_of_count_distributions(gts, preds, num_bins: Optional[int] = None) -> float:
    """
    KL(gt || pred) between the count histograms. High when the model's overall
    count distribution is mismatched (e.g. mean regression into a narrow band).
    """
    gts = np.asarray(gts, dtype=np.float64); preds = np.asarray(preds, dtype=np.float64)
    if len(gts) == 0:
        return 0.0
    if num_bins is None:
        num_bins = int(max(gts.max(), preds.max()) + 1) + 1
    hist_gt, _   = np.histogram(gts,   bins=num_bins, range=(0, num_bins))
    hist_pred, _ = np.histogram(preds, bins=num_bins, range=(0, num_bins))
    eps = 1e-8
    p_gt = (hist_gt   + eps) / (hist_gt.sum()   + eps * num_bins)
    p_pr = (hist_pred + eps) / (hist_pred.sum() + eps * num_bins)
    return float(np.sum(p_gt * np.log(p_gt / p_pr)))


def plot_count_distributions(gts, preds, kl: float, out_path: str) -> None:
    """Overlaid histogram of ground-truth vs predicted count distributions."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    max_c = int(max(max(gts), max(preds)) + 1)
    bins = np.arange(0, max_c + 2) - 0.5
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(gts,   bins=bins, alpha=0.6, color="teal",  label="ground truth", edgecolor="white")
    ax.hist(preds, bins=bins, alpha=0.6, color="coral", label="prediction",   edgecolor="white")
    ax.set_xlabel("count"); ax.set_ylabel("frequency"); ax.set_yscale("log")
    ax.set_title(f"Count distribution — KL(gt || pred) = {kl:.4f}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); _savefig(plt, out_path)


def plot_class_distribution(class_counts: dict, class_maes: dict, out_path: str, top_n: int = 20) -> None:
    """Two-panel: sample count per class + per-class MAE (sorted by frequency)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    classes = sorted(class_counts, key=lambda c: -class_counts[c])[:top_n]
    counts = [class_counts[c] for c in classes]
    maes   = [class_maes[c]   for c in classes]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(classes) * 0.35), 8))
    ax1.bar(range(len(classes)), counts, color="steelblue", edgecolor="white")
    ax1.set_xticks(range(len(classes))); ax1.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("sample count"); ax1.set_title(f"Class distribution (top {len(classes)})")
    ax1.grid(True, axis="y", alpha=0.3)
    ax2.bar(range(len(classes)), maes, color="coral", edgecolor="white")
    ax2.set_xticks(range(len(classes))); ax2.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("MAE"); ax2.set_title("Per-class MAE"); ax2.grid(True, axis="y", alpha=0.3)
    plt.tight_layout(); _savefig(plt, out_path)


def plot_density_grid(samples: list, out_path: str, n_cols: int = 2) -> None:
    """
    Grid of predicted vs GT density curves. `samples`: list of dicts with keys
    video_id, gt_density, pred_density, gt_count, pred_count, [class].
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(samples)
    if n == 0:
        return
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 2.4), squeeze=False)
    for i, s in enumerate(samples):
        ax = axes[i // n_cols][i % n_cols]
        gt = s["gt_density"]; pred = s["pred_density"]
        if isinstance(gt, torch.Tensor):   gt = gt.detach().cpu().numpy()
        if isinstance(pred, torch.Tensor): pred = pred.detach().cpu().numpy()
        ax.fill_between(np.arange(len(gt)), gt, alpha=0.35, color="teal",
                        label=f"GT ({s['gt_count']:.0f})")
        ax.plot(np.arange(len(pred)), pred, color="coral", lw=1.4,
                label=f"pred ({s['pred_count']:.1f})")
        title = s["video_id"] + (f"  [{s['class']}]" if s.get("class") else "")
        ax.set_title(title, fontsize=8); ax.grid(True, alpha=0.3); ax.legend(fontsize=7, loc="upper right")
    for i in range(n, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis("off")
    plt.tight_layout(); _savefig(plt, out_path)


def plot_in_repetition_grid(samples: list, out_path: str, n_cols: int = 2) -> None:
    """
    Grid of predicted (sigmoid probability) vs GT in-repetition curves.
    `samples`: list of dicts with keys video_id, gt_in_rep, pred_in_rep,
    [class]. gt_in_rep is the phase_mask (1.0 inside an annotated cycle, 0.0
    outside); pred_in_rep is the InRepetitionHead's sigmoid probability.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(samples)
    if n == 0:
        return
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 2.4), squeeze=False)
    for i, s in enumerate(samples):
        ax = axes[i // n_cols][i % n_cols]
        gt = s["gt_in_rep"]; pred = s["pred_in_rep"]
        if isinstance(gt, torch.Tensor):   gt = gt.detach().cpu().numpy()
        if isinstance(pred, torch.Tensor): pred = pred.detach().cpu().numpy()
        ax.fill_between(np.arange(len(gt)), gt, step="mid", alpha=0.35, color="teal", label="GT")
        ax.plot(np.arange(len(pred)), pred, color="coral", lw=1.4, label="pred (prob)")
        title = s["video_id"] + (f"  [{s['class']}]" if s.get("class") else "")
        ax.set_title(title, fontsize=8); ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7, loc="upper right")
    for i in range(n, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis("off")
    plt.tight_layout(); _savefig(plt, out_path)


def _savefig(plt, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150); plt.close()


def plot_density(pred_density, gt_density, out_path: str, title: str = "") -> None:
    """Overlay predicted vs GT density curves and save to out_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(pred_density, torch.Tensor): pred_density = pred_density.detach().cpu().numpy()
    if isinstance(gt_density, torch.Tensor):   gt_density = gt_density.detach().cpu().numpy()

    t = np.arange(len(pred_density))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(np.arange(len(gt_density)), gt_density, alpha=0.35, color="teal",
                    label=f"GT (sum={float(np.sum(gt_density)):.1f})")
    ax.plot(t, pred_density, color="coral", lw=1.5,
            label=f"pred (sum={float(np.sum(pred_density)):.1f})")
    ax.set_xlabel("frame"); ax.set_ylabel("density")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


