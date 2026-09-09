"""
Datasets over the decoded-frame cache produced by preprocess_frames.py.

Two access patterns:

FrameClipDataset (TRAINING)
    Random temporal crop augmentation. Every __getitem__:
        1. loads the video's metadata,
        2. picks a RANDOM start frame,
        3. extracts ONE clip of `clip_length` frames at `temporal_stride`
           (span = clip_length * temporal_stride raw frames),
        4. pads by repeating the last frame if the clip runs past the end,
        5. builds the density map for exactly that cropped span (cycle-accurate
           for RepCount, uniform prior for Countix),
        6. count = density.sum() over the crop.
    Because the start is random, the SAME video yields a DIFFERENT clip each
    epoch -> temporal data augmentation with zero re-decoding.

FrameVideoDataset (VALIDATION / INFERENCE)
    Deterministic: returns the WHOLE decoded video plus a full-length
    ground-truth density map, for sliding-window inference (see utils.py /
    infer.py). batch_size=1 (videos have variable length).

Both read the self-describing cache -- counts and cycle boundaries come from each
video's metadata.json, so no CSV is needed at train time.
"""

from __future__ import annotations

import json
import os
import random
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from density import (  # noqa: E402
    make_density_map_from_frames,
    make_density_map_uniform,
    make_phase_targets,
)


# ── frame I/O + density helpers ────────────────────────────────────────────────

def discover_videos(roots: Sequence[str]) -> list[str]:
    """Return every <root>/<video_id> dir that contains a metadata.json."""
    dirs = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if os.path.isfile(os.path.join(d, "metadata.json")):
                dirs.append(d)
    return dirs


def _resize_short_side_center_crop(img: Image.Image, size: int) -> Image.Image:
    """Resize so the SHORTER side is `size` (aspect ratio preserved), then center-crop to (size, size)."""
    w, h = img.size
    if w <= h:
        new_w, new_h = size, round(h * size / w)
    else:
        new_h, new_w = size, round(w * size / h)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left, top = (new_w - size) // 2, (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


class Preprocess:
    """
    Resize + to-tensor + ImageNet-style normalise, using only PIL + torch (no
    torchvision dependency). Callable on a PIL RGB image -> (3, H, W) float32.

    resize_mode:
        "squash" (default) -- resize directly to (image_size, image_size),
            ignoring aspect ratio. A no-op for frames already cached at
            image_size by preprocess_frames.py; the mode that matters is
            whatever THAT script used at decode time. For raw video read
            directly (e.g. infer.py, which never goes through the cache),
            this IS the resize actually applied.
        "short_side_crop" -- resize so the shorter side is image_size (aspect
            ratio preserved), then center-crop to (image_size, image_size) --
            standard ImageNet-style preprocessing, matching
            preprocess_frames.py's --resize_mode short_side_crop.
    """

    def __init__(self, image_size: int, mean, std, resize_mode: str = "squash"):
        if resize_mode not in ("squash", "short_side_crop"):
            raise ValueError(f"resize_mode must be 'squash' or 'short_side_crop', got {resize_mode!r}")
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std  = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.size != (self.image_size, self.image_size):
            if self.resize_mode == "short_side_crop":
                img = _resize_short_side_center_crop(img, self.image_size)
            else:
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
        return (arr - self.mean) / self.std


def build_transform(image_size: int, mean, std, resize_mode: str = "squash") -> Preprocess:
    """Resize + normalise for the selected encoder."""
    return Preprocess(image_size, mean, std, resize_mode=resize_mode)


def load_frames(video_dir: str, indices: Sequence[int], meta: dict, transform) -> torch.Tensor:
    """Load the given saved-frame indices as a normalised (len(indices),3,H,W) tensor."""
    ext = meta.get("image_ext", "jpg")
    imgs = []
    for idx in indices:
        path = os.path.join(video_dir, f"{idx:06d}.{ext}")
        with Image.open(path) as im:
            imgs.append(transform(im.convert("RGB")))
    return torch.stack(imgs, dim=0)


def _bin_sum(seg: torch.Tensor, n_bins: int) -> torch.Tensor:
    """
    Sum-preserving resample of a 1-D density segment into `n_bins` bins.

    Used for the Countix uniform prior when a crop is temporally strided: naive
    strided point-sampling of a per-frame density would DROP mass between
    sampled frames and undercount. Binning assigns every source frame's mass to
    exactly one output bin, so seg.sum() == output.sum() (count is preserved).
    """
    out = torch.zeros(n_bins)
    M = seg.shape[0]
    if M == 0:
        return out
    bins = (torch.arange(M, dtype=torch.float32) * n_bins / M).floor().long().clamp(max=n_bins - 1)
    out.index_add_(0, bins, seg)
    return out


def build_clip_density(
    meta: dict,
    start: int,
    span: int,
    clip_length: int,
    sigma_factor: float,
    density_kernel: str = "gaussian",
) -> torch.Tensor:
    """
    Density map for the crop [start, start+span) resampled to `clip_length`
    output positions. Output frame i corresponds to raw frame start + i*stride,
    so density peaks line up with the strided clip frames.

    RepCount (cycle_starts present): cycle-accurate bumps (density_kernel shape).
    Countix  (no cycles):            uniform prior, count-preserving under stride.
    """
    num_frames    = meta["num_frames"]
    cycle_starts  = meta.get("cycle_starts") or []
    cycle_ends    = meta.get("cycle_ends") or []
    count         = meta.get("count", 0)

    if cycle_starts:
        # keep cycles whose CENTER falls inside the crop; index relative to start
        rel_s, rel_e = [], []
        for s, e in zip(cycle_starts, cycle_ends):
            center = (s + e) / 2.0
            if start <= center < start + span:
                rel_s.append(s - start)
                rel_e.append(e - start)
        return make_density_map_from_frames(rel_s, rel_e, clip_length, span, sigma_factor, density_kernel)

    # Countix: uniform prior over the whole video, then crop + count-preserving rebin
    full = make_density_map_uniform(int(round(count)), num_frames, sigma_factor, density_kernel)
    seg  = full[start:min(start + span, num_frames)]
    return _bin_sum(seg, clip_length)


def clip_frame_indices(start: int, clip_length: int, stride: int, num_frames: int) -> list[int]:
    """Strided frame indices for a clip, clamped to the last frame (pad-by-repeat)."""
    return [min(start + i * stride, num_frames - 1) for i in range(clip_length)]


# ── training dataset: random temporal crop ─────────────────────────────────────

class FrameClipDataset(Dataset):
    """
    Random-temporal-crop clips from the decoded-frame cache.

    Returns (clip, density, count, phase_sin, phase_cos, phase_mask):
        clip       : (clip_length, 3, H, W) float32, normalised for the encoder
        density    : (clip_length,)         float32, sums to `count`
        count      : scalar float32
        phase_sin  : (clip_length,) float32, sin(2*pi*phase-in-cycle), see
                     density.make_phase_targets. Auxiliary target -- 0 where
                     phase_mask is 0.
        phase_cos  : (clip_length,) float32, cos(2*pi*phase-in-cycle).
        phase_mask : (clip_length,) float32, 1.0 inside an annotated cycle,
                     0.0 outside (e.g. Countix clips, which have no cycle
                     annotations, are all-zero here).
    """

    def __init__(
        self,
        roots: Sequence[str],
        clip_length: int = 64,
        temporal_stride: int = 1,
        image_size: int = 224,
        pixel_mean=(0.485, 0.456, 0.406),
        pixel_std=(0.229, 0.224, 0.225),
        sigma_factor: float = 0.4,
        density_kernel: str = "gaussian",
        random_crop: bool = True,
        crop_mode: str = "cycle",
        cycle_jitter: int = 0,
        seed: Optional[int] = None,
        resize_mode: str = "squash",
        temporal_downsample: int = 1,
    ):
        """
        temporal_downsample:
            1 (default) -- density/phase targets are built at `clip_length`
            positions, matching every encoder's normal per-frame T-in/T-out
            contract. > 1 -- targets are instead built at `clip_length //
            temporal_downsample` positions, matching an encoder that natively
            emits fewer output tokens than input frames (e.g.
            VideoMAEClipEncoder(interp_time="downsample"); pass its
            `encoder.temporal_downsample` here). `clip` itself is unaffected
            -- still `clip_length` raw frames -- since the encoder does its
            own downsampling internally.
        crop_mode:
            "cycle"   — snap the random start to a randomly chosen repetition
                        boundary (cycle_start), so every clip begins on a clean
                        rep and no Gaussian is truncated at the LEADING edge.
                        Falls back to "uniform" automatically for videos without
                        cycle annotations (e.g. Countix).
            "uniform" — random start at any frame (max phase diversity, but a
                        cycle can be sliced at the clip's leading edge).
        cycle_jitter:
            When crop_mode="cycle", perturb the snapped start by up to
            +/- cycle_jitter frames. 0 = start exactly on the boundary; a small
            value (e.g. temporal_stride) restores some phase diversity while
            keeping starts near a rep boundary.
        """
        if crop_mode not in ("cycle", "uniform"):
            raise ValueError("crop_mode must be 'cycle' or 'uniform'.")
        if clip_length % temporal_downsample != 0:
            raise ValueError(
                f"clip_length ({clip_length}) must be a multiple of temporal_downsample "
                f"({temporal_downsample}) so the GT density/phase target length divides evenly."
            )
        self.video_dirs      = discover_videos(roots)
        if not self.video_dirs:
            raise FileNotFoundError(f"No decoded videos with metadata.json under {list(roots)}")
        self.clip_length     = clip_length
        self.temporal_stride = temporal_stride
        self.sigma_factor    = sigma_factor
        self.density_kernel  = density_kernel
        self.random_crop     = random_crop
        self.crop_mode       = crop_mode
        self.cycle_jitter    = cycle_jitter
        self.transform       = build_transform(image_size, pixel_mean, pixel_std, resize_mode=resize_mode)
        self.temporal_downsample = temporal_downsample
        self._rng            = random.Random(seed)

        # cache lightweight (id, count, num_frames) so a WeightedRandomSampler
        # can be built without reopening every metadata file (see
        # utils.make_count_sampler).
        self.samples = []
        for d in self.video_dirs:
            with open(os.path.join(d, "metadata.json")) as f:
                m = json.load(f)
            self.samples.append((m["video_id"], float(m.get("count", 0)), int(m.get("num_frames", 0))))

    def __len__(self) -> int:
        return len(self.video_dirs)

    def _choose_start(self, meta: dict, span: int) -> int:
        """
        Pick the clip's start frame.

        crop_mode="cycle" (default): choose a random repetition boundary
        (cycle_start) that still leaves room for a full clip, so the clip opens
        on a clean rep. Falls back to uniform when the video has no cycles.
        Deterministic (start=0) when random_crop is False.
        """
        num_frames = meta["num_frames"]
        max_start  = max(0, num_frames - span)
        if not self.random_crop or max_start == 0:
            return 0

        cycles = meta.get("cycle_starts") or []
        if self.crop_mode == "cycle" and cycles:
            # candidate rep boundaries that leave room for a full clip; if a
            # video's reps are all late (none fit), keep the earliest so we
            # still start on a boundary rather than mid-rep.
            cand = [c for c in cycles if 0 <= c <= max_start]
            base = self._rng.choice(cand) if cand else min(max(cycles[0], 0), max_start)
            if self.cycle_jitter > 0:
                base += self._rng.randint(-self.cycle_jitter, self.cycle_jitter)
            return int(min(max(0, base), max_start))

        return self._rng.randint(0, max_start)

    def __getitem__(self, i: int):
        video_dir = self.video_dirs[i]
        with open(os.path.join(video_dir, "metadata.json")) as f:
            meta = json.load(f)

        num_frames = meta["num_frames"]
        span       = self.clip_length * self.temporal_stride
        start      = self._choose_start(meta, span)

        indices = clip_frame_indices(start, self.clip_length, self.temporal_stride, num_frames)
        clip    = load_frames(video_dir, indices, meta, self.transform)   # (L,3,H,W) -- L = clip_length always
        n_out   = self.clip_length // self.temporal_downsample
        density = build_clip_density(meta, start, span, n_out, self.sigma_factor, self.density_kernel)
        count   = density.sum()

        # phase targets are defined directly on real frame indices, not
        # rescaled like density -- normally (temporal_downsample=1) each
        # output position already maps 1:1 to a real sampled frame, so no
        # window/T rescaling is needed. When the encoder downsamples, output
        # token i covers raw input frames [i*downsample, (i+1)*downsample) --
        # e.g. VideoMAE's tubelet pairs (t,t+1) into one token (i, i+1) ->
        # token i (see frame_encoder._build_adjacent_pairs/_windowed_tokens)
        # -- so take one representative raw index per token (the first of
        # each group) instead of one per raw frame.
        phase_indices = indices[::self.temporal_downsample][:n_out]
        phase_sin, phase_cos, phase_mask = make_phase_targets(
            meta.get("cycle_starts") or [], meta.get("cycle_ends") or [], phase_indices,
        )
        return clip, density, count, phase_sin, phase_cos, phase_mask


class CachedFeatureClipDataset(FrameClipDataset):
    """
    Same crop selection and target construction as FrameClipDataset, but reads
    a PRECOMPUTED encoder feature tensor (extract_encoder_cache.py's output)
    instead of loading + encoding raw frames -- skips the (frozen, otherwise
    identical every epoch) encoder forward pass entirely at train time. See
    extract_encoder_cache.py's module docstring for the correctness argument
    and the empirical verification behind this.

    STRICT requirement, enforced at construction: crop_mode="cycle" and
    cycle_jitter=0. Only under exactly that combination does _choose_start
    (inherited unchanged from FrameClipDataset) ever return a start that
    extract_encoder_cache.py actually cached -- any other combination would
    look up a start_<c>.pt that was never written. If you need crop_mode=
    "uniform" or cycle_jitter>0, use FrameClipDataset (online encoding) for
    that data instead, or re-run extract_encoder_cache.py's jitter-extended
    variant (not implemented as of this class).

    roots must point at the ENCODED CACHE root (e.g.
    cache/encoded_features/countix/train), not the raw decoded-frame root --
    each cached video dir is self-contained (its own copied metadata.json +
    start_<c>.pt files, written by extract_encoder_cache.py), so no raw frame
    root or encoder is needed here at all.

    Returns (features, density, count, phase_sin, phase_cos, phase_mask) --
    identical contract to FrameClipDataset except `features` replaces `clip`:
        features: (clip_length, S, D) or (clip_length, D) float32 -- ALREADY
                  encoder output, not raw pixels. Feed directly to
                  model.forward_from_features(...); do NOT pass through
                  encoder(...) again.
    image_size/pixel_mean/pixel_std/resize_mode are accepted (as **_ignored)
    purely so this class can be constructed with the same kwargs train.py
    already passes to FrameClipDataset -- unused, since no frames are loaded.
    """

    def __init__(
        self,
        roots: Sequence[str],
        clip_length: int = 64,
        temporal_stride: int = 1,
        sigma_factor: float = 0.4,
        density_kernel: str = "gaussian",
        crop_mode: str = "cycle",
        cycle_jitter: int = 0,
        seed: Optional[int] = None,
        temporal_downsample: int = 1,
        **_ignored,
    ):
        if crop_mode != "cycle" or cycle_jitter != 0:
            raise ValueError(
                f"CachedFeatureClipDataset requires crop_mode='cycle' and cycle_jitter=0 -- "
                f"the cache only covers exact cycle-boundary starts (see "
                f"extract_encoder_cache.py). Got crop_mode={crop_mode!r}, "
                f"cycle_jitter={cycle_jitter}. Use FrameClipDataset (online encoding) instead "
                f"for this crop_mode/jitter combination."
            )
        super().__init__(
            roots, clip_length=clip_length, temporal_stride=temporal_stride,
            sigma_factor=sigma_factor, density_kernel=density_kernel, random_crop=True,
            crop_mode=crop_mode, cycle_jitter=cycle_jitter, seed=seed,
            temporal_downsample=temporal_downsample,
        )

    def __getitem__(self, i: int):
        video_dir = self.video_dirs[i]
        with open(os.path.join(video_dir, "metadata.json")) as f:
            meta = json.load(f)

        num_frames = meta["num_frames"]
        span       = self.clip_length * self.temporal_stride
        start      = self._choose_start(meta, span)

        feat_path = os.path.join(video_dir, f"start_{start}.pt")
        if not os.path.isfile(feat_path):
            raise FileNotFoundError(
                f"{feat_path} not cached -- re-run extract_encoder_cache.py for this root/config "
                f"(clip_length/temporal_stride mismatch, or cycle_starts changed since caching)."
            )
        features = torch.load(feat_path).float()

        n_out   = self.clip_length // self.temporal_downsample
        density = build_clip_density(meta, start, span, n_out, self.sigma_factor, self.density_kernel)
        count   = density.sum()

        indices = clip_frame_indices(start, self.clip_length, self.temporal_stride, num_frames)
        phase_indices = indices[::self.temporal_downsample][:n_out]
        phase_sin, phase_cos, phase_mask = make_phase_targets(
            meta.get("cycle_starts") or [], meta.get("cycle_ends") or [], phase_indices,
        )
        return features, density, count, phase_sin, phase_cos, phase_mask


# ── validation / inference dataset: whole video ────────────────────────────────

class FrameVideoDataset(Dataset):
    """
    Whole decoded videos for sliding-window evaluation.

    Returns a dict:
        video_id   : str
        frames     : (N, 3, H, W) float32 normalised   (N = num_frames)
        density    : (N_out,)     float32 full-length GT density (sums to count).
                     N_out = N // temporal_downsample (== N when
                     temporal_downsample=1, the default) -- see
                     utils.sliding_window_inference, which reconstructs its
                     own prediction at this same N_out resolution so the two
                     stay comparable.
        count      : scalar float32
        phase_sin  : (N_out,) float32, see FrameClipDataset / density.make_phase_targets
        phase_cos  : (N_out,) float32
        phase_mask : (N_out,) float32, 1.0 inside an annotated cycle, 0.0 outside
        cycle_starts, cycle_ends : list[int], RAW (un-subsampled) frame indices
                     from metadata.json -- [] for videos with no cycle
                     annotations (e.g. Countix). For evaluate.py-style per-window
                     ground-truth repetition counting off the raw annotations
                     directly, rather than the smoothed `density` curve.
        num_frames : int, RAW (un-subsampled) frame count -- the unit
                     cycle_starts/cycle_ends are expressed in.
    """

    def __init__(
        self,
        roots: Sequence[str],
        image_size: int = 224,
        pixel_mean=(0.485, 0.456, 0.406),
        pixel_std=(0.229, 0.224, 0.225),
        sigma_factor: float = 0.4,
        density_kernel: str = "gaussian",
        temporal_stride: int = 1,
        resize_mode: str = "squash",
        temporal_downsample: int = 1,
    ):
        self.video_dirs      = discover_videos(roots)
        if not self.video_dirs:
            raise FileNotFoundError(f"No decoded videos with metadata.json under {list(roots)}")
        self.sigma_factor    = sigma_factor
        self.density_kernel  = density_kernel
        self.temporal_stride = temporal_stride
        self.transform       = build_transform(image_size, pixel_mean, pixel_std, resize_mode=resize_mode)
        self.temporal_downsample = temporal_downsample
        self.samples = []
        for d in self.video_dirs:
            with open(os.path.join(d, "metadata.json")) as f:
                m = json.load(f)
            self.samples.append((m["video_id"], float(m.get("count", 0)), int(m.get("num_frames", 0))))

    def __len__(self) -> int:
        return len(self.video_dirs)

    def _full_density(self, meta: dict, n_out: int) -> torch.Tensor:
        """Full-length GT density (n_out positions) covering the entire video."""
        num_frames   = meta["num_frames"]
        cycle_starts = meta.get("cycle_starts") or []
        cycle_ends   = meta.get("cycle_ends") or []
        if cycle_starts:
            return make_density_map_from_frames(
                cycle_starts, cycle_ends, n_out, num_frames, self.sigma_factor, self.density_kernel
            )
        return make_density_map_uniform(
            int(round(meta.get("count", 0))), n_out, self.sigma_factor, self.density_kernel
        )

    def __getitem__(self, i: int) -> dict:
        video_dir = self.video_dirs[i]
        with open(os.path.join(video_dir, "metadata.json")) as f:
            meta = json.load(f)

        num_frames = meta["num_frames"]
        # optionally subsample the whole video by temporal_stride to match the
        # temporal resolution the model was trained at
        indices = list(range(0, num_frames, self.temporal_stride))
        frames  = load_frames(video_dir, indices, meta, self.transform)   # (N,3,H,W) -- N = len(indices), always raw resolution
        n_out   = len(indices) // self.temporal_downsample
        density = self._full_density(meta, n_out)
        # see FrameClipDataset.__getitem__ -- same "one representative raw
        # index per output token" resampling, needed here because
        # sliding_window_inference reconstructs its prediction at this same
        # N // temporal_downsample resolution (see utils.py).
        phase_indices = indices[::self.temporal_downsample][:n_out]
        phase_sin, phase_cos, phase_mask = make_phase_targets(
            meta.get("cycle_starts") or [], meta.get("cycle_ends") or [], phase_indices,
        )
        return {
            "video_id":   meta["video_id"],
            "class":      meta.get("class_label", meta.get("dataset_name", "unknown")),
            "frames":     frames,
            "density":    density,
            "count":      torch.tensor(float(meta.get("count", 0)), dtype=torch.float32),
            "phase_sin":  phase_sin,
            "phase_cos":  phase_cos,
            "phase_mask": phase_mask,
            "cycle_starts": meta.get("cycle_starts") or [],
            "cycle_ends":   meta.get("cycle_ends") or [],
            "num_frames":   num_frames,
        }


def collate_videos(batch: list) -> list:
    """Videos have variable length -> keep the batch as a plain list of dicts."""
    return batch
