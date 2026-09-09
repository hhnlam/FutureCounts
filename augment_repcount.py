"""
Offline domain-gap augmentation over the decoded RepCount frame cache.

Motivation
----------
Per-class evaluation on Countix-OOD (evaluate_countix_ood.py) consistently shows
"doing aerobics" as the worst-performing category (and, close behind, other
fast/high-frequency or handheld-camera categories: playing tennis, slicing
onion, spinning poi, playing ukulele, shaking head). RepCount, by contrast, is
homogeneous tripod/gym footage with essentially zero camera motion and no
speed diversity. This script synthesizes RepCount training variants with the
properties that make those Countix categories hard: variable playback speed,
camera tilt, and smoothed handheld-style camera shake.

This does NOT touch raw video or decord -- it reads the already-decoded JPEG
frame cache produced by preprocess_frames.py and writes new augmented
"videos" into a sibling cache directory, in the EXACT same on-disk schema:

    <out_root>/<video_id>_aug<k>/
        000000.jpg, 000001.jpg, ...
        metadata.json

Because dataset.py's discover_videos() just walks a root for
<name>/metadata.json, these augmented directories are picked up as ordinary
videos the moment out_root is added to config.yaml's decoded_frame_root list
-- no changes to dataset.py / train.py are needed.

Per-video-variant pipeline
---------------------------
  1. speed retime   -- resample the frame timeline by a random factor (linear
                        blend between neighbouring source frames), rescaling
                        cycle_starts/cycle_ends accordingly. count is
                        invariant under a pure retime.
  2. camera rotation -- one constant tilt angle for the whole variant.
  3. camera jitter    -- a smoothed random-walk 2-D translation trajectory
                        (simulated handheld shake), one (dx, dy) per output
                        frame.
Rotation + jitter are combined into a single per-frame affine warp
(cv2.warpAffine, BORDER_REFLECT_101 so there are no black wedges). Each
effect is independently gated by its own probability, so variants range from
speed-only to fully stacked.

Example
-------
    python augment_repcount.py \\
        --source_root cache/decoded_frames/repcount_notrim/train \\
        --out_root    cache/decoded_frames/repcount_aug/train \\
        --variants_per_video 2 --num_workers 8
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import multiprocessing
import os
import random
import warnings
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))


# ── frame cache discovery ──────────────────────────────────────────────────────

def load_reference_steps_per_rep(root: str) -> np.ndarray:
    """
    Scan a decoded frame cache and return every video's steps_per_rep
    (num_frames / count) as a numpy array -- the empirical distribution
    sample_variant_params' speed_mode="countix_matched" bootstraps its
    per-variant TARGET from (see that function's docstring).
    """
    vals = []
    if not os.path.isdir(root):
        return np.array(vals)
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "metadata.json")
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            meta = json.load(f)
        n, c = meta.get("num_frames"), meta.get("count")
        if n and c and c > 0:
            vals.append(n / c)
    return np.array(vals, dtype=np.float64)


def discover_source_videos(root: str) -> list[str]:
    """Return every <root>/<name> dir that contains a metadata.json (sorted)."""
    dirs = []
    if not os.path.isdir(root):
        return dirs
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isfile(os.path.join(d, "metadata.json")):
            dirs.append(d)
    return dirs


def load_source_frames(video_dir: str, meta: dict) -> list[np.ndarray]:
    """Load every saved frame as an (H, W, 3) uint8 RGB array, in order."""
    ext = meta.get("image_ext", "jpg")
    n = int(meta["num_frames"])
    frames = []
    for idx in range(n):
        path = os.path.join(video_dir, f"{idx:06d}.{ext}")
        with Image.open(path) as im:
            frames.append(np.array(im.convert("RGB"), dtype=np.uint8))
    return frames


# ── augmentation parameter sampling ────────────────────────────────────────────

def _seed_for(base_seed: int, video_id: str, variant_idx: int) -> int:
    """Deterministic per-(video, variant) seed, independent of Pool.imap_unordered's
    non-deterministic completion order (a single shared RNG would not be)."""
    h = hashlib.sha256(f"{base_seed}:{video_id}:{variant_idx}".encode()).hexdigest()
    return int(h[:16], 16) % (2 ** 32)


def sample_variant_params(rng: random.Random, args: argparse.Namespace,
                           base_steps_per_rep: Optional[float] = None) -> dict:
    """Independently gate each effect by its own probability, so variants range
    from a single effect to all three stacked.

    speed_mode="uniform" (default): the original scheme -- speed_factor drawn
    from a fixed range, independent of this particular source video's own
    pace. Simple, but empirically still left the augmented cache's steps_per_rep
    distribution well to the slow side of Countix's across almost the whole
    range (median 48.9 vs Countix's 23.4), not just the tail -- a flat range
    applied to RepCount's own (much slower, median 75) base distribution can't
    reshape it into Countix's shape, only shift it.

    speed_mode="countix_matched": per-variant TARGET steps_per_rep is
    bootstrapped directly from a real Countix reference sample
    (args.countix_reference, see load_reference_steps_per_rep), and
    speed_factor is solved for directly: speed_factor = base_steps_per_rep /
    target, clipped to args.speed_range. In aggregate this makes the
    augmented cache's resulting steps_per_rep distribution track Countix's
    actual empirical shape (verified in simulation: median 27.8 vs Countix's
    23.4 at speed_range=[0.2,6.0], vs uniform mode's 48.9) instead of hoping a
    fixed range happens to land close. Direction (speed up vs down) is
    whatever the data implies for that (base, target) pair, not a fixed
    speed_up_bias -- Countix isn't uniformly faster than every RepCount video,
    just faster in aggregate.
    """
    speed_factor = 1.0
    if rng.random() < args.prob_speed:
        lo, hi = args.speed_range
        if getattr(args, "speed_mode", "uniform") == "countix_matched":
            if base_steps_per_rep is None:
                pass  # zero-count video, no rate to match against -- leave speed_factor=1.0
            else:
                ref = args.countix_reference
                target = float(ref[rng.randrange(len(ref))])
                speed_factor = min(max(base_steps_per_rep / target, lo), hi)
        elif rng.random() < args.speed_up_bias:
            speed_factor = rng.uniform(1.0, hi)   # speed up -- mimics fast, aerobics-like motion
        else:
            speed_factor = rng.uniform(lo, 1.0)   # slow down

    rotation_deg = 0.0
    if rng.random() < args.prob_rotation:
        rotation_deg = rng.uniform(-args.rotation_deg_range, args.rotation_deg_range)

    jitter_enabled = rng.random() < args.prob_jitter

    return {
        "speed_factor": speed_factor,
        "rotation_deg": rotation_deg,
        "jitter_enabled": jitter_enabled,
    }


# ── speed retiming ─────────────────────────────────────────────────────────────

def retime_frames(frames: list[np.ndarray], speed_factor: float) -> list[np.ndarray]:
    """
    Resample `frames` at `speed_factor` (>1 = speed up / fewer output frames,
    <1 = slow down / more output frames).

    speed_factor > 1 (speed-up): NEAREST-frame selection, not blending. Two
    source frames a full frame-interval apart, alpha-blended together,
    produces a visible ghosting/double-image artifact -- a different visual
    signature from genuine motion blur, which comes from a camera's own
    exposure integration during capture, not post-hoc frame averaging. Real
    fast footage has sharp individual frames spaced further apart in time,
    not blended ones. Empirically this mattered: a model trained on the
    blended version of this augmentation transferred WORSE to real fast
    video (Countix) than plain, un-augmented RepCount training did, despite
    having genuine steps_per_rep coverage down to single digits -- nearest-
    frame sampling is meant to fix that mismatch by making the synthetic
    "fast" frames look like real fast frames instead of blend artifacts.

    speed_factor < 1 (slow-down): UNCHANGED linear blend between the two
    nearest source frames. Interpolating between adjacent frames to
    manufacture extra in-between frames is a reasonable way to simulate
    slower motion and is not the artifact-prone direction -- only speed-up
    creates the "average of two temporally-distant sharp frames" ghosting
    problem described above.
    """
    n = len(frames)
    if speed_factor == 1.0 or n <= 1:
        return list(frames)

    new_n = max(1, round(n / speed_factor))
    out = []
    if speed_factor > 1.0:
        for i in range(new_n):
            p = i * speed_factor
            idx = min(int(round(p)), n - 1)
            out.append(frames[idx])
    else:
        for i in range(new_n):
            p = i * speed_factor
            lo = min(int(np.floor(p)), n - 1)
            hi = min(lo + 1, n - 1)
            frac = p - lo
            if hi == lo or frac == 0.0:
                out.append(frames[lo])
            else:
                blended = (1.0 - frac) * frames[lo].astype(np.float32) + frac * frames[hi].astype(np.float32)
                out.append(np.round(blended).astype(np.uint8))
    return out


def rescale_cycles(starts: list[int], ends: list[int], speed_factor: float, new_num_frames: int):
    """Rescale cycle boundaries to the retimed timeline: frame f in the source
    lands at f / speed_factor in the output."""
    new_starts = [int(np.clip(round(s / speed_factor), 0, new_num_frames - 1)) for s in starts]
    new_ends = [int(np.clip(round(e / speed_factor), 0, new_num_frames - 1)) for e in ends]
    return new_starts, new_ends


# ── camera rotation + jitter ────────────────────────────────────────────────────

def build_jitter_trajectory(np_rng: np.random.Generator, n_frames: int, smooth_sigma: float,
                             amplitude_px: float) -> np.ndarray:
    """
    Smoothed random-walk 2-D translation trajectory (simulated handheld shake):
    cumulative Gaussian noise, low-pass filtered, mean-removed (no net drift),
    peak-normalised and scaled to amplitude_px so it can't drift off-frame.
    """
    if amplitude_px <= 0.0 or n_frames <= 1:
        return np.zeros((n_frames, 2), dtype=np.float32)

    raw = np_rng.normal(size=(n_frames, 2))
    walk = np.cumsum(raw, axis=0)
    walk = walk - walk.mean(axis=0, keepdims=True)
    smoothed = gaussian_filter1d(walk, sigma=smooth_sigma, axis=0, mode="nearest")
    peak = float(np.abs(smoothed).max())
    if peak > 1e-6:
        smoothed = smoothed / peak
    return (smoothed * amplitude_px).astype(np.float32)


def warp_frame(frame: np.ndarray, theta_deg: float, dx: float, dy: float) -> np.ndarray:
    """Combined rotation (about frame center) + translation, one affine warp.
    Reflect-padded so rotation/translation never introduces black wedges."""
    if theta_deg == 0.0 and dx == 0.0 and dy == 0.0:
        return frame
    h, w = frame.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), theta_deg, 1.0)
    m[0, 2] += dx
    m[1, 2] += dy
    return cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


# ── per-video-variant generation ────────────────────────────────────────────────

def augment_video_variant(video_dir: str, meta: dict, out_root: str, variant_idx: int,
                           args: argparse.Namespace) -> tuple[str, bool, str]:
    video_id = meta["video_id"]
    out_name = f"{video_id}_aug{variant_idx}"
    out_dir = os.path.join(out_root, out_name)
    meta_path = os.path.join(out_dir, "metadata.json")
    if os.path.isfile(meta_path) and not args.overwrite:
        return out_name, True, "skipped_existing"

    seed = _seed_for(args.seed, video_id, variant_idx)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    n0, c0 = meta.get("num_frames"), meta.get("count")
    base_steps_per_rep = (n0 / c0) if (n0 and c0 and c0 > 0) else None
    params = sample_variant_params(rng, args, base_steps_per_rep)
    speed_factor = params["speed_factor"]
    theta_deg = params["rotation_deg"]

    frames = load_source_frames(video_dir, meta)
    retimed = retime_frames(frames, speed_factor)
    new_num_frames = len(retimed)

    cycle_starts = meta.get("cycle_starts") or []
    cycle_ends = meta.get("cycle_ends") or []
    new_starts, new_ends = rescale_cycles(cycle_starts, cycle_ends, speed_factor, new_num_frames)
    if any(e <= s for s, e in zip(new_starts, new_ends)):
        return out_name, False, "skipped_degenerate"

    frame_size = int(meta.get("frame_width") or retimed[0].shape[1])
    jitter_amp_px = args.jitter_px_frac * frame_size if params["jitter_enabled"] else 0.0
    trajectory = build_jitter_trajectory(np_rng, new_num_frames, args.jitter_smooth_frames, jitter_amp_px)

    os.makedirs(out_dir, exist_ok=True)
    ext = args.image_ext
    for i, frame in enumerate(retimed):
        dx, dy = float(trajectory[i, 0]), float(trajectory[i, 1])
        warped = warp_frame(frame, theta_deg, dx, dy)
        fname = os.path.join(out_dir, f"{i:06d}.{ext}")
        img = Image.fromarray(warped)
        if ext == "jpg":
            img.save(fname, quality=args.jpeg_quality)
        else:
            img.save(fname)

    new_meta = dict(meta)
    new_meta.update({
        "video_id": out_name,
        "dataset_name": "repcount_aug",
        "num_frames": new_num_frames,
        "count": meta.get("count", 0),           # invariant under a pure retime
        "cycle_starts": new_starts,
        "cycle_ends": new_ends,
        "image_ext": ext,
        "jpeg_quality": args.jpeg_quality if ext == "jpg" else None,
        "source_video_id": video_id,
        "augmentation": {
            "speed_factor": speed_factor,
            "rotation_deg": theta_deg,
            "jitter_amplitude_px": jitter_amp_px,
            "jitter_smooth_frames": args.jitter_smooth_frames,
            "variant_idx": variant_idx,
            "seed": seed,
        },
    })
    with open(meta_path, "w") as f:
        json.dump(new_meta, f, indent=2)
    return out_name, True, "ok"


def _augment_record(record: tuple, static_kwargs: dict) -> tuple[str, bool, str]:
    """multiprocessing.Pool worker: unpack one (video_dir, variant_idx) work item.
    Module-level (not a closure) so it's picklable for the pool's IPC."""
    video_dir, variant_idx = record
    video_id = os.path.basename(video_dir.rstrip("/"))
    try:
        with open(os.path.join(video_dir, "metadata.json")) as f:
            meta = json.load(f)
        return augment_video_variant(video_dir, meta, static_kwargs["out_root"], variant_idx,
                                      static_kwargs["args"])
    except Exception as e:  # keep one bad video from killing the whole run
        warnings.warn(f"[{video_id}_aug{variant_idx}] failed: {e}")
        return f"{video_id}_aug{variant_idx}", False, "failed"


# ── main ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source_root", default="cache/decoded_frames/repcount_notrim/train",
                   help="Decoded RepCount frame cache to read from (relative to this script's dir, "
                        "or absolute).")
    p.add_argument("--out_root", default="cache/decoded_frames/repcount_aug/train",
                   help="Where <video_id>_aug<k>/ dirs are written (relative to this script's dir, "
                        "or absolute).")
    p.add_argument("--variants_per_video", type=int, default=2)
    p.add_argument("--speed_mode", choices=["uniform", "countix_matched"], default="uniform",
                   help="uniform (default): speed_factor drawn from --speed_range independent of "
                        "this video's own pace -- a flat range applied to RepCount's own (much "
                        "slower) base distribution can only shift it, not reshape it into Countix's "
                        "actual shape (verified: still median 48.9 vs Countix's 23.4, off by ~2.1x, "
                        "even after widening the range and biasing toward speed-up). "
                        "countix_matched: per-variant TARGET steps_per_rep is bootstrapped from "
                        "--countix_reference_root's own real distribution, and speed_factor is "
                        "solved for directly (base_steps_per_rep / target, clipped to --speed_range) "
                        "-- makes the resulting augmented distribution track Countix's actual shape "
                        "instead of hoping a fixed range lands close (simulated: median 27.8 vs "
                        "23.4 at speed_range=[0.2,6.0]). Use a wider --speed_range with this mode "
                        "than the uniform-mode default -- see that flag's help.")
    p.add_argument("--countix_reference_root", default="cache/decoded_frames/countix_density_nopad/train",
                   help="Only used when --speed_mode=countix_matched: decoded frame cache to "
                        "bootstrap per-variant target steps_per_rep from (see "
                        "load_reference_steps_per_rep).")
    p.add_argument("--speed_range", type=float, nargs=2, default=[0.6, 2.5], metavar=("LO", "HI"),
                   help="Speed-factor range (1.0 = unchanged). >1 speeds up, <1 slows down. In "
                        "uniform mode this is drawn from directly; in countix_matched mode it's the "
                        "clip bound on the solved-for speed_factor -- [0.2, 6.0] is what the "
                        "simulation above used (6x is already fairly extreme for nearest-frame "
                        "sampling -- most information is dropped at that ratio -- so don't push it "
                        "much further just to chase a tighter median match). Default [0.6, 2.5] is "
                        "tuned for uniform mode; widen it explicitly when using countix_matched.")
    p.add_argument("--speed_up_bias", type=float, default=0.85,
                   help="uniform mode only. Probability of sampling from the speed-UP half [1.0, HI] "
                        "rather than the slow-down half [LO, 1.0) -- mimics Countix's faster "
                        "repetition rate. Ignored in countix_matched mode, where direction is "
                        "whatever the (base, target) pair implies, not a fixed bias -- Countix isn't "
                        "uniformly faster than every RepCount video, just faster in aggregate.")
    p.add_argument("--rotation_deg_range", type=float, default=6.0,
                   help="Symmetric +/- degrees for the constant per-variant camera tilt. Lowered "
                        "from an earlier 12.0 default -- that magnitude measurably hurt subtle/"
                        "localized-motion Countix classes (e.g. playing ukulele, shaking head) "
                        "in an ablation, for only a marginal gain on fast full-body-motion classes.")
    p.add_argument("--jitter_px_frac", type=float, default=0.015,
                   help="Handheld-shake amplitude, as a fraction of frame size. Lowered from an "
                        "earlier 0.03 default -- same reasoning as rotation_deg_range above.")
    p.add_argument("--jitter_smooth_frames", type=float, default=8.0,
                   help="Gaussian smoothing sigma (frames) for the shake trajectory -- larger = "
                        "slower, more sweeping camera drift; smaller = jerkier shake.")
    p.add_argument("--prob_speed", type=float, default=0.9,
                   help="Raised from an earlier 0.6 -- more variants get speed-perturbed, same "
                        "reasoning as speed_range/speed_up_bias above.")
    p.add_argument("--prob_rotation", type=float, default=0.3,
                   help="Lowered from an earlier 0.5 default, same reasoning as rotation_deg_range.")
    p.add_argument("--prob_jitter", type=float, default=0.3,
                   help="Lowered from an earlier 0.5 default, same reasoning as rotation_deg_range.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--jpeg_quality", type=int, default=95)
    p.add_argument("--image_ext", default="jpg", choices=["jpg", "png"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help="Augment at most this many SOURCE videos (0 = all), applied before variant "
                        "expansion. Useful for smoke tests.")
    return p.parse_args()


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def main() -> None:
    args = parse_args()
    source_root = _resolve(args.source_root)
    out_root = _resolve(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    args.countix_reference = None
    if args.speed_mode == "countix_matched":
        ref_root = _resolve(args.countix_reference_root)
        args.countix_reference = load_reference_steps_per_rep(ref_root)
        if len(args.countix_reference) == 0:
            raise FileNotFoundError(
                f"--speed_mode=countix_matched but no videos with a usable metadata.json "
                f"found under --countix_reference_root={ref_root}")
        print(f"[augment] speed_mode=countix_matched: bootstrapping targets from "
              f"{len(args.countix_reference)} videos under {ref_root} "
              f"(median steps_per_rep={np.median(args.countix_reference):.1f})")

    video_dirs = discover_source_videos(source_root)
    if args.limit > 0:
        video_dirs = video_dirs[:args.limit]
    if not video_dirs:
        raise FileNotFoundError(f"No decoded videos with metadata.json under {source_root}")

    work_items = [(d, k) for d in video_dirs for k in range(args.variants_per_video)]
    print(f"[augment] {len(video_dirs)} source videos under {source_root}, "
          f"{args.variants_per_video} variant(s) each -> {len(work_items)} total -> {out_root}")

    static_kwargs = {"out_root": out_root, "args": args}
    n_ok = n_fail = n_skip_deg = n_skip_exist = 0

    def _tally(status: str) -> None:
        nonlocal n_ok, n_fail, n_skip_deg, n_skip_exist
        if status == "ok":
            n_ok += 1
        elif status == "skipped_existing":
            n_skip_exist += 1
        elif status == "skipped_degenerate":
            n_skip_deg += 1
        else:
            n_fail += 1

    if args.num_workers <= 1:
        for record in tqdm(work_items, desc="augment repcount"):
            _, _, status = _augment_record(record, static_kwargs)
            _tally(status)
    else:
        worker_fn = functools.partial(_augment_record, static_kwargs=static_kwargs)
        with multiprocessing.Pool(args.num_workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
            for _, _, status in tqdm(pool.imap_unordered(worker_fn, work_items), total=len(work_items),
                                      desc="augment repcount"):
                _tally(status)

    print(f"[augment] done: {n_ok} ok, {n_fail} failed, {n_skip_deg} skipped (degenerate cycles), "
          f"{n_skip_exist} skipped (already exist) -> {out_root}")


if __name__ == "__main__":
    main()
