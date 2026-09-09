"""
Evaluate a trained frame-pipeline checkpoint on decoded videos.

Runs full-video sliding-window inference (utils.sliding_window_inference) over a
validation frame cache and reports / plots:

  * Overall metrics: MAE (alpha=0 and alpha=0.1), RMSE, OBO, Accuracy
  * KL divergence between predicted and GT count distributions
  * Per-class breakdown (from metadata.json class_label)
  * Count distribution plot        -> count_distributions.png
  * Class distribution + per-class MAE -> class_distribution.png
  * Predicted vs GT density maps (when use_density) -> density_maps.png
  * Predicted vs GT in-repetition maps (when use_in_repetition) -> in_repetition_maps.png
  * Per-video CSV                    -> per_video_results.csv
  * Per-window CSV                   -> per_window_results.csv

Example:
    python evaluate.py --config config.yaml \\
        --checkpoint runs/videomae_frames/best.pt \\
        --val_frame_root /data/decoded_frames/repcount_val \\
        --out_dir eval_results
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import RepetitionCounter  # noqa: E402

from dataset import FrameVideoDataset, collate_videos
from frame_encoder import build_encoder
from utils import (
    compute_in_repetition_metrics,
    compute_metrics,
    compute_window_starts,
    kl_divergence_of_count_distributions,
    load_checkpoint,
    load_config,
    plot_class_distribution,
    plot_count_distributions,
    plot_density_grid,
    plot_in_repetition_grid,
    save_config,
    sliding_window_inference,
)


def count_cycles_in_window(cycle_starts, cycle_ends, raw_start: float, raw_end: float) -> int:
    """
    Count of annotated repetition cycles whose CENTER falls inside the raw
    (un-subsampled) frame range [raw_start, raw_end) -- mirrors
    dataset.build_clip_density's "keep cycles whose center falls inside the
    crop" rule, applied here to a sliding-window eval window instead of a
    training crop. 0 for videos with no cycle annotations (e.g. Countix).
    """
    return sum(1 for s, e in zip(cycle_starts, cycle_ends) if raw_start <= (s + e) / 2.0 < raw_end)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="Defaults to <checkpoint's dir>/config.yaml (the snapshot train.py saves "
                        "there), falling back to this script's own config.yaml if that's missing.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val_frame_root", nargs="+", default=None,
                   help="Decoded-frame root(s). Defaults to config val_frame_root.")
    p.add_argument("--clip_length", type=int, default=None)
    p.add_argument("--window_stride", type=int, default=None)
    p.add_argument("--temporal_stride", type=int, default=None)
    p.add_argument("--aggregation", choices=["mean", "median"], default=None,
                   help="Defaults to config validation_window_aggregation ('mean').")
    p.add_argument("--out_dir", default="eval_results")
    p.add_argument("--num_density_plots", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_model_and_encoder(cfg, checkpoint, device):
    enc_kwargs = dict(freeze=True)
    if cfg["encoder"] == "videomae":
        enc_kwargs.update(model_name=cfg["videomae_backbone"],
                          pooling=cfg.get("encoder_pooling", "mean"),
                          spatial_grid=cfg.get("encoder_spatial_grid", 4),
                          return_intermediate=cfg.get("encoder_return_intermediate"),
                          interp_time=cfg.get("encoder_interp_time", True))
    else:
        enc_kwargs.update(model_name=cfg.get("videomae_backbone"))
    encoder = build_encoder(cfg["encoder"], **enc_kwargs).to(device).eval()

    future_cfg = cfg.get("future_prediction", {}) or {}
    model = RepetitionCounter(
        dropout=cfg["dropout"],
        spatial_pool=cfg.get("spatial_pool", False), feature_dim=encoder.feature_dim,
        # must match the checkpoint's architecture exactly (density head /
        # future_head presence changes state_dict keys) -- read from the
        # same config keys train.py writes, not just the always-present ones above.
        use_in_repetition=cfg.get("use_in_repetition", False),
        in_rep_hidden_dim=cfg.get("in_rep_hidden_dim", 512),
        in_rep_dropout=cfg.get("in_rep_dropout"),
        use_count_head=(cfg.get("count_head", {}) or {}).get("enabled", False),
        count_head_mode=(cfg.get("count_head", {}) or {}).get("mode", "per_video"),
        count_head_dropout=(cfg.get("count_head", {}) or {}).get("dropout", 0.25),
        use_density=cfg.get("use_density", True),
        density_hidden_1=cfg.get("density_hidden_1", 64),
        density_hidden_2=cfg.get("density_hidden_2", 64),
        use_future_pred=future_cfg.get("enabled", False),
        future_horizon=future_cfg.get("horizon", 4),
        use_dilated_tcn=cfg.get("use_dilated_tcn", False),
        tcn_kernel_size=cfg.get("tcn_kernel_size", 5),
        tcn_dilations=tuple(cfg.get("tcn_dilations", (1, 2, 4, 8, 16))),
        tcn_expansion=cfg.get("tcn_expansion", 1),
        tcn_dim=cfg.get("tcn_dim", 256),
        use_temporal_transformer=cfg.get("use_temporal_transformer", False),
        transformer_nhead=cfg.get("transformer_nhead", 8),
        transformer_num_layers=cfg.get("transformer_num_layers", 5),
        transformer_dim_feedforward=cfg.get("transformer_dim_feedforward"),
        transformer_clip_length=cfg.get("transformer_clip_length", cfg["clip_length"]),
    ).to(device).eval()
    load_checkpoint(checkpoint, model, map_location=device)
    return encoder, model


def resolve_config_path(config_arg: str | None, checkpoint_path: str) -> str:
    """
    --config resolution: an explicit --config always wins. Otherwise, look
    for the config.yaml train.py snapshots alongside the checkpoint (see
    train.py's save_config call -- runs/xxxx/config.yaml is the exact
    effective config that run actually used), which avoids a structural
    mismatch (use_in_repetition/use_dilated_tcn/etc. changing the
    state_dict shape) against this script's own default config.yaml, which
    may have changed since the checkpoint was trained. Falls back to that
    default only if no sibling config.yaml exists (e.g. older checkpoints
    trained before this snapshotting existed).
    """
    if config_arg is not None:
        return config_arg
    sibling = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "config.yaml")
    if os.path.isfile(sibling):
        print(f"[evaluate] --config not given -- using the checkpoint's own config: {sibling}")
        return sibling
    default = os.path.join(os.path.dirname(__file__), "config.yaml")
    print(f"[evaluate] --config not given and no config.yaml found next to the checkpoint "
          f"({sibling}) -- falling back to {default}, which may not match this checkpoint's "
          f"architecture.")
    return default


def main() -> None:
    args = parse_args()
    args.config = resolve_config_path(args.config, args.checkpoint)
    cfg = load_config(args.config)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    clip_length     = args.clip_length     or cfg["clip_length"]
    window_stride   = args.window_stride   or cfg["validation_window_stride"]
    temporal_stride = args.temporal_stride or cfg["temporal_stride"]
    aggregation     = args.aggregation     or cfg.get("validation_window_aggregation", "mean")
    roots           = args.val_frame_root  or cfg["val_frame_root"]

    eval_cfg = dict(cfg)
    eval_cfg.update({
        "clip_length": clip_length,
        "validation_window_stride": window_stride,
        "temporal_stride": temporal_stride,
        "val_frame_root": roots,
        "checkpoint": args.checkpoint,
    })
    save_config(eval_cfg, os.path.join(args.out_dir, "config.yaml"))

    encoder, model = build_model_and_encoder(cfg, args.checkpoint, device)
    use_amp = device.type == "cuda"
    temporal_downsample = getattr(encoder, "temporal_downsample", 1)

    ds = FrameVideoDataset(roots, image_size=encoder.image_size,
                           pixel_mean=encoder.pixel_mean, pixel_std=encoder.pixel_std,
                           sigma_factor=cfg["sigma_factor"],
                           density_kernel=cfg.get("density_kernel", "gaussian"),
                           temporal_stride=temporal_stride,
                           resize_mode=cfg.get("resize_mode", "squash"),
                           temporal_downsample=temporal_downsample)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_videos,
                        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"])

    # ── inference ──────────────────────────────────────────────────────────────
    use_in_repetition = cfg.get("use_in_repetition", False)
    use_count_head = (cfg.get("count_head", {}) or {}).get("enabled", False)
    use_density = cfg.get("use_density", True)
    rows, window_rows, density_samples, in_rep_samples = [], [], [], []
    in_rep_logits_all, in_rep_targets_all = [], []
    for batch in tqdm(loader, desc="evaluate"):
        s = batch[0]
        result = sliding_window_inference(
            encoder, model, s["frames"], clip_length, window_stride, device, use_amp,
            return_in_rep=use_in_repetition,
            return_count_head=use_count_head,
            temporal_downsample=temporal_downsample,
            aggregation=aggregation,
        )
        if use_count_head:
            *result, full_count_head = result
            result = result[0] if len(result) == 1 else tuple(result)
        else:
            full_count_head = None
        if use_in_repetition:
            full_density, full_in_rep = result
        else:
            full_density, full_in_rep = result, None

        if full_density is not None:
            full_density = full_density.clamp(min=0)
            pred_count = float(full_density.sum().item())
        else:
            # count_head-only model (use_density=False): CountHead's own
            # prediction IS the count, no per-frame curve to reconstruct from.
            pred_count = float(full_count_head.clamp(min=0).item())
            full_density = torch.zeros_like(s["density"])
        gt_count   = float(s["count"].item())
        rows.append({"video_id": s["video_id"], "class": s["class"],
                     "gt": gt_count, "pred": pred_count, "T": s["frames"].shape[0]})

        # ── per-window breakdown ──────────────────────────────────────────────
        # Re-derives the same window starts sliding_window_inference used
        # internally (compute_window_starts is a pure function of N/clip_length/
        # window_stride) -- cheap (no forward pass), so no need to change that
        # function's return signature just to expose this.
        N = s["frames"].shape[0]
        for window_index, start in enumerate(compute_window_starts(N, clip_length, window_stride)):
            valid = min(clip_length, N - start)
            end = start + valid
            reps_in_window = count_cycles_in_window(
                s["cycle_starts"], s["cycle_ends"],
                start * temporal_stride, end * temporal_stride,
            )
            window_rows.append({
                "video_id": s["video_id"],
                "window_index": window_index,
                "gt_window_count": float(s["density"][start:end].sum().item()),
                "pred_window_count": float(full_density[start:end].sum().item()),
                "valid_frames": valid,
                "repetitions_per_window": reps_in_window,
            })
        if use_density and len(density_samples) < args.num_density_plots:
            # Only a real per-frame curve is worth plotting -- when use_density=False
            # full_density above is a torch.zeros_like(...) placeholder (so pred_count
            # can still be reported / count_cycles_in_window above still works), not
            # an actual prediction, so it's excluded here rather than plotted as if it
            # were meaningful.
            density_samples.append({
                "video_id": s["video_id"], "class": s["class"],
                "gt_density": s["density"], "pred_density": full_density,
                "gt_count": gt_count, "pred_count": pred_count,
            })
        if full_in_rep is not None:
            in_rep_logits_all.append(full_in_rep)
            in_rep_targets_all.append(s["phase_mask"])
            if len(in_rep_samples) < args.num_density_plots:
                in_rep_samples.append({
                    "video_id": s["video_id"], "class": s["class"],
                    "gt_in_rep": s["phase_mask"], "pred_in_rep": full_in_rep.sigmoid(),
                })

    gts   = [r["gt"] for r in rows]
    preds = [r["pred"] for r in rows]

    # ── overall metrics ────────────────────────────────────────────────────────
    m0 = compute_metrics(gts, preds, alpha=0.0)
    m1 = compute_metrics(gts, preds, alpha=0.1)
    kl = kl_divergence_of_count_distributions(gts, preds)
    print("\n" + "=" * 56 + "\nOVERALL\n" + "=" * 56)
    print(f"  n        = {m0['n']}")
    print(f"  MAE α=0  = {m0['MAE']:.4f}   MAE α=0.1 = {m1['MAE']:.4f}")
    print(f"  RMSE     = {m0['RMSE']:.4f}")
    print(f"  OBO      = {m0['OBO']*100:.2f}%   OBO10 = {m0['OBO10']*100:.2f}%   Accuracy = {m0['Accuracy']*100:.2f}%")
    print(f"  KL(gt||pred) = {kl:.4f}")

    # ── in-repetition head metrics ─────────────────────────────────────────────
    if in_rep_logits_all:
        ir_metrics = compute_in_repetition_metrics(
            torch.cat(in_rep_logits_all), torch.cat(in_rep_targets_all)
        )
        print("\n" + "=" * 56 + "\nIN-REPETITION HEAD\n" + "=" * 56)
        print(f"  positive ratio = {ir_metrics['positive_ratio']*100:.2f}%")
        print(f"  precision      = {ir_metrics['precision']*100:.2f}%")
        print(f"  recall         = {ir_metrics['recall']*100:.2f}%")
        print(f"  specificity    = {ir_metrics['specificity']*100:.2f}%")

    # ── per-class breakdown ────────────────────────────────────────────────────
    by_gt, by_pred = defaultdict(list), defaultdict(list)
    for r in rows:
        by_gt[r["class"]].append(r["gt"]); by_pred[r["class"]].append(r["pred"])
    print("\n" + "=" * 56 + "\nPER-CLASS (worst MAE first)\n" + "=" * 56)
    per_class = [(c, compute_metrics(by_gt[c], by_pred[c], alpha=0.1)) for c in by_gt]
    per_class.sort(key=lambda x: -x[1]["MAE"])
    for c, cm in per_class:
        print(f"  {c:<38s} n={cm['n']:4d}  MAE={cm['MAE']:.3f}  "
              f"OBO={cm['OBO']*100:5.1f}%  OBO10={cm['OBO10']*100:5.1f}%  Acc={cm['Accuracy']*100:5.1f}%")
    class_counts = {c: cm["n"]   for c, cm in per_class}
    class_maes   = {c: cm["MAE"] for c, cm in per_class}

    # ── plots ──────────────────────────────────────────────────────────────────
    plot_count_distributions(gts, preds, kl, os.path.join(args.out_dir, "count_distributions.png"))
    plot_class_distribution(class_counts, class_maes, os.path.join(args.out_dir, "class_distribution.png"))
    plot_names = "count_distributions,class_distribution"
    if use_density:
        plot_density_grid(density_samples, os.path.join(args.out_dir, "density_maps.png"))
        plot_names += ",density_maps"
    if in_rep_samples:
        plot_in_repetition_grid(in_rep_samples, os.path.join(args.out_dir, "in_repetition_maps.png"))
        plot_names += ",in_repetition_maps"
    print(f"\nPlots -> {args.out_dir}/{{{plot_names}}}.png")

    # ── per-video CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "per_video_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "class", "gt_count", "pred_count", "abs_error", "T"])
        for r in rows:
            w.writerow([r["video_id"], r["class"], r["gt"], r["pred"], abs(r["pred"] - r["gt"]), r["T"]])
    print(f"Per-video CSV -> {csv_path}")

    # ── per-window CSV ─────────────────────────────────────────────────────────
    window_csv_path = os.path.join(args.out_dir, "per_window_results.csv")
    with open(window_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "window_index", "gt_window_count", "pred_window_count",
                     "valid_frame_count", "repetitions_per_window"])
        for r in window_rows:
            w.writerow([r["video_id"], r["window_index"], r["gt_window_count"], r["pred_window_count"],
                        r["valid_frames"], r["repetitions_per_window"]])
    print(f"Per-window CSV -> {window_csv_path}")


if __name__ == "__main__":
    main()
