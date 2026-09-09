"""
Train the repetition counter with an ONLINE frozen VideoMAE encoder over the
decoded-frame cache.

Pipeline per iteration:
    random-temporal-crop clip (B,L,3,H,W)          [dataset.FrameClipDataset]
        -> frozen VideoMAE encoder -> (B,L,D)       [frame_encoder]
        -> RepetitionCounter (TSSM+Conv+Transformer+MLP) -> density (B,L)
        -> CombinedLoss = MSE(density) + lambda*MAE(count)

The encoder is frozen and lives OUTSIDE the trainable model, so only the
downstream network receives gradients. Validation uses sliding-window inference
over full videos (see utils.sliding_window_inference).

Single GPU:
    python train.py --config config.yaml

Multi-GPU (DDP):
    torchrun --nproc_per_node=4 train.py --config config.yaml

Features: AMP, TensorBoard, checkpoint/resume, early stopping, tqdm, weighted
sampler, async prefetch (pin_memory + persistent + prefetch_factor workers).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import RepetitionCounter, unpack_output, unpack_future, unpack_in_repetition, unpack_count_head  # noqa: E402  (parent project, reused unchanged)

from dataset import CachedFeatureClipDataset, FrameClipDataset, FrameVideoDataset, collate_videos
from frame_encoder import build_encoder
from loss import CombinedLoss
from utils import (
    FullVideoPrefetcher,
    FullVideoSplitSampler,
    compute_metrics,
    load_checkpoint,
    load_config,
    make_count_sampler,
    make_zero_upweight_sampler,
    merge_overrides,
    plot_density,
    save_checkpoint,
    save_config,
    seed_worker,
    set_seed,
    sliding_window_inference,
    sliding_window_train_count,
)


# ── CLI ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    # Common overrides (None -> fall back to config). Add more as needed.
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--clip_length", type=int, default=None)
    p.add_argument("--temporal_stride", type=int, default=None)
    p.add_argument("--videomae_backbone", default=None)
    p.add_argument("--encoder", default=None)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--resume", default=None,
                   help="Continue an interrupted run: restores model, optimizer, scheduler, "
                        "scaler, epoch, and best_metric (early-stopping baseline) from this "
                        "checkpoint.")
    p.add_argument("--num_workers", type=int, default=None)
    return p.parse_args()


# ── DDP helpers ────────────────────────────────────────────────────────────────

def setup_ddp() -> tuple[bool, int, int, int]:
    """Return (is_ddp, rank, local_rank, world_size). Honours torchrun env vars."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def is_main(rank: int) -> bool:
    return rank == 0


# ── validation ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(encoder, model, val_ds, cfg, device, writer=None, epoch=0, plot_dir=None) -> dict:
    """Sliding-window full-video inference over the validation set -> metrics."""
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model
    use_count_head = (cfg.get("count_head", {}) or {}).get("enabled", False)
    temporal_downsample = getattr(encoder, "temporal_downsample", 1)
    gts, preds = [], []

    # Capped independently of the training num_workers: the training loader's
    # `persistent_workers` keeps its own worker pool alive across this call, so
    # reusing the same worker count here would briefly run two full pools at
    # once (train's + val's) and can blow past the container's memory limit.
    val_num_workers = min(8, cfg["num_workers"])
    loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=val_num_workers, collate_fn=collate_videos,
                        pin_memory=cfg["pin_memory"])

    plotted = 0
    for batch in tqdm(loader, desc=f"val e{epoch}", leave=False):
        sample = batch[0]
        frames = sample["frames"]
        result = sliding_window_inference(
            encoder, raw_model, frames,
            clip_length=cfg["clip_length"],
            window_stride=cfg["validation_window_stride"],
            device=device, use_amp=cfg["amp"],
            return_count_head=use_count_head,
            temporal_downsample=temporal_downsample,
            aggregation=cfg.get("validation_window_aggregation", "mean"),
        )
        if use_count_head:
            full_density, full_count_head = result
        else:
            full_density, full_count_head = result, None

        if full_density is not None:
            pred_count = float(full_density.clamp(min=0).sum().item())
        else:
            # count_head-only inference (use_density=False): CountHead's own
            # prediction IS the count, no per-frame curve to reconstruct from.
            pred_count = float(full_count_head.clamp(min=0).item())
        gt_count   = float(sample["count"].item())
        preds.append(pred_count)
        gts.append(gt_count)

        if plot_dir and plotted < 8 and full_density is not None:
            out_path = os.path.join(plot_dir, f"e{epoch}_{sample['video_id']}.png")
            title = f"{sample['video_id']}  gt={gt_count:.0f} pred={pred_count:.1f}"
            plot_density(full_density, sample["density"], out_path, title=title)
            plotted += 1
            # else: count_head-only inference (use_density=False) -- no
            # per-frame curve exists to plot, skip silently rather than
            # calling plot_density(None, ...) (would crash on len(None)).

    m = compute_metrics(gts, preds, alpha=0.1)
    if writer is not None:
        for k, v in m.items():
            if k != "n":
                writer.add_scalar(f"val/{k}", v, epoch)
    return m


# ── main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = merge_overrides(cfg, {
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "clip_length": args.clip_length,
        "temporal_stride": args.temporal_stride, "videomae_backbone": args.videomae_backbone,
        "encoder": args.encoder, "save_dir": args.save_dir, "resume": args.resume,
        "num_workers": args.num_workers,
    })

    fvs_cfg     = cfg.get("full_video_sampling", {}) or {}
    fvs_enabled = fvs_cfg.get("enabled", False)
    if fvs_enabled and cfg.get("weighted_sampler", False):
        raise ValueError(
            "full_video_sampling.enabled and weighted_sampler cannot both be true "
            "(unsupported combination) -- disable one."
        )
    if fvs_enabled and cfg.get("zero_count_upweight", 1.0) > 1.0:
        raise ValueError(
            "full_video_sampling.enabled and zero_count_upweight>1.0 cannot both be "
            "set (unsupported combination, same as weighted_sampler) -- disable one."
        )

    use_cached_features = cfg.get("use_cached_features", False)
    if use_cached_features and fvs_enabled:
        raise ValueError(
            "use_cached_features and full_video_sampling.enabled cannot both be true -- "
            "full_video_sampling tiles the whole video into its own windows and calls the "
            "encoder on them directly (see sliding_window_train_count), which does not go "
            "through the cached-feature dataset and would silently re-encode online while "
            "the crop path uses the cache -- disable one."
        )
    if use_cached_features and (cfg.get("crop_mode", "cycle") != "cycle" or cfg.get("cycle_jitter", 0) != 0):
        raise ValueError(
            "use_cached_features requires crop_mode='cycle' and cycle_jitter=0 -- see "
            "dataset.CachedFeatureClipDataset / extract_encoder_cache.py for why."
        )

    is_ddp, rank, local_rank, world_size = setup_ddp()
    if fvs_enabled and is_ddp:
        raise RuntimeError(
            "full_video_sampling.enabled=true does not support DDP (multi-GPU / "
            "torchrun) training. Run single-GPU, or set full_video_sampling.enabled=false."
        )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["random_seed"] + rank, deterministic=cfg.get("deterministic", False))

    # On many-core boxes torch defaults to one intra-op thread per core, which
    # oversubscribes the CPU for what's actually lightweight per-batch glue code
    # (the heavy compute runs on the GPU) and steals cycles from the DataLoader
    # workers doing the real CPU work (JPEG decode + resize).
    torch_threads = cfg.get("torch_threads")
    if torch_threads:
        torch.set_num_threads(torch_threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))

    # DataLoader worker->main IPC defaults to /dev/shm-backed tensors, whose
    # capacity is fixed by the container (df -h /dev/shm) and independent of
    # num_workers/prefetch_factor -- large clips (clip_length * image_size^2)
    # times many workers can exceed it and crash with "unable to allocate
    # shared memory". 'file_system' backs the same IPC with regular tmp files
    # instead, bounded by the fd limit (ulimit -n) rather than shm size.
    torch.multiprocessing.set_sharing_strategy("file_system")
    cfg["amp"] = bool(cfg.get("amp", True)) and device.type == "cuda"

    if is_main(rank):
        os.makedirs(cfg["save_dir"], exist_ok=True)
        # Snapshot the EFFECTIVE config (post CLI-override merge) into the run
        # dir, so evaluate.py (and anyone else) can find the exact config this
        # run actually used at runs/xxxx/config.yaml -- without this, evaluating
        # an old checkpoint against the current (possibly since-changed)
        # frame_pipeline/config.yaml risks a structural mismatch (use_in_repetition/
        # use_dilated_tcn etc. changing the state_dict shape).
        save_config(cfg, os.path.join(cfg["save_dir"], "config.yaml"))
        writer = SummaryWriter(os.path.join(cfg["save_dir"], "tb"))
        plot_dir = os.path.join(cfg["save_dir"], "density_plots")
        os.makedirs(plot_dir, exist_ok=True)
    else:
        writer, plot_dir = None, None

    # ── encoder (frozen, separate module) ──────────────────────────────────────
    enc_kwargs = dict(freeze=cfg.get("freeze_encoder", True))
    if cfg["encoder"] == "videomae":
        enc_kwargs.update(
            model_name=cfg["videomae_backbone"],
            pooling=cfg.get("encoder_pooling", "mean"),
            spatial_grid=cfg.get("encoder_spatial_grid", 4),
            return_intermediate=cfg.get("encoder_return_intermediate"),
            interp_time=cfg.get("encoder_interp_time", True),
        )
    else:
        enc_kwargs.update(model_name=cfg.get("videomae_backbone"))
    encoder = build_encoder(cfg["encoder"], **enc_kwargs).to(device)
    encoder.eval()
    # > 1 only for VideoMAEClipEncoder(interp_time="downsample") -- see
    # frame_encoder.py. Threaded through every dataset/loss/sliding-window
    # call below so GT targets and window-stitching stay aligned with the
    # encoder's actual (possibly reduced) output resolution.
    temporal_downsample = getattr(encoder, "temporal_downsample", 1)

    # ── downstream model (encoder built separately, above) ─────────────────────
    use_in_repetition = cfg.get("use_in_repetition", False)
    use_density = cfg.get("use_density", True)
    future_cfg = cfg.get("future_prediction", {}) or {}
    use_future_pred = future_cfg.get("enabled", False)
    future_horizon = future_cfg.get("horizon", 4)
    lambda_future = future_cfg.get("loss_weight", 0.2)
    future_mode = future_cfg.get("mode", "cosine")
    future_smooth_l1_beta = future_cfg.get("smooth_l1_beta", 1.0)
    future_residual = future_cfg.get("residual", False)
    zero_cfg = cfg.get("zero_count_penalty", {}) or {}
    use_zero_penalty = zero_cfg.get("enabled", False)
    count_head_cfg = cfg.get("count_head", {}) or {}
    use_count_head = count_head_cfg.get("enabled", False)
    count_head_mode = count_head_cfg.get("mode", "per_video")
    model = RepetitionCounter(
        dropout=cfg["dropout"],
        spatial_pool=cfg.get("spatial_pool", False),
        feature_dim=encoder.feature_dim,
        use_in_repetition=use_in_repetition,
        in_rep_hidden_dim=cfg.get("in_rep_hidden_dim", 512),
        in_rep_dropout=cfg.get("in_rep_dropout"),
        use_count_head=use_count_head,
        count_head_mode=count_head_mode,
        count_head_dropout=count_head_cfg.get("dropout", 0.25),
        use_density=use_density,
        density_hidden_1=cfg.get("density_hidden_1", 64),
        density_hidden_2=cfg.get("density_hidden_2", 64),
        use_future_pred=use_future_pred,
        future_horizon=future_horizon,
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
    ).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                    find_unused_parameters=False)

    # ── data ───────────────────────────────────────────────────────────────────
    if use_cached_features:
        # points at extract_encoder_cache.py's output (e.g. cache/encoded_features/
        # countix/train), NOT the raw decoded-frame cache -- decoded_frame_root is
        # left untouched/unused here so it still documents where the raw frames
        # that were encoded actually live.
        train_ds = CachedFeatureClipDataset(
            roots=cfg["cached_feature_root"],
            clip_length=cfg["clip_length"],
            temporal_stride=cfg["temporal_stride"],
            sigma_factor=cfg["sigma_factor"],
            density_kernel=cfg.get("density_kernel", "gaussian"),
            crop_mode=cfg.get("crop_mode", "cycle"),
            cycle_jitter=cfg.get("cycle_jitter", 0),
            seed=cfg["random_seed"] + rank,
            temporal_downsample=temporal_downsample,
        )
    else:
        train_ds = FrameClipDataset(
            roots=cfg["decoded_frame_root"],
            clip_length=cfg["clip_length"],
            temporal_stride=cfg["temporal_stride"],
            image_size=encoder.image_size,
            pixel_mean=encoder.pixel_mean,
            pixel_std=encoder.pixel_std,
            sigma_factor=cfg["sigma_factor"],
            density_kernel=cfg.get("density_kernel", "gaussian"),
            random_crop=True,
            crop_mode=cfg.get("crop_mode", "cycle"),
            cycle_jitter=cfg.get("cycle_jitter", 0),
            seed=cfg["random_seed"] + rank,
            resize_mode=cfg.get("resize_mode", "squash"),
            temporal_downsample=temporal_downsample,
        )

    if is_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
    elif fvs_enabled:
        train_sampler = FullVideoSplitSampler(
            len(train_ds), fraction=fvs_cfg.get("fraction", 0.2), seed=cfg["random_seed"])
        shuffle = False
    elif cfg.get("weighted_sampler", False) or cfg.get("zero_count_upweight", 1.0) > 1.0:
        zero_weight = cfg.get("zero_count_upweight", 1.0)
        bin_strategy = cfg.get("weighted_sampler_bin_strategy", "width")
        if zero_weight > 1.0:
            train_sampler = make_zero_upweight_sampler(
                train_ds, zero_weight=zero_weight,
                rarity_weighted=cfg.get("weighted_sampler", False),
                bin_strategy=bin_strategy,
            )
        else:
            train_sampler = make_count_sampler(train_ds, bin_strategy=bin_strategy)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    # Typed handle so the full-video training step below doesn't need to
    # type-narrow train_sampler (which is DistributedSampler/WeightedRandomSampler/
    # None in the other branches) -- fvs_enabled implies train_sampler IS this.
    full_video_sampler = train_sampler if fvs_enabled else None

    loader_kwargs = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        worker_init_fn=seed_worker,
    )
    if cfg["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = cfg.get("persistent_workers", True)
        loader_kwargs["prefetch_factor"] = cfg.get("prefetch_factor", 4)

    train_loader = DataLoader(train_ds, sampler=train_sampler, shuffle=shuffle, **loader_kwargs)

    full_video_ds = None
    if fvs_enabled:
        full_video_ds = FrameVideoDataset(
            roots=cfg["decoded_frame_root"],
            image_size=encoder.image_size,
            pixel_mean=encoder.pixel_mean,
            pixel_std=encoder.pixel_std,
            sigma_factor=cfg["sigma_factor"],
            density_kernel=cfg.get("density_kernel", "gaussian"),
            temporal_stride=cfg["temporal_stride"],
            resize_mode=cfg.get("resize_mode", "squash"),
            temporal_downsample=temporal_downsample,
        )
        assert full_video_ds.video_dirs == train_ds.video_dirs, (
            "full_video_ds and train_ds must enumerate videos in the same order "
            "(both call discover_videos on the same roots) for index correspondence "
            "to hold between FullVideoSplitSampler.full_video_indices and full_video_ds."
        )

    val_ds = None
    if is_main(rank) and cfg.get("val_frame_root"):
        val_ds = FrameVideoDataset(
            roots=cfg["val_frame_root"],
            image_size=encoder.image_size,
            pixel_mean=encoder.pixel_mean,
            pixel_std=encoder.pixel_std,
            sigma_factor=cfg["sigma_factor"],
            density_kernel=cfg.get("density_kernel", "gaussian"),
            temporal_stride=cfg["temporal_stride"],
            resize_mode=cfg.get("resize_mode", "squash"),
            temporal_downsample=temporal_downsample,
        )

    # ── optim / loss / amp ─────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    warmup_epochs = cfg.get("warmup_epochs", 0)  # 0 (default): unchanged legacy behaviour -- plain
                                                  # CosineAnnealingLR from epoch 1, same schedule/
                                                  # state_dict shape as before this existed. >0: a
                                                  # linear LR warmup (start_factor=0.01 -> 1.0 over
                                                  # `warmup_epochs` epochs) before the cosine decay --
                                                  # added for TransformerTCN runs, which (unlike
                                                  # DilatedTCN) can diverge to NaN a few dozen epochs
                                                  # in without it (a large from-scratch attention
                                                  # stack at a learning_rate tuned for the much
                                                  # smaller conv TCN). Changes scheduler.state_dict()
                                                  # shape (SequentialLR wrapping two schedulers
                                                  # instead of one CosineAnnealingLR) -- a checkpoint
                                                  # saved with one warmup_epochs value can't resume
                                                  # cleanly into a run with a different one --
                                                  # start a fresh run instead of --resume when
                                                  # changing this mid-experiment.
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(cfg["epochs"] - warmup_epochs, 1))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler    = torch.cuda.amp.GradScaler(enabled=cfg["amp"])
    criterion = CombinedLoss(
        lambda_count=cfg["lambda_count"],
        lambda_density=cfg.get("lambda_density", 500),
        lambda_support=cfg.get("lambda_support", 0.0),
        count_mode=cfg.get("count_mode", "relative"),
        # temporal feature/density resolution, not raw frame count -- see
        # loss.CountLoss's fast_rep_weighted docstring ("steps_per_rep").
        clip_length=cfg["clip_length"] // temporal_downsample,
        use_in_repetition=use_in_repetition,
        lambda_in_repetition=cfg.get("weight_in_repetition", 1.0),
        in_rep_loss_mode=cfg.get("in_rep_loss_mode", "focal"),
        in_rep_focal_alpha=cfg.get("in_rep_focal_alpha", 0.7),
        in_rep_focal_gamma=cfg.get("in_rep_focal_gamma", 2.0),
        use_future_pred=use_future_pred,
        lambda_future=lambda_future,
        future_horizon=future_horizon,
        future_mode=future_mode,
        future_smooth_l1_beta=future_smooth_l1_beta,
        future_residual=future_residual,
        use_zero_penalty=use_zero_penalty,
        lambda_zero=zero_cfg.get("loss_weight", 1.0),
        zero_tolerance=zero_cfg.get("tolerance", 0.1),
        zero_count_threshold=zero_cfg.get("count_threshold", 0.0),
        use_count_head=use_count_head,
        lambda_count_head=count_head_cfg.get("loss_weight", 1.0),
        count_head_loss_mode=count_head_cfg.get("loss_mode", "smooth_l1"),
        count_head_loss_beta=count_head_cfg.get("beta", 1.0),
    )

    # ── resume ─────────────────────────────────────────────────────────────────
    start_epoch, best_mae, patience = 1, float("inf"), 0
    if cfg.get("resume"):
        ckpt = load_checkpoint(cfg["resume"], model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_mae    = ckpt.get("best_metric", float("inf"))
        if is_main(rank):
            print(f"[resume] from {cfg['resume']} @ epoch {start_epoch}, best MAE {best_mae:.4f}")

    if is_main(rank):
        n_train = sum(p.numel() for p in trainable)
        print(f"[train] device={device} ddp={is_ddp} world_size={world_size} "
              f"trainable={n_train:,} encoder={cfg['encoder']}:{cfg.get('videomae_backbone')} (frozen)")

    # ── training loop ──────────────────────────────────────────────────────────
    global_step = 0
    debug_once = True

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        agg = {"count": 0.0, "total": 0.0, "grad_norm": 0.0}
        if fvs_enabled:
            agg["full_video_count"] = 0.0
        if use_density:
            agg["density"] = 0.0
            agg["support"] = 0.0
        if use_in_repetition:
            agg["in_repetition"] = 0.0
        if use_future_pred:
            agg["future"] = 0.0
            agg["future_zero_improvement"] = 0.0
        if use_zero_penalty:
            agg["zero"] = 0.0
        if use_count_head:
            agg["count_head"] = 0.0
        n_batches = 0
        n_future_zero_batches = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", disable=not is_main(rank))
        for clip, gt_density, gt_count, _gt_phase_sin, _gt_phase_cos, gt_phase_mask in pbar:
            clip          = clip.to(device, non_blocking=True)          # (B,L,3,H,W)
            gt_density    = gt_density.to(device, non_blocking=True)    # (B,L)
            gt_count      = gt_count.to(device, non_blocking=True)      # (B,)
            gt_phase_mask = gt_phase_mask.to(device, non_blocking=True) # (B,L)

            with torch.autocast(device_type=device.type, enabled=cfg["amp"]):
                # use_cached_features=true: `clip` IS ALREADY the encoder's output
                # (CachedFeatureClipDataset returns precomputed features, not pixels) --
                # skip the (redundant, frozen-so-identical-every-epoch) encoder forward
                # pass entirely. See extract_encoder_cache.py / dataset.CachedFeatureClipDataset.
                feats = clip if use_cached_features else encoder(clip)  # (B,L,D) frozen, no grad

                if debug_once:
                    print("\n===== DEBUG =====")
                    print("clip shape:", clip.shape)
                    print("features shape:", feats.shape)
                    print("features mean:", feats.mean().item())
                    print("features std:", feats.std().item())
                    print("features temporal std:", feats.std(dim=1).mean().item())

                out = (model.module if is_ddp else model).forward_from_features(feats)
                pred_density, pred_count, _ = unpack_output(out)
                pred_in_rep = unpack_in_repetition(out)
                pred_future, target_future = unpack_future(out)
                pred_count_head = unpack_count_head(out)

                if debug_once:
                    if pred_density is not None:
                        print("pred_density shape:", pred_density.shape)
                        print("pred_density min:", pred_density.min().item())
                        print("pred_density max:", pred_density.max().item())
                        print("pred_density mean:", pred_density.mean().item())
                        print("pred_density std:", pred_density.std().item())
                    else:
                        print("pred_density: None (use_density=False, count_head-only inference)")

                    print("gt_density shape:", gt_density.shape)
                    print("gt_density min:", gt_density.min().item())
                    print("gt_density max:", gt_density.max().item())
                    print("gt_density mean:", gt_density.mean().item())
                    print("gt_density std:", gt_density.std().item())

                    print("gt density sum:", gt_density.sum(dim=-1))
                    print("gt count:", gt_count)
                loss, comps = criterion(
                    pred_density, gt_density, gt_count,
                    pred_count=pred_count,
                    phase_mask=gt_phase_mask,
                    pred_in_rep=pred_in_rep,
                    pred_future=pred_future,
                    target_future=target_future,
                    pred_count_head=pred_count_head,
                )

            # Zero-predictor baseline for the future head: residual targets
            # (future_prediction.residual=true) are typically centred near
            # zero, so a shrinking `future` loss can mean the head learned
            # real motion OR that it's just collapsing toward predicting ~0 --
            # a trivial baseline that can score deceptively well on its own.
            # Compares the head's actual prediction against literal zeros on
            # the SAME shifted/residual target the loss trains against (see
            # loss.FuturePredictionLoss.zero_baseline). improvement ~0 or
            # negative -> the head isn't beating "always predict zero"; more
            # lambda_future won't fix that, the head needs to actually change.
            future_zero_improvement = None
            if use_future_pred and pred_future is not None and target_future is not None:
                zb = criterion.future_loss.zero_baseline(pred_future, target_future)
                if zb is not None:
                    zero_loss, model_loss, future_zero_improvement = zb
                    agg["future_zero_improvement"] += future_zero_improvement
                    n_future_zero_batches += 1

            optimizer.zero_grad(set_to_none=True)

            log_this_step = (
                is_main(rank) and writer is not None
                and (global_step + 1) % cfg.get("log_every", 20) == 0
            )

            scaler.scale(loss).backward()
            # Always unscale + compute the (pre-clip) grad norm so it can be logged
            # even when grad_clip is unset -- max_norm=inf makes clip_grad_norm_ a
            # pure norm computation in that case (never actually clips).
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, cfg.get("grad_clip") or float("inf")
            )
            scaler.step(optimizer)
            scaler.update()
            if debug_once:
                debug_once = False
            for k in agg:
                if k == "grad_norm":
                    agg[k] += grad_norm.item()
                elif k in ("future_zero_improvement", "full_video_count"):
                    pass  # accumulated separately (zero_baseline above / full-video step below)
                else:
                    agg[k] += comps[k]
            n_batches += 1
            global_step += 1
            if is_main(rank):
                postfix = {"loss": f"{comps['total']:.3f}", "grad_norm": f"{grad_norm.item():.3f}"}
                if future_zero_improvement is not None:
                    postfix["future_vs_zero"] = f"{future_zero_improvement:.3f}"
                pbar.set_postfix(**postfix)
                if log_this_step:
                    writer.add_scalar("train/loss", comps["total"], global_step)
                    writer.add_scalar("train/count", comps["count"], global_step)
                    writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
                    if "density" in comps:
                        writer.add_scalar("train/density", comps["density"], global_step)
                    if "support" in comps:
                        writer.add_scalar("train/support", comps["support"], global_step)
                    if "in_repetition" in comps:
                        writer.add_scalar("train/in_repetition", comps["in_repetition"], global_step)
                    if "zero" in comps:
                        writer.add_scalar("train/zero", comps["zero"], global_step)
                    if "count_head" in comps:
                        writer.add_scalar("train/count_head", comps["count_head"], global_step)
                    if "future" in comps:
                        writer.add_scalar("train/future", comps["future"], global_step)
                    if future_zero_improvement is not None:
                        # 1 - model_loss/zero_loss vs. an all-zeros prediction on the same
                        # (shifted, residual-if-applicable) target -- ~0 or negative means
                        # the future head isn't beating a trivial zero predictor yet.
                        writer.add_scalar("train/future_vs_zero_improvement", future_zero_improvement, global_step)
                        writer.add_scalar("train/future_zero_loss", zero_loss, global_step)
                        writer.add_scalar("train/future_model_loss_fixed_beta", model_loss, global_step)

        # ── full-video count step (full_video_sampling) ────────────────────────
        # The fraction of videos FullVideoSplitSampler excluded from the normal
        # batched loop above: run the WHOLE video through the same tiling
        # sliding_window_inference uses at eval, stitch per-window density into
        # one full-length curve, sum to a predicted total count, and backprop
        # just the count loss against that video's ground-truth count -- one
        # optimizer step per video, since the stitch needs every window's
        # prediction before a single loss/backward can be computed.
        #
        # FullVideoPrefetcher decodes video i+1's frames on a background thread
        # while the GPU is busy with video i's forward/backward/optimizer step
        # below -- full_video_ds[vi] alone (no DataLoader/num_workers on this
        # path) would otherwise decode every frame of every video synchronously
        # on the main thread, stalling the GPU between videos. Matters most at
        # a high full_video_sampling.fraction, where this loop dominates epoch time.
        n_full_video = 0
        if fvs_enabled:
            window_stride = fvs_cfg.get("window_stride") or cfg["clip_length"]
            raw_model = model.module if is_ddp else model
            for sample in FullVideoPrefetcher(full_video_ds, full_video_sampler.full_video_indices):
                frames = sample["frames"].to(device, non_blocking=True)
                gt_count = sample["count"].to(device).unsqueeze(0)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=cfg["amp"]):
                    pred_count = sliding_window_train_count(
                        encoder, raw_model, frames,
                        clip_length=cfg["clip_length"], window_stride=window_stride,
                        device=device, use_amp=cfg["amp"],
                        temporal_downsample=temporal_downsample,
                    ).unsqueeze(0)
                    fv_loss = criterion.lambda_count * criterion.count_loss(pred_count, gt_count)

                scaler.scale(fv_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, cfg.get("grad_clip") or float("inf"))
                scaler.step(optimizer)
                scaler.update()

                agg["full_video_count"] += fv_loss.item()
                n_full_video += 1
                global_step += 1
                if is_main(rank) and writer is not None:
                    writer.add_scalar("train/full_video_count", fv_loss.item(), global_step)

        scheduler.step()

        if is_main(rank):
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
            msg = (f"Epoch {epoch:03d} | loss={agg['total']/max(n_batches,1):.4f} "
                   f"count={agg['count']/max(n_batches,1):.4f} "
                   f"grad_norm={agg['grad_norm']/max(n_batches,1):.4f}")
            if "density" in agg:
                msg += f" density={agg['density']/max(n_batches,1):.4f}"
            if "support" in agg:
                msg += f" support={agg['support']/max(n_batches,1):.4f}"
            if "in_repetition" in agg:
                msg += f" in_repetition={agg['in_repetition']/max(n_batches,1):.4f}"
            if "zero" in agg:
                msg += f" zero={agg['zero']/max(n_batches,1):.4f}"
            if "future" in agg:
                msg += f" future={agg['future']/max(n_batches,1):.4f}"
            if "future_zero_improvement" in agg and n_future_zero_batches > 0:
                msg += f" future_vs_zero={agg['future_zero_improvement']/n_future_zero_batches:.4f}"
            if "full_video_count" in agg and n_full_video > 0:
                msg += f" full_video_count={agg['full_video_count']/n_full_video:.4f} (n={n_full_video})"

            # ── validation + checkpointing (rank 0) ────────────────────────────
            do_val = val_ds is not None and epoch % cfg.get("val_every", 1) == 0
            if do_val:
                m = validate(encoder, model, val_ds, cfg, device, writer, epoch, plot_dir)
                msg += (f" | val MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f} "
                        f"OBO={m['OBO']*100:.1f}% Acc={m['Accuracy']*100:.1f}%")
                improved = m["MAE"] < best_mae - cfg.get("early_stop_min_delta", 0.0)
                if improved:
                    best_mae = m["MAE"]
                    patience = 0
                    save_checkpoint(os.path.join(cfg["save_dir"], "best.pt"),
                                    model, optimizer, scheduler, scaler, epoch, best_mae)
                    msg += "  *best*"
                else:
                    patience += 1

            save_checkpoint(os.path.join(cfg["save_dir"], "last.pt"),
                            model, optimizer, scheduler, scaler, epoch, best_mae)
            print(msg)

            if val_ds is not None and patience >= cfg.get("early_stop_patience", 10 ** 9):
                print(f"[early stop] no val improvement in {patience} checks. Best MAE={best_mae:.4f}")
                break

    if is_main(rank):
        writer.close()
        print(f"[done] best val MAE={best_mae:.4f} -> {os.path.join(cfg['save_dir'], 'best.pt')}")
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
