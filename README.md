# FutureCounts

Video repetition counting via per-frame density regression over a frozen
VideoMAE encoder, with an auxiliary future-embedding-prediction task that
shapes the shared temporal representation.

<img width="6000" height="1496" alt="image" src="https://github.com/user-attachments/assets/2b8e8861-0e57-48ce-bca2-ac846c5fc8a1" />


Auxiliary heads (each independently switchable in `config.yaml`, off by
default except where noted) share the same temporal bottleneck and are
trained jointly with the density head:

- **InRepetitionHead** — per-frame binary classifier (inside vs. outside an
  annotated repetition), BCE/focal loss against the cycle-annotation mask.
- **FuturePredictionHead** — predicts the frozen encoder's own embedding
  `horizon` steps ahead (optionally as a residual), trained with a
  cosine/smooth-L1 loss against the real future embedding. This is the
  representation-shaping signal the repo is named for.
- **CountHead** — a small MLP regressed directly against the ground-truth
  count, independent of the density map.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.14.1 pyyaml==6.0.3 tqdm==4.68.0 tensorboard==2.21.0 \
            numpy==2.5.1 pillow==12.3.0 opencv-python-headless==5.0.0.93 scipy==1.18.0
```

## Data

Training/evaluation read a **decoded-frame cache**, not raw video: one
directory per video, holding sequentially-numbered JPEG frames plus a
`metadata.json` (`video_id`, `count`, `cycle_starts`/`cycle_ends`,
`class_label`). `dataset.py`'s `discover_videos()` just walks a root
directory for `<name>/metadata.json`, so any root with that layout works.
The decoding step itself (raw video -> this cache) isn't included here —
build it from RepCount / Countix with your own decoder that writes to this
schema.

`augment_repcount.py` synthesizes additional training variants from an
already-decoded cache — random playback-speed retiming, camera tilt, and
smoothed handheld-style jitter — to close the domain gap between RepCount's
static tripod footage and more varied real-world video:

```bash
python augment_repcount.py \
    --source_root cache/decoded_frames/repcount/train \
    --out_root    cache/decoded_frames/repcount_aug/train \
    --variants_per_video 2 --num_workers 8
```

Point `decoded_frame_root` / `val_frame_root` in `config.yaml` at one or
more such directories (list several to combine datasets).

## Training

```bash
python train.py --config config.yaml
# common overrides:
python train.py --config config.yaml --epochs 100 --batch_size 8 --save_dir runs/my_run
```

Single GPU by default; multi-GPU via `torchrun --nproc_per_node=N train.py --config config.yaml`.
Every run snapshots its exact effective config to `<save_dir>/config.yaml`
(post CLI-override merge) — always evaluate a checkpoint against that
snapshot, not a possibly-since-changed `config.yaml`, since several options
(`use_in_repetition`, `use_dilated_tcn`, `tcn_dim`, ...) change the model's
`state_dict` shape.

## Evaluation

```bash
python evaluate.py --checkpoint runs/my_run/best.pt --out_dir eval_results
```

`--config` defaults to the checkpoint's own snapshot
(`<checkpoint's dir>/config.yaml`), so it's usually enough to just point
`--checkpoint` at a run. Reports MAE/RMSE/OBO/Accuracy (overall and
per-class), a count-distribution KL divergence, and per-video/per-window
CSVs; add `--val_frame_root` to evaluate against a different split than the
one the run trained against.

## Reference configs

`configs/` holds the exact configs behind two reported results (metrics
from a full training run, evaluated with `evaluate.py`):

| config | eval set | n | MAE | RMSE | OBO | Acc |
|---|---|---|---|---|---|---|
| `repcount_notrim_1024tcn.yaml` | RepCount test | 152 | 0.170 | 4.80 | 52.0% | 28.3% |
| `repcountaug2_prematch_baseline_dilation_0.7relative.yaml` | Countix test | 2455 | 0.357 | 4.06 | 47.7% | 26.5% |

MAE uses `alpha=0.1` (`|pred-gt| / (gt+0.1)`, averaged); OBO is the
off-by-one accuracy (`|pred-gt| <= 1`). Both configs assume their
`decoded_frame_root`/`val_frame_root`/`cached_feature_root` paths exist
locally (see Data above) — edit those paths for your own cache layout
before training.
