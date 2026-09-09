"""
Swappable, frozen clip encoders.

Design goal #4 from the spec: swapping VideoMAE for DINOv2 / InternVideo /
VideoMamba must NOT require touching the downstream RepetitionCounter. Every
encoder therefore obeys one contract:

    forward(clip) : (B, T, 3, H, W)  ->  (B, T, D)     one feature vector per frame

The downstream model (model.RepetitionCounter) consumes exactly that (B, T, D)
via forward_from_features() -- it never builds or calls an encoder itself. So
the encoder is a fully separate module -- VideoMAE is NEVER placed inside the
trainable model, it is wrapped here, frozen, and called online during training.

Why per-FRAME (T out for T in)?
    The density head is per-frame and the TSSM is a T x T self-similarity matrix,
    so downstream code assumes "output position t corresponds to input frame t".
    VideoMAE natively emits T/2 tubelet tokens (its tubelet spans 2 frames); we
    linearly interpolate the temporal axis back to T so the per-frame contract
    holds regardless of backbone. Frame-level models (DINOv2) already emit T.

Freezing
    freeze=True (default) sets requires_grad=False on every backbone parameter
    and forces .eval() so BatchNorm/Dropout stay in inference mode even when the
    parent model is in train() mode. forward() runs under no_grad in that case,
    so no encoder activations are retained for backprop -- this is what makes
    running the encoder online every iteration affordable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _linear_interp_time(seq: torch.Tensor, T: int) -> torch.Tensor:
    """seq: (B, L, [S,] D) -> (B, T, [S,] D) via linear interpolation on the time axis."""
    if seq.shape[1] == T:
        return seq
    B = seq.shape[0]
    extra = seq.shape[2:]                                    # (D,) or (S, D)
    flat = seq.reshape(B, seq.shape[1], -1).transpose(1, 2)   # (B, F, L)
    flat = F.interpolate(flat, size=T, mode="linear", align_corners=False)
    return flat.transpose(1, 2).reshape(B, T, *extra)


# ── base contract ──────────────────────────────────────────────────────────────

class BaseClipEncoder(nn.Module, ABC):
    """
    Abstract frozen clip encoder.

    Subclasses set ``self.feature_dim`` (D) and implement ``_encode`` returning
    ``(B, T, D)`` per-frame features. ``forward`` handles freeze/grad bookkeeping
    so subclasses don't repeat it.

    ``image_size``, ``pixel_mean`` and ``pixel_std`` describe the preprocessing
    the encoder expects; dataset.py reads them so the on-disk frames are
    normalised correctly for whichever backbone is selected.
    """

    image_size: int = 224
    # ImageNet statistics -- shared by VideoMAE and DINOv2.
    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std  = (0.229, 0.224, 0.225)

    # Raw input frames per output token. 1 for every encoder here except
    # VideoMAEClipEncoder(interp_time="downsample"), the only mode that
    # breaks the per-frame T-in/T-out contract (see that class's docstring).
    # Callers that need to build a T-aligned target (GT density, phase, ...)
    # for a clip of T input frames should size it via output_length(T), not
    # by dividing T by this directly (VideoMAE pads to a multiple of its
    # native window size first -- see VideoMAEClipEncoder.output_length).
    temporal_downsample: int = 1

    def __init__(self, freeze: bool = True):
        super().__init__()
        self.freeze = freeze
        self.feature_dim: int = 0  # subclasses must set

    def output_length(self, T: int) -> int:
        """Number of temporal tokens this encoder emits for T input frames.
        Default: T (every encoder here normally honours the per-frame
        contract) -- see VideoMAEClipEncoder.output_length for the one
        override (interp_time="downsample")."""
        return T

    def _apply_freeze(self, module: nn.Module) -> None:
        if self.freeze:
            for p in module.parameters():
                p.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True):  # noqa: D401 - keep frozen backbone in eval
        super().train(mode)
        if self.freeze:
            self._freeze_eval()
        return self

    def _freeze_eval(self) -> None:
        """Force frozen submodules back to eval() after a train() call."""
        for m in self.children():
            m.eval()

    @abstractmethod
    def _encode(self, clip: torch.Tensor) -> torch.Tensor:
        """clip: (B, T, 3, H, W) -> (B, T, D). Called under the right grad ctx."""

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        clip: (B, T, 3, H, W) already normalised -> (B, T, D).

        Runs under no_grad when frozen (no activations kept), otherwise honours
        the ambient grad state so the encoder can be fine-tuned if desired.
        """
        grad_enabled = torch.is_grad_enabled() and not self.freeze
        with torch.set_grad_enabled(grad_enabled):
            return self._encode(clip)


# ── VideoMAE ────────────────────────────────────────────────────────────────────

class VideoMAEClipEncoder(BaseClipEncoder):
    """
    HuggingFace VideoMAE wrapped to emit one feature vector per input frame.

    A VideoMAE forward pass consumes a fixed clip of ``num_frames`` (usually 16)
    and returns ``(num_frames/tubelet) * num_spatial`` tokens -- half as many
    temporal positions as input frames, since each tubelet spans 2 frames. To
    turn an arbitrary-length clip (e.g. 64 frames) into ``(B, T, D)`` (one
    token per input frame, per the per-frame contract this class guarantees)
    there are two strategies, controlled by ``interp_time``:

    interp_time=True (cheap, default):
        1. split the clip into consecutive windows of ``clip_len`` frames
           (padding the tail by repeating the last frame),
        2. run the backbone once per window,
        3. mean-pool the spatial patches at each temporal tubelet position
           (or keep a spatial grid if ``spatial_grid > 0``),
        4. concatenate windows along time (T/2 native tokens) and linearly
           interpolate the temporal axis back up to exactly T -- every other
           output token is therefore a synthetic linear blend of its two
           neighbours, not a real backbone output.

    interp_time=False (exact, ~2x backbone compute):
        Instead of interpolating, get T *genuine* tubelet tokens directly by
        feeding the backbone a doubled, duplicated-frame sequence where every
        consecutive pair IS a real adjacent-frame pair from the original clip:
        ``[F0,F1, F1,F2, F2,F3, ..., F(T-2),F(T-1), F(T-1),F(T-1)]`` (length
        2T; the final pair duplicates the last frame so the count comes out
        to exactly T rather than T-1). Windowing this exactly as above and
        skipping the interpolation step yields T output tokens, each one an
        actual backbone embedding of a real (t, t+1) pair -- at the cost of
        running roughly 2x as many raw frames through the frozen backbone.

    interp_time="downsample" (cheapest, breaks the T-in/T-out contract):
        Same windowed backbone pass as interp_time=True, but skip step 4 --
        return the T/2 native tubelet tokens as-is, with no upsampling back
        to T and no extra backbone compute either. Output is (B,T/2,...),
        NOT (B,T,...): unlike the other two modes, this one does not honour
        this class's normal "one output token per input frame" contract, so
        callers must build/resample every T-aligned target (GT density,
        phase, in-repetition mask, ...) to the encoder's actual output
        length (``feats.shape[1]``) themselves rather than assuming it
        equals the input clip length -- see check.py's load_sample for the
        existing pattern (it already sizes GT density off ``feats.shape[1]``
        instead of clip_length).

    Args:
        model_name:    any key in ``FEATURE_DIMS`` (configurable backbone).
        freeze:        freeze weights + eval mode (default True).
        pooling:       'mean' -> spatial mean-pool (returns (B,T,D)).
                       'grid' -> keep a spatial_grid x spatial_grid grid
                                 (returns (B,T,S,D); pair with the model's
                                 SpatialAttentionPool, spatial_pool=True).
        spatial_grid:  grid side length when pooling='grid'.
        return_intermediate: if set, use hidden_states[return_intermediate]
                       instead of last_hidden_state (intermediate feature
                       extraction, per spec).
        interp_time:   True (default) | False | "downsample" -- see above.
                       True/False both preserve the (B,T,...) per-frame
                       contract; "downsample" does not (output is (B,T/2,...)).
    """

    FEATURE_DIMS = {
        "MCG-NJU/videomae-base":                     768,
        "MCG-NJU/videomae-base-finetuned-kinetics":  768,
        "MCG-NJU/videomae-large":                    1024,
        "MCG-NJU/videomae-huge":                     1280,
    }

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        freeze: bool = True,
        pooling: str = "mean",
        spatial_grid: int = 4,
        return_intermediate: Optional[int] = None,
        interp_time: bool = True,
    ):
        super().__init__(freeze=freeze)
        try:
            from transformers import VideoMAEModel
        except ImportError as e:  # pragma: no cover
            raise ImportError("VideoMAE needs transformers>=4.25: pip install -U transformers") from e

        if model_name not in self.FEATURE_DIMS:
            raise ValueError(f"Unknown VideoMAE model '{model_name}'. Options: {list(self.FEATURE_DIMS)}")
        if pooling not in ("mean", "grid"):
            raise ValueError("pooling must be 'mean' or 'grid'.")
        if interp_time not in (True, False, "downsample"):
            raise ValueError(f"interp_time must be True, False, or 'downsample', got {interp_time!r}.")

        self.model_name          = model_name
        self.pooling             = pooling
        self.spatial_grid        = spatial_grid
        self.return_intermediate = return_intermediate
        self.interp_time         = interp_time

        self.backbone = VideoMAEModel.from_pretrained(
            model_name,
            output_hidden_states=(return_intermediate is not None),
        )
        self.feature_dim   = self.FEATURE_DIMS[model_name]
        self.image_size    = int(getattr(self.backbone.config, "image_size", 224))
        self.clip_len      = int(getattr(self.backbone.config, "num_frames", 16))
        self.tubelet_size  = int(getattr(self.backbone.config, "tubelet_size", 2))
        self.tokens_per_win = self.clip_len // self.tubelet_size
        self.temporal_downsample = self.tubelet_size if interp_time == "downsample" else 1

        self._apply_freeze(self.backbone)

    def output_length(self, T: int) -> int:
        """Number of temporal tokens this encoder emits for T input frames.
        True/False (per-frame contract preserved): always T. "downsample":
        T padded up to a multiple of clip_len (same padding _pad_to_multiple
        applies), then divided by tubelet_size -- see _encode."""
        if self.interp_time != "downsample":
            return T
        rem = T % self.clip_len
        T_pad = T if rem == 0 else T + (self.clip_len - rem)
        return T_pad // self.tubelet_size

    # -- helpers ----------------------------------------------------------------

    def _pad_to_multiple(self, clip: torch.Tensor) -> tuple[torch.Tensor, int]:
        """(B,T,3,H,W) -> (B,T_pad,3,H,W) where T_pad is a multiple of clip_len."""
        T = clip.shape[1]
        rem = T % self.clip_len
        if rem == 0:
            return clip, T
        pad = self.clip_len - rem
        tail = clip[:, -1:].expand(-1, pad, *clip.shape[2:])
        return torch.cat([clip, tail], dim=1), T

    def _run_windows(self, windows: torch.Tensor) -> torch.Tensor:
        """
        windows: (N, clip_len, 3, H, W) -> (N, tokens_per_win, D)    [pooling='mean']
                                        or  (N, tokens_per_win, S, D) [pooling='grid'].

        N = B * n_windows -- every window in the batch is run through the
        backbone in ONE call so the GPU actually sees a full batch instead of
        being fed one clip_len-frame window at a time from a Python loop.
        """
        out = self.backbone(pixel_values=windows)
        if self.return_intermediate is not None:
            tokens = out.hidden_states[self.return_intermediate]  # (N, L, D)
        else:
            tokens = out.last_hidden_state                        # (N, L, D)
        N = windows.shape[0]
        num_spatial = tokens.shape[1] // self.tokens_per_win
        tokens = tokens.reshape(N, self.tokens_per_win, num_spatial, -1)  # (N, t, S, D)

        if self.pooling == "mean":
            return tokens.mean(dim=2)  # (N, t, D)

        # grid pooling: adaptive-pool the native patch grid to spatial_grid^2
        side = int(round(num_spatial ** 0.5))
        if side * side != num_spatial:
            raise ValueError(f"num_spatial={num_spatial} not a perfect square; cannot grid-pool.")
        D = tokens.shape[-1]
        g = tokens.reshape(N * self.tokens_per_win, side, side, D).permute(0, 3, 1, 2)
        g = F.adaptive_avg_pool2d(g, (self.spatial_grid, self.spatial_grid))
        g = g.permute(0, 2, 3, 1).reshape(N, self.tokens_per_win, self.spatial_grid ** 2, D)
        return g  # (N, t, S, D)

    @staticmethod
    def _interp_time(seq: torch.Tensor, T: int) -> torch.Tensor:
        """seq: (B, L, [S,] D) -> (B, T, [S,] D) via linear interpolation on the time axis."""
        return _linear_interp_time(seq, T)

    @staticmethod
    def _build_adjacent_pairs(clip: torch.Tensor) -> torch.Tensor:
        """
        clip: (B, T, 3, H, W) -> (B, 2T, 3, H, W) where every consecutive pair
        in the output is a REAL adjacent pair from the input: output[2i] =
        clip[i], output[2i+1] = clip[i+1], with clip[T-1] duplicated for the
        final pair (so the pair count comes out to T, not T-1). Windowing this
        sequence with the class's tubelet_size=2 grouping turns each pair back
        into exactly one genuine (non-interpolated) tubelet token per original
        input frame.
        """
        B, T = clip.shape[0], clip.shape[1]
        right = torch.cat([clip[:, 1:], clip[:, -1:]], dim=1)   # frame t+1, last frame duplicated
        return torch.stack([clip, right], dim=2).reshape(B, 2 * T, *clip.shape[2:])

    def _windowed_tokens(self, clip: torch.Tensor) -> torch.Tensor:
        """clip: (B, L, 3, H, W) -> (B, L_pad/tubelet_size, [S,] D) via the
        window-split + backbone + spatial-pool pipeline (no temporal resampling)."""
        B = clip.shape[0]
        clip_pad, _ = self._pad_to_multiple(clip)          # (B, L_pad, 3, H, W)
        n_windows = clip_pad.shape[1] // self.clip_len

        windows = clip_pad.reshape(B * n_windows, self.clip_len, *clip_pad.shape[2:])
        win_feats = self._run_windows(windows)             # (B*n_windows, tokens_per_win, [S,] D)

        extra = win_feats.shape[2:]
        return win_feats.reshape(B, n_windows * self.tokens_per_win, *extra)

    def _encode(self, clip: torch.Tensor) -> torch.Tensor:
        """(B,T,3,H,W) -> (B,T,D)/(B,T,S,D) for interp_time True/False, or
        (B,T/2,D)/(B,T/2,S,D) for interp_time="downsample" (see class
        docstring -- that mode alone breaks the length-T-out contract). One
        batched backbone call per branch, no per-sample/per-window Python loop."""
        T = clip.shape[1]

        if self.interp_time == "downsample":
            # cheapest: T/2 native tokens, returned as-is -- no interpolation
            # (unlike interp_time=True) and no doubled backbone pass (unlike
            # interp_time=False). Caller must handle the shorter output length.
            return self._windowed_tokens(clip)

        if self.interp_time:
            # cheap: T/2 native tokens, linearly interpolated up to T
            seq = self._windowed_tokens(clip)
            return self._interp_time(seq, T)

        # exact: T genuine tokens from a doubled real-adjacent-pair sequence,
        # no interpolation (~2x backbone compute vs the interp_time=True path)
        seq = self._windowed_tokens(self._build_adjacent_pairs(clip))  # (B, >=T, [S,] D)
        return seq[:, :T]


# ── DINOv2 (frame-level reference implementation of the swap contract) ──────────

class DINOv2ClipEncoder(BaseClipEncoder):
    """
    Per-frame DINOv2 encoder -- demonstrates that swapping encoders needs no
    downstream change (design goal #4). Each frame is encoded independently and
    the CLS token is used as its feature, so output is naturally (B, T, D).
    """

    FEATURE_DIMS = {
        "facebook/dinov2-small": 384,
        "facebook/dinov2-base":  768,
        "facebook/dinov2-large": 1024,
    }

    def __init__(self, model_name: str = "facebook/dinov2-base", freeze: bool = True):
        super().__init__(freeze=freeze)
        try:
            from transformers import AutoModel
        except ImportError as e:  # pragma: no cover
            raise ImportError("DINOv2 needs transformers: pip install -U transformers") from e
        if model_name not in self.FEATURE_DIMS:
            raise ValueError(f"Unknown DINOv2 model '{model_name}'. Options: {list(self.FEATURE_DIMS)}")
        self.backbone    = AutoModel.from_pretrained(model_name)
        self.feature_dim = self.FEATURE_DIMS[model_name]
        self._apply_freeze(self.backbone)

    def _encode(self, clip: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = clip.shape
        flat = clip.reshape(B * T, C, H, W)
        out  = self.backbone(pixel_values=flat)
        cls  = out.last_hidden_state[:, 0]           # (B*T, D)
        return cls.reshape(B, T, self.feature_dim)


# ── R(2+1)D-18 (genuine 3D-CNN encoder, unlike VideoMAE/DINOv2 above) ───────────

class R2Plus1D18ClipEncoder(BaseClipEncoder):
    """
    torchvision's R(2+1)D-18 (Tran et al., 2018 -- "A Closer Look at Spatiotemporal
    Convolutions for Action Recognition"), Kinetics-400 pretrained. Unlike
    VideoMAEClipEncoder (transformer) or DINOv2ClipEncoder (per-frame, no
    temporal modeling at all), this is a genuinely spatiotemporal 3D CNN: each
    layer factorizes a 3D conv into a 2D spatial conv + 1D temporal conv.

    Only the backbone up to (not including) avgpool/fc is used -- the Kinetics
    classification head is discarded; this is a pure feature extractor.

    stem+layer1..4 downsamples time by a fixed 8x (T -> T/8, via a stride-2
    temporal conv at the start of layer2/3/4) and space to a 7x7 grid. To honour
    the "T frames in -> T frames out" contract every encoder here guarantees,
    this mean-pools the 7x7 spatial grid then linearly interpolates the T/8
    temporal positions back up to T -- the same strategy VideoMAEClipEncoder
    uses for ITS temporal downsampling (see that class's docstring / interp_time),
    just with a fixed 8x factor here instead of VideoMAE's tubelet-driven 2x, and
    no non-interpolated alternative (R(2+1)D-18 has no analogue of VideoMAE's
    "feed real adjacent-frame pairs" trick -- its 8x reduction is baked into
    strided convs, not a token-grouping choice).

    model_name is accepted (unused) only so the constructor matches the same
    `model_name=...` call convention every other registered encoder uses (see
    build_encoder / train.py's `else: enc_kwargs.update(model_name=...)`) --
    torchvision currently ships only this one R(2+1)D-18 + Kinetics-400
    combination, so there is nothing to select between.
    """

    FEATURE_DIM = 512
    # torchvision's Kinetics-400 R(2+1)D-18 normalization -- NOT ImageNet stats
    # (see torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1.transforms()).
    pixel_mean = (0.43216, 0.394666, 0.37645)
    pixel_std  = (0.22803, 0.22145, 0.216989)

    def __init__(self, model_name: str = "r2plus1d_18", freeze: bool = True, pretrained: bool = True):
        super().__init__(freeze=freeze)
        try:
            from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        except ImportError as e:  # pragma: no cover
            raise ImportError("R(2+1)D-18 needs torchvision: pip install -U torchvision") from e

        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        backbone = r2plus1d_18(weights=weights)
        # Drop avgpool/fc -- keep the spatiotemporal feature map (B,512,T/8,7,7).
        self.backbone = nn.Sequential(
            backbone.stem, backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.feature_dim = self.FEATURE_DIM
        self.image_size  = 112   # matches the pretrained weights' crop_size

        self._apply_freeze(self.backbone)

    def _encode(self, clip: torch.Tensor) -> torch.Tensor:
        """(B,T,3,H,W) -> (B,T,D). See class docstring for the T/8 -> T upsample."""
        T = clip.shape[1]
        x = clip.permute(0, 2, 1, 3, 4)       # (B,3,T,H,W) -- Conv3d wants channels-first-3D
        x = self.backbone(x)                  # (B,512,T/8,7,7)
        x = x.mean(dim=(-1, -2))              # (B,512,T/8) -- spatial mean pool
        x = x.transpose(1, 2)                 # (B,T/8,512)
        return _linear_interp_time(x, T)      # (B,T,512)


# ── factory ─────────────────────────────────────────────────────────────────────

_ENCODERS = {
    "videomae":    VideoMAEClipEncoder,
    "dinov2":      DINOv2ClipEncoder,
    "r2plus1d18":  R2Plus1D18ClipEncoder,
    "r2plus1d_18": R2Plus1D18ClipEncoder,   # alias matching torchvision's own function name
}


def build_encoder(name: str, **kwargs) -> BaseClipEncoder:
    """
    Construct a frozen clip encoder by short name.

        build_encoder("videomae",   model_name="MCG-NJU/videomae-base")
        build_encoder("dinov2",     model_name="facebook/dinov2-base")
        build_encoder("r2plus1d18")

    To add InternVideo / VideoMamba, implement a BaseClipEncoder subclass
    returning (B, T, D) and register it here -- nothing downstream changes.
    """
    key = name.lower()
    if key not in _ENCODERS:
        raise ValueError(f"Unknown encoder '{name}'. Registered: {list(_ENCODERS)}")
    return _ENCODERS[key](**kwargs)
