"""
Repetition counting model: dilated TCN + density head, over PRE-ENCODED
VideoMAE features -- this module never builds or calls an encoder itself.

Architecture:
    encoder features (B, T, D) or (B, T, S, D)   [see frame_encoder.py --
                                                    encoded separately by the
                                                    caller, e.g. train.py]
        -> [learned spatial pool, if a spatial grid (B,T,S,D) was passed]  -> (B, T, D)
        -> Linear projection                                      -> (B, T, tcn_dim)
        -> TCN (single TemporalBlock, or DilatedTCN stack)        -> (B, T, tcn_dim)
        -> DensityHead (per-frame FC + Softplus)                   -> (B, T)
        -> count = density.sum(dim=-1)                             -> (B,)

Two earlier versions of this file are gone (see git history if you need to
resurrect either):
  - An inline encoder: RepetitionCounter could optionally build its own
    videomae_encoder.VideoMAEEncoderBatched (`build_encoder=True`) and accept
    raw (B,T,3,H,W) frames directly, for multispeed_eval.py's on-the-fly
    re-encoding at multiple strides. That script and this path were both
    removed as unused -- every remaining caller encodes separately (see
    frame_encoder.VideoMAEClipEncoder) and calls forward_from_features().
  - A RepNet-style Temporal Self-Similarity Matrix -> fixed-width conv stack
    -> Transformer aggregator that used to sit between the encoder features
    and the heads; replaced by the plain linear-projection + TCN pipeline
    above (`TemporalSelfSimilarity`/`TSMConvStack`/`TemporalTransformerAggregator`).

T (frames per video) can be VARIABLE across videos -- e.g. sample more frames
for longer videos, fewer for shorter ones. This works for free here: Conv1d
(inside TemporalBlock/DilatedTCN) and the per-frame DensityHead both operate
pointwise/along the T axis with no fixed-size assumption, so no positional-
embedding or row-width interpolation trick is needed.

IMPORTANT PRACTICAL CONSEQUENCE: since T differs per video, you can no longer
naively torch.stack() a batch of videos with different T into one tensor.
Either use batch_size=1, or pad shorter videos to a common T within a batch
and mask the padding (not implemented here -- out of scope for this pass).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def print_stats(name, x, i):
    if i==0:
        if x is None:
            return
        print(f"\n{name}")
        print("shape:", tuple(x.shape))
        print("mean :", x.mean().item())
        print("std  :", x.std().item())

        # If x has shape (B,T,C)
        if x.ndim == 3:
            print("temporal std:", x.std(dim=1).mean().item())

        # If x has shape (B,T,H,C)
        elif x.ndim == 4:
            # std over time
            print("temporal std:", x.std(dim=1).mean().item())

# ── learned spatial pooling ────────────────────────────────────────────────────

class SpatialAttentionPool(nn.Module):
    """
    Collapse the S spatial patch-tokens at each timestep into ONE vector, using a
    learned query that attends over the patches -- i.e. a TRAINABLE replacement
    for the global spatial mean-pool in videomae_encoder._encode_window.

    Input:  (B, T, S, D)   Output: (B, T, D)

    Why this exists: mean-pooling averages the ~5% moving patches (which carry the
    repetition signal) with the ~95% static-background patches, attenuating the
    per-timestep signal contrast ~20x (measured: moving-patch cosine 0.49 vs
    mean-pooled 0.94) -- enough to kill the whole TSM pipeline. A learned query,
    trained end-to-end with the density head, can instead put its attention mass
    on the moving patches and preserve that contrast. This is the RepNet insight
    (make the pooling learned) applied to a frozen VideoMAE grid.
    """

    def __init__(self, dim: int, nheads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn  = nn.MultiheadAttention(dim, nheads, dropout=dropout, batch_first=True)
        self.norm  = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, S, D) -> (B, T, D)"""
        B, T, S, D = x.shape
        x = self.norm(x).reshape(B * T, S, D)
        q = self.query.expand(B * T, 1, D)
        out, _ = self.attn(q, x, x)          # (B*T, 1, D)
        return out.reshape(B, T, D)


# ── TCN blocks ─────────────────────────────────────────────────────────────────

class TemporalBlock(nn.Module):
    """
    expansion=1 (default): unchanged legacy structure -- the pointwise step is
        a single d->d Conv1d, identical state_dict shape to before this arg
        existed, old checkpoints still load with strict=True.
    expansion>1: inverted-bottleneck / ConvNeXt-style block -- the depthwise
        dilated conv (the only part that mixes across TIME) stays at width d,
        but the pointwise step (the only part that mixes across CHANNELS)
        becomes a small 2-layer MLP, d -> expansion*d -> d, instead of one
        linear map. This mirrors a transformer block's split: attention mixes
        across tokens and does no per-token feature transform, so the FFN is
        deliberately widened (4x) to give per-token processing real capacity
        -- here the depthwise conv is the "mixing" step and the pointwise
        step is the "per-timestep transform" step, so it's the pointwise step
        that gets widened, not the depthwise conv. Does not change receptive
        field at all (that's purely `dilation`/block count) -- purely a
        per-block capacity lever, orthogonal to depth.
    """
    def __init__(self, d=256, k=5, dilation=1, expansion=1):
        super().__init__()

        layers = [
            nn.Conv1d(d, d, kernel_size=k, padding=dilation * (k // 2),
                      dilation=dilation, groups=d),                   # depthwise, dilated
            nn.GELU(),
        ]
        if expansion == 1:
            layers += [nn.Conv1d(d, d, kernel_size=1), nn.GELU()]     # pointwise
        else:
            hidden = d * expansion
            layers += [
                nn.Conv1d(d, hidden, kernel_size=1),                  # pointwise expand
                nn.GELU(),
                nn.Conv1d(hidden, d, kernel_size=1),                  # pointwise project back
            ]
        self.net = nn.Sequential(*layers)

        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        # x: (B,T,D)
        residual = x

        x = x.transpose(1,2)
        x = self.net(x)
        x = x.transpose(1,2)

        return self.norm(x + residual)

class DilatedTCN(nn.Module):
    def __init__(self, d=256, k=5, dilations=(1,2,4,8,16), expansion=1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TemporalBlock(d, k, dilation=dil, expansion=expansion) for dil in dilations]
        )
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return x


class TransformerTCN(nn.Module):
    """
    Drop-in alternative to DilatedTCN/TemporalBlock: SAME (B,T,d) -> (B,T,d)
    contract (see RepetitionCounter._forward_feats) -- mixes across TIME with
    bidirectional self-attention instead of dilated convolutions. Every
    window this codebase ever runs through the TCN is exactly `clip_length`
    frames (training clips, and eval's sliding-window chunks -- see
    sliding_window_inference's fixed-length last-frame-repeat padding), so a
    learned positional embedding sized to `clip_length` is safe here, unlike
    a general seq2seq transformer that must handle arbitrary lengths.

    No causal mask: DilatedTCN's dilated convs are non-causal ('same'
    padding, symmetric context both directions), so full bidirectional
    attention matches that existing convention rather than changing it.

    pre-norm (norm_first=True) transformer encoder layers -- standard for
    training stability at depth.
    """
    def __init__(self, d=1024, nhead=8, num_layers=5, dim_feedforward=None,
                 dropout=0.1, clip_length=128):
        super().__init__()
        dim_feedforward = dim_feedforward or d * 4
        self.pos_encoding = nn.Parameter(torch.zeros(1, clip_length, d))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        """x: (B,T,d), T must equal clip_length (see class docstring)."""
        T = x.shape[1]
        if T > self.pos_encoding.shape[1]:
            raise ValueError(
                f"TransformerTCN got T={T} but its positional embedding only covers "
                f"{self.pos_encoding.shape[1]} positions (clip_length at construction time) -- "
                "every window must be exactly clip_length long, same as DilatedTCN's own usage."
            )
        x = x + self.pos_encoding[:, :T]
        return self.encoder(x)


# ── density head ─────────────────────────────────────────────────────────────

class DensityHead(nn.Module):
    """
    Per-frame density values. (B, T, input_dim) -> (B, T).

    No output activation (unlike an earlier Softplus version): this matches
    TransRAC's actual head design and gives better gradient flow, at the cost
    of no hard non-negativity guarantee. MSE loss against a non-negative GT
    density map naturally pushes predictions positive over training; small
    negative dips can still occur, especially early in training. Clamp at
    evaluation time (predict_count already does this on the summed count).
    """

    def __init__(self, input_dim: int = 512, n_hidden_1: int = 512,
                 n_hidden_2: int = 256, out_dim: int = 1, dropout: float = 0.25):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, n_hidden_1),
            nn.LayerNorm(n_hidden_1),
            nn.Dropout(p=dropout, inplace=False),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(n_hidden_2, out_dim),
        )

    def forward(self, x):
        """x: (B, T, input_dim) -> (B, T)"""
        x = self.fc(x)          # (B, T, 1)
        x = F.softplus(x)
        return x.squeeze(-1)    # (B, T)


class InRepetitionHead(nn.Module):
    """
    Auxiliary per-frame binary classifier: is this frame currently inside a
    repetition or not? Target is density.make_phase_targets' own phase_mask
    (1.0 = inside an annotated cycle, 0.0 = not).

    Purely auxiliary and independent: consumes the SAME shared temporal
    feature (`rows`, see RepetitionCounter._forward_feats) as DensityHead/
    FuturePredictionHead, but its own output never feeds back into
    density/count -- it exists only to shape the upstream
    representation via its own BCE loss term (see loss.InRepetitionLoss).

    (B, T, input_dim) -> (B, T): raw logit per frame. No output activation
    (matching DensityHead's no-activation convention) -- use
    F.binary_cross_entropy_with_logits (numerically stabler than a separate
    sigmoid + BCE) against the phase_mask target.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) -> (B, T)  raw logit"""
        return self.fc(x).squeeze(-1)


class FuturePredictionHead(nn.Module):
    """
    Auxiliary future-embedding predictor: given the shared temporal bottleneck
    h_t, predicts the (frozen) encoder's embedding e_{t+k} for some horizon k
    (see FuturePredictionLoss for the shift + loss). Purely an MLP applied
    independently per timestep -- no autoregressive decoding, no transformer
    decoder, no reconstruction of the input frames themselves. Exists only to
    push the upstream temporal aggregator toward representations that carry
    predictive information about future encoder states; its own output never
    feeds the count path (see RepetitionCounter._forward_feats).

    (B, T, hidden_dim) -> (B, T, embed_dim), embed_dim = frozen encoder's
    feature dim D (e.g. 768 for videomae-base).

    Consumes the SAME shared temporal bottleneck `rows` as DensityHead (see
    RepetitionCounter._forward_feats); its own output never feeds back into
    density/count.
    """

    def __init__(self, hidden_dim: int = 256, embed_dim: int = 768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, hidden_dim) -> (B, T, embed_dim)"""
        return self.fc(h)


class CountHead(nn.Module):
    """
    Auxiliary count-regression head: a small pooled/per-frame MLP that tests
    whether the shared `rows` bottleneck already carries enough information
    to regress the repetition count directly, independent of the unsupervised
    per-frame density map DensityHead has to discover on its own. Follows up
    on the P-vs-2P periodicity investigation (see p_vs_2p_analysis.py): about
    a third of fast, bilateral-motion clips have their strongest latent
    period at 2x the annotated repetition rate (already true BEFORE the TCN,
    equally in Countix and RepCount) -- this head is the suggested next
    experiment, testing directly whether "this latent 2P cycle = 2 annotated
    reps" is learnable from a global/per-frame regression instead of relying
    on the density map to encode it implicitly.

    mode="per_video" (default): mean-pool `rows` over T FIRST, then MLP ->
    ONE scalar per clip -- "how many repetitions total, given the whole
    clip's pooled representation." Cannot use any per-frame timing at all,
    so if the count is learnable this way, it must be recoverable from the
    clip's overall temporal texture (e.g. dominant periodicity), not from
    localizing individual repetitions.
    mode="per_frame": MLP applied independently per timestep -> (B,T), then
    summed over T -- mirrors DensityHead's per-frame-then-sum count path,
    but with this head's own (smaller, dropout-regularized) architecture
    instead of DensityHead's, and no per-frame density supervision (only the
    summed total is trained against gt_count -- see loss.CountHeadLoss).

    Purely auxiliary and independent of DensityHead/InRepetitionHead/
    FuturePredictionHead: consumes the SAME shared `rows` bottleneck, but its
    own output never feeds back into density/count. Target is gt_count
    itself, regardless of the clip's T (see loss.CountHeadLoss).

    (B, T, input_dim) -> (B,): raw scalar (no output activation, matching
    DensityHead's no-activation convention -- a regression loss against a
    non-negative target naturally pushes it positive over training).
    """

    def __init__(self, input_dim: int = 256, mode: str = "per_video", dropout: float = 0.25):
        super().__init__()
        if mode not in ("per_video", "per_frame"):
            raise ValueError(f"mode must be 'per_video' or 'per_frame', got {mode!r}")
        self.mode = mode
        self.count_mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) -> (B,)  raw predicted count."""
        if self.mode == "per_video":
            pooled = x.mean(dim=1)                       # (B, input_dim)
            return self.count_mlp(pooled).squeeze(-1)    # (B,)
        # per_frame: independent per-timestep prediction, summed into a total count
        per_frame = self.count_mlp(x).squeeze(-1)        # (B, T)
        return per_frame.sum(dim=-1)                      # (B,)


def unpack_output(out):
    """
    Normalise RepetitionCounter's forward()/forward_from_features() output to
    (density, count, phase). phase is always None now (the auxiliary phase
    head was removed as unused -- see git history) -- kept as a 3-tuple so
    callers that still unpack (density, count, _) need no changes.

    The model returns a plain (density, count) tuple when every auxiliary
    head is disabled -- and a {"density", "count", ...} dict otherwise (see
    RepetitionCounter._forward_feats). This helper lets callers (training
    loop, sliding_window_inference, ...) handle both shapes uniformly.
    """
    if isinstance(out, dict):
        return out["density"], out["count"], None
    density, count = out
    return density, count, None


def unpack_future(out):
    """
    Companion to unpack_output(): pulls (future_pred, encoder_feats) out of
    RepetitionCounter's output for the auxiliary future-prediction loss.
    Returns (None, None) when the model was built with use_future_pred=False
    (out is the legacy tuple, or a dict without these keys) -- callers can
    then skip the future-prediction loss term unconditionally.
    """
    if isinstance(out, dict):
        return out.get("future_pred"), out.get("encoder_feats")
    return None, None


def unpack_in_repetition(out):
    """
    Companion to unpack_output(): pulls the InRepetitionHead's raw logit out
    of RepetitionCounter's output for the auxiliary in-repetition BCE loss.
    Returns None when the model was built with use_in_repetition=False (out
    is the legacy tuple, or a dict without this key) -- callers can then skip
    the in-repetition loss term unconditionally.
    """
    if isinstance(out, dict):
        return out.get("in_rep")
    return None


def unpack_count_head(out):
    """
    Companion to unpack_output(): pulls the CountHead's raw per-clip scalar
    out of RepetitionCounter's output for the auxiliary count-regression
    loss. Returns None when the model was built with use_count_head=False
    (out is the legacy tuple, or a dict without this key) -- callers can then
    skip the count-head loss term unconditionally.
    """
    if isinstance(out, dict):
        return out.get("count_head")
    return None


# ── full model ────────────────────────────────────────────────────────────────

class RepetitionCounter(nn.Module):
    """
    Downstream model only: (B, T, D) or (B, T, S, D) encoder features ->
    linear_proj1 -> TCN -> DensityHead (+ auxiliary heads) -> count.

    This model never builds or calls a VideoMAE encoder itself -- callers
    encode video frames separately (see frame_encoder.VideoMAEClipEncoder,
    used by train.py/evaluate.py/infer.py) and pass the resulting features to
    forward_from_features(). This used to be optional (a build_encoder=True
    path could construct videomae_encoder.VideoMAEEncoderBatched inline and
    accept raw frames directly, for multispeed_eval.py's on-the-fly re-
    encoding at multiple strides), but multispeed_eval.py and that inline
    encoder path were both removed as unused -- every remaining caller in
    this repo already encodes separately. See git history to resurrect it.
    """

    def __init__(
        self,
        dropout:            float = 0.1,
        spatial_pool:       bool  = False,      # True: input features are a per-timestep
                                                 # spatial GRID (B,T,S,D) (from
                                                 # extract_dense_features --spatial_grid>0);
                                                 # a trainable SpatialAttentionPool collapses
                                                 # S->1 (learns to weight moving patches)
                                                 # before linear_proj1/the TCN. False: features
                                                 # are already (B,T,D) (global-pooled at
                                                 # extraction).
        feature_dim:        int   = 768,        # encoder feature dim D -- sizes linear_proj1's
                                                 # input (must match whatever encoder produced the
                                                 # features passed to forward_from_features, e.g.
                                                 # encoder.feature_dim) and, when spatial_pool=True,
                                                 # SpatialAttentionPool. 768 (default) matches
                                                 # VideoMAE-base/DINOv2-base -- unchanged legacy
                                                 # behaviour. Changing it changes linear_proj1's
                                                 # state_dict shape.
        spatial_nheads:     int   = 4,
        use_in_repetition:  bool  = False,      # attach the auxiliary InRepetitionHead (see
                                                 # InRepetitionHead docstring) -- a per-frame
                                                 # BCE classifier trained against phase_mask.
                                                 # False (default): self.in_rep_head=None,
                                                 # forward output and state_dict are IDENTICAL
                                                 # to before this feature existed -- old
                                                 # checkpoints still load with strict=True.
        in_rep_hidden_dim:  int   = 512,        # InRepetitionHead's first hidden width.
        in_rep_dropout:     Optional[float] = None,  # InRepetitionHead dropout; None -> reuse `dropout`.
        use_count_head:     bool  = False,      # attach the auxiliary CountHead (see CountHead
                                                 # docstring) -- a small pooled/per-frame MLP
                                                 # regressed directly against gt_count, independent
                                                 # of use_in_repetition. False (default):
                                                 # self.count_head=None, forward
                                                 # output and state_dict are IDENTICAL to before
                                                 # this feature existed -- old checkpoints still
                                                 # load with strict=True.
        count_head_mode:    str   = "per_video", # 'per_video': mean-pool `rows` over T then MLP
                                                 # -> one scalar per clip. 'per_frame': MLP per
                                                 # timestep, summed over T -- mirrors DensityHead's
                                                 # per-frame-then-sum count path with CountHead's
                                                 # own architecture. Only used when
                                                 # use_count_head=True.
        count_head_dropout: float = 0.25,       # CountHead's Dropout probability (its one
                                                 # regularized layer, between the two hidden
                                                 # Linears).
        use_density:        bool  = True,       # True (default): build DensityHead, count =
                                                 # density.sum() -- unchanged legacy behaviour.
                                                 # False: no DensityHead at all (self.head=None);
                                                 # count instead comes directly from CountHead's
                                                 # own prediction. Requires use_count_head=True --
                                                 # there is otherwise no signal to count from. This
                                                 # is how to get "count from CountHead only": set
                                                 # use_density=False, use_count_head=True (config:
                                                 # use_density: false, count_head.enabled: true).
        density_hidden_1:   int   = 64,         # DensityHead's first hidden width. 64 (default)
                                                 # matches this codebase's long-standing hardcoded
                                                 # value -- unchanged legacy behaviour. This is the
                                                 # head that actually produces pred_count in every
                                                 # use_density=True run (density.sum()), yet its
                                                 # hidden size has never scaled with tcn_dim across
                                                 # any tcn_dim sweep (256/512/1024/2048) -- 64 was
                                                 # already a steep bottleneck at tcn_dim=256, and a
                                                 # 16:1 squeeze at tcn_dim=1024. DensityHead's OWN
                                                 # class defaults are 512 -- this arg exists so that
                                                 # more reasonable value (or any other) is reachable
                                                 # from config instead of only from a code edit.
                                                 # Changes state_dict shape -- must match between
                                                 # train/eval/checkpoint, like tcn_dim.
        density_hidden_2:   int   = 64,         # DensityHead's second hidden width. 64 (default)
                                                 # matches this codebase's long-standing hardcoded
                                                 # value -- unchanged legacy behaviour. See
                                                 # density_hidden_1 above. DensityHead's own class
                                                 # default is 256. Changes state_dict shape.
        use_future_pred:    bool  = False,      # attach the auxiliary FuturePredictionHead (see
                                                 # FuturePredictionHead docstring). False (default):
                                                 # self.future_head=None, forward output and
                                                 # state_dict are IDENTICAL to before this
                                                 # feature existed -- old checkpoints still
                                                 # load with strict=True.
        future_horizon:     int   = 4,          # predict the encoder embedding this many
                                                 # timesteps ahead (see FuturePredictionLoss).
                                                 # Stored on the model purely for bookkeeping/
                                                 # logging -- the shift itself happens in the loss.
        use_dilated_tcn:    bool  = False,      # False (default): self.tcn is a single
                                                 # TemporalBlock (kernel-size receptive
                                                 # field only) -- unchanged legacy behaviour,
                                                 # old checkpoints still load with strict=True.
                                                 # True: self.tcn is a DilatedTCN stack
                                                 # (dilations (1,2,4,8,16)) -- exponentially
                                                 # larger temporal receptive field, changes
                                                 # the state_dict shape.
        tcn_kernel_size:    int   = 5,          # kernel size for TemporalBlock / DilatedTCN.
        tcn_dilations:      tuple = (1, 2, 4, 8, 16),  # only used when use_dilated_tcn=True.
        tcn_expansion:      int   = 1,          # 1 (default): each TemporalBlock's pointwise
                                                 # step is a single d->d Conv1d -- unchanged
                                                 # legacy behaviour, old checkpoints still load
                                                 # with strict=True. >1: inverted-bottleneck /
                                                 # ConvNeXt-style block -- the pointwise step
                                                 # becomes a 2-layer MLP (d -> tcn_expansion*d ->
                                                 # d) instead of one linear map, mirroring a
                                                 # transformer FFN's 4x expansion (attention/
                                                 # depthwise-conv mixes across
                                                 # tokens/time and does no per-position feature
                                                 # transform; the FFN/pointwise step is where
                                                 # that happens, so THAT'S the part that gets
                                                 # widened). Does not change receptive field --
                                                 # a per-block CAPACITY lever, orthogonal to
                                                 # tcn_dilations (depth/receptive field) and
                                                 # tcn_dim (width). Changes the state_dict shape
                                                 # of every TemporalBlock when >1 -- must match
                                                 # between train/eval/checkpoint, like tcn_dim.
        tcn_dim:            int   = 256,        # channel width of linear_proj1's output and the
                                                 # TemporalBlock/DilatedTCN stack (`rows`) -- also
                                                 # sizes DensityHead/InRepetitionHead's
                                                 # input_dim and FuturePredictionHead's hidden_dim,
                                                 # since they all consume `rows` directly. 256
                                                 # (default) matches the original hardcoded width --
                                                 # unchanged legacy behaviour, old checkpoints still
                                                 # load with strict=True. Changing it changes the
                                                 # state_dict shape of every one of those modules.
        use_temporal_transformer: bool = False, # False (default): self.tcn is DilatedTCN/
                                                 # TemporalBlock per use_dilated_tcn, unchanged
                                                 # legacy behaviour. True: self.tcn is a
                                                 # TransformerTCN instead -- bidirectional
                                                 # self-attention over time (with a learned,
                                                 # clip_length-sized positional embedding) in
                                                 # place of dilated convolutions. Takes priority
                                                 # over use_dilated_tcn when both are set. Changes
                                                 # the state_dict shape (new module) -- must match
                                                 # between train/eval/checkpoint, like
                                                 # use_dilated_tcn/tcn_dim.
        transformer_nhead: int = 8,             # TransformerTCN's attention head count. Must
                                                 # evenly divide tcn_dim. Only used when
                                                 # use_temporal_transformer=True.
        transformer_num_layers: int = 5,        # TransformerTCN's encoder-layer depth. 5
                                                 # (default) mirrors DilatedTCN's 5-dilation
                                                 # default so the two are depth-comparable.
        transformer_dim_feedforward: Optional[int] = None,  # TransformerTCN's FFN width; None
                                                 # (default) -> 4*tcn_dim, the standard
                                                 # transformer ratio.
        transformer_clip_length: int = 128,     # size of TransformerTCN's learned positional
                                                 # embedding -- must equal the `clip_length`
                                                 # every window is actually run at (training
                                                 # clips and eval's sliding-window chunks are
                                                 # both always exactly this long -- see
                                                 # TransformerTCN's docstring). Mismatches raise
                                                 # at forward time, not silently.
    ):
        super().__init__()

        if not use_density and not use_count_head:
            raise ValueError(
                "use_density=False requires use_count_head=True -- with both off there is "
                "no head left to produce a count from (no density to sum, no count_head to "
                "read directly)."
            )

        self.spatial_pool = (
            SpatialAttentionPool(feature_dim, nheads=spatial_nheads, dropout=dropout)
            if spatial_pool else None
        )
        self.linear_proj1 = nn.Linear(feature_dim, tcn_dim)
        if use_temporal_transformer:
            self.tcn = TransformerTCN(
                d=tcn_dim, nhead=transformer_nhead, num_layers=transformer_num_layers,
                dim_feedforward=transformer_dim_feedforward, dropout=dropout,
                clip_length=transformer_clip_length,
            )
        elif use_dilated_tcn:
            self.tcn = DilatedTCN(d=tcn_dim, k=tcn_kernel_size, dilations=tcn_dilations, expansion=tcn_expansion)
        else:
            self.tcn = TemporalBlock(d=tcn_dim, k=tcn_kernel_size, expansion=tcn_expansion)

        # None when disabled: no DensityHead submodule at all -- count then
        # comes exclusively from CountHead's own prediction (see
        # _forward_feats). use_density=True (default): unchanged legacy
        # head, built exactly as before this flag existed.
        self.head = (
            DensityHead(input_dim=tcn_dim, n_hidden_1=density_hidden_1, n_hidden_2=density_hidden_2,
                        out_dim=1, dropout=dropout)
            if use_density else None
        )

        # auxiliary in-repetition classifier -- reuses the SAME shared feature
        # (`rows`) as self.head above (own head, own BCE loss, see
        # InRepetitionHead). None when disabled: no extra submodule, so
        # state_dict / forward output are unchanged from before this feature
        # existed.
        self.in_rep_head = (
            InRepetitionHead(input_dim=tcn_dim, hidden_dim=in_rep_hidden_dim,
                              dropout=in_rep_dropout if in_rep_dropout is not None else dropout)
            if use_in_repetition else None
        )

        # auxiliary count-regression head -- reuses the SAME shared feature
        # (`rows`) as self.head/self.in_rep_head above, independent of both
        # (own head, own loss, see CountHead/loss.CountHeadLoss). None when
        # disabled: no extra submodule, so state_dict / forward output are
        # unchanged from before this feature existed.
        self.count_head = (
            CountHead(input_dim=tcn_dim, mode=count_head_mode, dropout=count_head_dropout)
            if use_count_head else None
        )

        # auxiliary future-embedding predictor -- reuses the SAME shared feature
        # (`rows`) as self.head above, predicting the frozen encoder's OWN
        # embedding `future_horizon` steps ahead (see FuturePredictionHead,
        # FuturePredictionLoss). Its output never feeds back into
        # density/count. None when disabled: no extra submodule, so
        # state_dict / forward output are unchanged from before this feature
        # existed.
        self.future_head = (
            FuturePredictionHead(hidden_dim=tcn_dim, embed_dim=feature_dim)
            if use_future_pred else None
        )
        self.future_horizon = future_horizon

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[RepetitionCounter] Trainable: {n:,}")
        print(f"  Encoder: none -- forward_from_features() only, caller encodes separately")
        print(f"  T is variable per video (Conv1d/DensityHead operate pointwise along T, "
              f"no fixed-size assumption)")
        if self.spatial_pool is not None:
            print(f"  SpatialAttentionPool: ON (feature_dim={feature_dim}, nheads={spatial_nheads}) "
                  f"-- learned spatial pool over (B,T,S,{feature_dim}) grid before linear_proj1")
        if self.in_rep_head is not None:
            print(f"  InRepetitionHead: ON (auxiliary, hidden_dim={in_rep_hidden_dim}) "
                  f"-- shares `rows` with DensityHead, BCE against phase_mask")
        if self.count_head is not None:
            note = ""
            if self.head is None:
                note = " -- DensityHead is OFF, so this IS the model's count (not just auxiliary)"
            print(f"  CountHead: ON (auxiliary, mode={count_head_mode}, dropout={count_head_dropout}) "
                  f"-- shares `rows` with DensityHead, regresses gt_count directly"
                  f"{note}")
        if self.head is None:
            print(f"  DensityHead: OFF -- count comes directly from CountHead's own prediction.")
        if self.future_head is not None:
            print(f"  FuturePredictionHead: ON (auxiliary, horizon={future_horizon}) "
                  f"-- shares `rows` with DensityHead, predicts encoder embedding t+{future_horizon}")
        if use_temporal_transformer:
            dff = transformer_dim_feedforward or tcn_dim * 4
            print(f"  TCN: TransformerTCN (dim={tcn_dim}, nhead={transformer_nhead}, "
                  f"num_layers={transformer_num_layers}, dim_feedforward={dff}, "
                  f"clip_length={transformer_clip_length}) -- bidirectional self-attention over time")
        elif use_dilated_tcn:
            expansion_note = f", expansion={tcn_expansion}" if tcn_expansion != 1 else ""
            print(f"  TCN: DilatedTCN (dim={tcn_dim}, k={tcn_kernel_size}, dilations={tcn_dilations}"
                  f"{expansion_note}) -- receptive field grows exponentially with depth")
        else:
            expansion_note = f", expansion={tcn_expansion}" if tcn_expansion != 1 else ""
            print(f"  TCN: single TemporalBlock (dim={tcn_dim}, k={tcn_kernel_size}{expansion_note})")

        self.i = 0

    def _forward_feats(self, feats):
        """
        feats: (B, T, D) -> (density (B,T), count (B,))
        or, when spatial_pool is on: (B, T, S, D) -- the learned spatial pool
        collapses S->1 first, so the rest of the pipeline is unchanged.

        When in_rep_head is enabled, future_head is enabled, count_head is
        enabled, OR the density head is disabled, returns
        {"density": (B,T) or None, "count": (B,), "in_rep": (B,T) or None,
         "future_pred": (B,T,D) or None, "encoder_feats": (B,T,D) or None,
         "count_head": (B,) or None}
        instead -- see InRepetitionHead, FuturePredictionHead, CountHead,
        unpack_output(), unpack_in_repetition(), unpack_future(), and
        unpack_count_head(). The legacy (density, count) tuple is returned
        ONLY in the original configuration (use_in_repetition=False,
        use_future_pred=False, use_count_head=False, use_density=True, the
        defaults) -- the one behaviour change vs. before these flags existed.

        "encoder_feats" is the (B,T,D) per-timestep encoder embedding (post
        spatial-pool) echoed back so callers can compute the future-
        prediction loss without duplicating that step; it is undetached but
        the encoder itself is frozen, so no extra graph cost.
        """
        print_stats('encoder', feats, self.i)

        if feats.dim() == 4:
            if self.spatial_pool is None:
                raise RuntimeError(
                    "Got 4D features (B,T,S,D) but this model was built with "
                    "spatial_pool=False. Rebuild RepetitionCounter(spatial_pool=True, "
                    "feature_dim=D) to consume a spatial grid, or extract with "
                    "--spatial_grid 0."
                )
            feats = self.spatial_pool(feats)   # (B, T, S, D) -> (B, T, D)
            print_stats('spatial pool', feats, self.i)

        tokens = self.linear_proj1(feats)
        print_stats('linear proj', tokens, self.i)
        rows = self.tcn(tokens)
        print_stats('tcn', rows, self.i)
        encoder_feats_out = feats  # already (B,T,D) here

        # in-repetition classifier: consumes the SAME `rows` bottleneck as
        # DensityHead above -- its own BCE loss (see loss.InRepetitionLoss)
        # trains against phase_mask, and its output never feeds back into
        # density/count below.
        in_rep = self.in_rep_head(rows) if self.in_rep_head is not None else None  # (B,T) logit or None
        print_stats('in_rep', in_rep, self.i)

        # count-regression head: consumes the SAME `rows` bottleneck as
        # DensityHead/InRepetitionHead above, independent of both -- its own
        # loss (see loss.CountHeadLoss) trains against gt_count directly, and
        # its output never feeds back into density/count below.
        count_head_out = self.count_head(rows) if self.count_head is not None else None  # (B,) or None
        print_stats('count_head', count_head_out, self.i)

        # future-embedding prediction: consumes the SAME `rows` bottleneck as
        # DensityHead above (not detached -- its loss also updates the shared
        # temporal aggregator), and never feeds back into density/count below.
        future_pred = self.future_head(rows) if self.future_head is not None else None  # (B,T,D) or None
        print_stats('future_pred', future_pred, self.i)

        if self.head is not None:
            density = self.head(rows)              # (B, T)
            print_stats('density', density, self.i)
            count = density.sum(dim=-1)             # (B,)
        else:
            # density head disabled (use_density=False) -- CountHead's own
            # prediction IS the model's count here, not just an auxiliary
            # side-signal (contrast with the use_density=True case, where
            # count_head_out is computed above but never feeds `count`).
            density = None
            count   = count_head_out                # (B,)
        self.i += 1

        if (self.future_head is None and self.in_rep_head is None
                and self.count_head is None and self.head is not None):
            return density, count                    # unchanged legacy shape

        return {
            "density": density, "count": count,
            "in_rep": in_rep,
            "future_pred": future_pred,
            "encoder_feats": encoder_feats_out,
            "count_head": count_head_out,
        }

    def forward_from_features(self, feats):
        """feats: (B, T, D) pre-extracted -> (density, count), or a dict if any auxiliary head is on."""
        return self._forward_feats(feats)