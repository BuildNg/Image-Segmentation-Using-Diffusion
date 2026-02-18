"""
Native PyTorch model definitions for LDSeg (2-D segmentation).

Architecturally identical to the TF/Keras originals in ``utilities/model.py``
but written as ``nn.Module``s with PyTorch (NCHW) conventions.

The four models are:
    1. LabelEncoder   — compress a binary mask into a latent code
    2. LabelDecoder   — reconstruct a segmentation from the denoised latent
    3. ImageEncoder    — encode a raw image into a conditioning embedding
    4. Denoiser        — conditional U-Net that predicts noise at each timestep

Every model class exposes its hyper-parameters as constructor arguments so
that layer counts, filter widths, attention heads, etc. can be changed
without editing the source.

Author: auto-ported from Fahim Ahmed Zaman's TF implementation
"""

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================================== #
#  Weight-initialisation helpers                                             #
# ========================================================================== #

def _variance_scaling_init_(tensor: torch.Tensor, scale: float = 1.0):
    """Fan-avg uniform init (matches Keras ``VarianceScaling(scale, fan_avg, uniform)``)."""
    scale = max(scale, 1e-10)
    fan_in = tensor.shape[1] if tensor.dim() >= 2 else tensor.shape[0]
    fan_out = tensor.shape[0]
    if tensor.dim() > 2:
        receptive_field = 1
        for s in tensor.shape[2:]:
            receptive_field *= s
        fan_in *= receptive_field
        fan_out *= receptive_field
    limit = math.sqrt(3.0 * scale / ((fan_in + fan_out) / 2.0))
    nn.init.uniform_(tensor, -limit, +limit)


def _init_conv(module: nn.Module, scale: float = 1.0):
    """Apply ``_variance_scaling_init_`` to all Conv2d / ConvTranspose2d / Linear layers inside *module*."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            _variance_scaling_init_(m.weight, scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ========================================================================== #
#  Activation helper                                                         #
# ========================================================================== #

_ACTIVATIONS = {
    "swish": F.silu,
    "silu": F.silu,
    "relu": F.relu,
    "gelu": F.gelu,
}


def _get_act(name_or_fn):
    """Return an activation callable from a string name or passthrough."""
    if callable(name_or_fn):
        return name_or_fn
    return _ACTIVATIONS[name_or_fn]


# ========================================================================== #
#  Building blocks                                                           #
# ========================================================================== #

class ResConvBlock(nn.Module):
    """Residual convolutional block (mirrors ``res_conv_block`` in TF code).

    ``Conv → [BN] → Act → Conv → [BN] → [Dropout]``  +  ``1×1 shortcut [→ BN]``
    → Add → Act

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    dropout : float
    use_bn : bool
    activation : str or callable
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
        use_bn: bool = True,
        activation: str = "swish",
    ):
        super().__init__()
        pad = kernel_size // 2
        self.act_fn = _get_act(activation)
        self.use_bn = use_bn

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Shortcut
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        self.bn_sc = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        # He-uniform initialisation
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        h = self.act_fn(self.bn1(self.conv1(x)))
        h = self.drop(self.bn2(self.conv2(h)))
        # Shortcut
        sc = self.bn_sc(self.shortcut(x))
        return self.act_fn(h + sc)


class Downsample(nn.Module):
    """Spatial downsample: Conv2d with stride 2."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        _init_conv(self, scale=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsample: interpolate × 2 then Conv2d."""

    def __init__(self, channels: int, mode: str = "nearest"):
        super().__init__()
        self.mode = mode
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        _init_conv(self, scale=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode=self.mode)
        return self.conv(x)


# ---------- Image-Encoder conv block -------------------------------------- #

class ConvBlock(nn.Module):
    """Two convs + residual + activation + DownSample + GroupNorm.

    Mirrors ``conv_block`` in the TF code (used by ImageEncoder).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 4,
        dropout: float = 0.2,
        activation: str = "swish",
    ):
        super().__init__()
        pad = kernel_size // 2
        self.act_fn = _get_act(activation)

        self.residual_proj = nn.Conv2d(in_channels, out_channels, 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad)

        self.down = Downsample(out_channels)
        self.gn = nn.GroupNorm(groups, out_channels)

        _init_conv(self.residual_proj, scale=1.0)
        _init_conv(self.conv1, scale=1.0)
        _init_conv(self.conv2, scale=0.0)  # zero-init like TF

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        h = self.act_fn(x)
        h = self.conv1(h)
        h = self.drop(h)
        h = self.act_fn(h)
        h = self.conv2(h)
        h = h + residual
        h = self.act_fn(h)
        h = self.down(h)
        h = self.gn(h)
        return h


# ---------- Attention ----------------------------------------------------- #

class MultiHeadAttentionBlock(nn.Module):
    """Multi-head self-attention with GroupNorm and residual (NCHW)."""

    def __init__(self, channels: int, num_heads: int = 8, groups: int = 8):
        super().__init__()
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

        _init_conv(self.qkv, scale=1.0)
        _init_conv(self.proj, scale=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        # Norm → reshape to (B, H*W, C)
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, C)

        # Q, K, V
        qkv = self.qkv(h).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, heads, N, head_dim)

        # Merge heads
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out)

        # Reshape back to (B, C, H, W)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return residual + out


class AttentionBlock(nn.Module):
    """Single-head self-attention with GroupNorm and residual (NCHW).

    Mirrors the ``AttentionBlock`` in the TF code (used by Denoiser).
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.channels = channels
        self.scale = channels ** -0.5

        self.norm = nn.GroupNorm(groups, channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.proj = nn.Linear(channels, channels)

        _init_conv(self.q, scale=1.0)
        _init_conv(self.k, scale=1.0)
        _init_conv(self.v, scale=1.0)
        _init_conv(self.proj, scale=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, C)

        q = self.q(h)  # (B, N, C)
        k = self.k(h)
        v = self.v(h)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, N, N)
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, N, C)

        out = self.proj(out)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + out


# ---------- Time embedding ------------------------------------------------ #

class TimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        emb = math.log(10_000) / (half - 1)
        self.register_buffer("freqs", torch.exp(torch.arange(half, dtype=torch.float32) * -emb))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) scalar timesteps.
        Returns:
            (B, dim) sinusoidal embeddings.
        """
        t = t.float()
        emb = t[:, None] * self.freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimeMLP(nn.Module):
    """Variable-depth MLP for time embeddings: (Dense → Act) × (depth-1) → Dense.

    Parameters
    ----------
    units : int
        Width of every linear layer (input, hidden, and output all share the same width).
    depth : int
        Total number of linear layers. Must be >= 2. Default 2 matches the original.
    activation : str
        Activation applied between layers.
    """

    def __init__(self, units: int, depth: int = 2, activation: str = "swish"):
        super().__init__()
        assert depth >= 2, "TimeMLP depth must be at least 2"
        self.act_fn = _get_act(activation)
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(units, units))
        self.layers = nn.ModuleList(layers)
        _init_conv(self, scale=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # activation after every layer except the last
                x = self.act_fn(x)
        return x


# ---------- Denoiser residual block --------------------------------------- #

class DenoiserResBlock(nn.Module):
    """Residual block with time-embedding injection (used by Denoiser).

    ``GroupNorm → Act → Conv → (+ time_emb) → GroupNorm → Act → Conv → (+ residual)``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_dim: int,
        groups: int = 8,
        activation: str = "swish",
    ):
        super().__init__()
        self.act_fn = _get_act(activation)

        # Channel projection for residual when dimensions differ
        self.skip_proj = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels
            else nn.Identity()
        )

        self.gn1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Time embedding projection: temb_dim → out_channels
        self.time_proj = nn.Linear(temb_dim, out_channels)

        self.gn2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        _init_conv(self.conv1, scale=1.0)
        _init_conv(self.conv2, scale=0.0)
        if isinstance(self.skip_proj, nn.Conv2d):
            _init_conv(self.skip_proj, scale=1.0)
        _init_conv(self.time_proj, scale=1.0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)
            temb: (B, C_temb) time embedding
        """
        residual = self.skip_proj(x)

        h = self.act_fn(self.gn1(x))
        h = self.conv1(h)

        # Inject time embedding: project then broadcast to (B, C_out, 1, 1)
        t = self.act_fn(temb)
        t = self.time_proj(t)[:, :, None, None]
        h = h + t

        h = self.act_fn(self.gn2(h))
        h = self.conv2(h)
        return h + residual


# ========================================================================== #
#  Model 1: Label Encoder                                                    #
# ========================================================================== #

class LabelEncoder(nn.Module):
    """Compress a binary segmentation mask to a latent code.

    Architecture: stacked ``ResConvBlock``s with ``MaxPool2d`` between them,
    then a 1×1 conv to 1 channel and LayerNorm.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for a single-channel mask).
    elayers : list[int]
        Filter multipliers per stage. Length determines number of stages.
        Default ``[1, 2, 4, 4, 2]`` (5 stages, 4 pooling ops → 16× downsample).
    filter_num : int
        Base filter count; each stage uses ``elayers[i] * filter_num`` channels.
    filter_size : int
        Kernel size for ResConvBlock convolutions.
    dropout : float
    use_bn : bool
    activation : str
    """

    def __init__(
        self,
        in_channels: int = 1,
        elayers: Sequence[int] = (1, 2, 4, 4, 2),
        filter_num: int = 16,
        filter_size: int = 3,
        dropout: float = 0.2,
        use_bn: bool = True,
        activation: str = "swish",
    ):
        super().__init__()
        self.elayers = list(elayers)

        blocks = []
        ch_in = in_channels
        for i, mult in enumerate(elayers):
            ch_out = mult * filter_num
            blocks.append(ResConvBlock(ch_in, ch_out, filter_size, dropout, use_bn, activation))
            if i < len(elayers) - 1:
                blocks.append(nn.MaxPool2d(2))
            ch_in = ch_out
        self.blocks = nn.Sequential(*blocks)

        self.proj = nn.Conv2d(ch_in, 1, 1)
        # LayerNorm over (C, H, W) — we use a wrapper since spatial dims are dynamic
        self.norm = _ChannelLayerNorm(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W)
        Returns:
            (B, 1, H/16, W/16)  (assuming 4 pooling stages)
        """
        h = self.blocks(x)
        h = self.proj(h)
        return self.norm(h)


class _ChannelLayerNorm(nn.Module):
    """LayerNorm applied over (C, H, W) that works with dynamic spatial dims.

    Equivalent to ``keras.layers.LayerNormalization(axis=(1,2,3))`` which normalises
    across all spatial + channel dims per sample.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise over (C, H, W) per sample
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / (var + 1e-5).sqrt()
        return x * self.weight + self.bias


# ========================================================================== #
#  Model 2: Label Decoder                                                    #
# ========================================================================== #

class LabelDecoder(nn.Module):
    """Reconstruct a segmentation mask from the denoised latent code.

    Architecture: stacked ``ConvTranspose2d`` (stride 2) + ``ResConvBlock``,
    then swish → 1×1 Conv → BatchNorm → Softmax.

    Parameters
    ----------
    in_channels : int
        Latent channels (1 for the default encoder output).
    dlayers : list[int]
        Filter multipliers per upsample stage. Length = number of upsample ops.
        Default ``[2, 4, 4, 2]`` (4 stages → 16× upsample).
    filter_num : int
        Base filter count.
    filter_size : int
        Kernel size for ResConvBlock.
    dropout : float
    use_bn : bool
    num_classes : int
        Number of output segmentation classes.
    activation : str
    """

    def __init__(
        self,
        in_channels: int = 1,
        dlayers: Sequence[int] = (2, 4, 4, 2),
        filter_num: int = 16,
        filter_size: int = 3,
        dropout: float = 0.2,
        use_bn: bool = True,
        num_classes: int = 2,
        activation: str = "swish",
    ):
        super().__init__()
        self.dlayers = list(dlayers)

        upblocks = nn.ModuleList()
        resblocks = nn.ModuleList()
        ch_in = in_channels
        for mult in dlayers:
            ch_out = mult * filter_num
            upblocks.append(nn.ConvTranspose2d(ch_in, ch_out, 3, stride=2, padding=1, output_padding=1))
            resblocks.append(ResConvBlock(ch_out, ch_out, filter_size, dropout, use_bn, activation))
            ch_in = ch_out

        self.upblocks = upblocks
        self.resblocks = resblocks

        self.head_conv = nn.Conv2d(ch_in, num_classes, 1)
        self.head_bn = nn.BatchNorm2d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H_lat, W_lat)
        Returns:
            (B, num_classes, H, W) — softmax probabilities.
        """
        for up, res in zip(self.upblocks, self.resblocks):
            x = up(x)
            x = res(x)

        x = F.silu(x)
        x = self.head_conv(x)
        x = self.head_bn(x)
        return x.softmax(dim=1)


# ========================================================================== #
#  Model 3: Image Encoder                                                    #
# ========================================================================== #

class ImageEncoder(nn.Module):
    """Encode a raw image into a conditioning embedding for the denoiser.

    Architecture: initial Conv → repeated ``ConvBlock`` + ``MultiHeadAttentionBlock``
    stages → 1×1 Conv → MultiHeadAttention → activation → BatchNorm.

    Parameters
    ----------
    in_channels : int
        Image channels (3 for RGB).
    filter_size : int
        Base filter count.
    kernel_size : int
        Conv kernel size within ``ConvBlock``.
    dropout : float
    groups : int
        GroupNorm groups inside ConvBlock and attention.
    out_channels : int
        Output embedding channels (1 to match latent space).
    block_mults : list[int]
        Filter multipliers for the main ConvBlock stages.
        Default ``[2, 4, 4, 2]`` — same as the TF code:
        ``[2*fs, 4*fs, 4*fs, 2*fs]``.
    attention_after : list[int]
        Which ConvBlock indices should be followed by a MultiHeadAttention.
        Default ``[2, 3]`` (0-indexed), matching the TF code layout.
    activation : str
    """

    def __init__(
        self,
        in_channels: int = 3,
        filter_size: int = 16,
        kernel_size: int = 3,
        dropout: float = 0.2,
        groups: int = 4,
        out_channels: int = 1,
        block_mults: Sequence[int] = (2, 4, 4, 2),
        attention_after: Sequence[int] = (2, 3),
        activation: str = "swish",
    ):
        super().__init__()
        self.act_fn = _get_act(activation)

        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, filter_size, 3, padding=1)
        _init_conv(self.init_conv, scale=1.0)

        # Build stages
        self.stages = nn.ModuleList()
        ch_in = filter_size
        attention_set = set(attention_after)
        for idx, mult in enumerate(block_mults):
            ch_out = mult * filter_size
            stage = nn.ModuleDict({
                "conv_block": ConvBlock(ch_in, ch_out, kernel_size, groups, dropout, activation),
            })
            if idx in attention_set:
                stage["attn"] = MultiHeadAttentionBlock(ch_out, num_heads=8, groups=groups)
            self.stages.append(stage)
            ch_in = ch_out

        # Final projection
        self.final_conv = nn.Conv2d(ch_in, out_channels, 1)
        _init_conv(self.final_conv, scale=0.0)

        self.final_attn = MultiHeadAttentionBlock(
            out_channels, num_heads=max(1, out_channels), groups=1
        )
        self.final_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, out_channels, H/16, W/16)
        """
        h = self.init_conv(x)

        for stage in self.stages:
            h = stage["conv_block"](h)
            if "attn" in stage:
                h = stage["attn"](h)

        h = self.final_conv(h)
        h = self.final_attn(h)
        h = self.act_fn(h)
        return self.final_bn(h)


# ========================================================================== #
#  Model 4: Denoiser (conditional U-Net)                                     #
# ========================================================================== #

class Denoiser(nn.Module):
    """Conditional U-Net that predicts noise for each diffusion timestep.

    Parameters
    ----------
    latent_channels : int
        Channels of the noisy latent input (1 for default LDSeg).
    cond_channels : int
        Channels of the image-encoder conditioning embedding (1 for default).
    first_conv_channels : int
        Channels after the initial conv on the concatenated latent + cond.
    widths : list[int]
        Channel widths for each U-Net resolution level.
    has_attention : list[bool]
        Whether to add attention after residual blocks at each level.
    num_res_blocks : int
        Number of residual blocks per level in the down path.
    norm_groups : int
        Groups for ``GroupNorm``.
    out_channels : int
        Output channels (predicted noise). Defaults to ``latent_channels``.
    interpolation : str
        Interpolation mode for upsampling (``"nearest"`` or ``"bilinear"``).
    activation : str
    """

    def __init__(
        self,
        latent_channels: int = 1,
        cond_channels: int = 1,
        first_conv_channels: int = 16,
        widths: Sequence[int] = (16, 32, 64),
        has_attention: Sequence[bool] = (False, True, True),
        num_res_blocks: int = 2,
        norm_groups: int = 4,
        out_channels: Optional[int] = None,
        interpolation: str = "nearest",
        activation: str = "swish",
        time_mlp_depth: int = 2,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = latent_channels

        widths = list(widths)
        has_attention = list(has_attention)
        assert len(widths) == len(has_attention)
        self.act_fn = _get_act(activation)

        # ---- Initial conv ------------------------------------------------ #
        in_ch = latent_channels + cond_channels
        self.init_conv = nn.Conv2d(in_ch, first_conv_channels, 3, padding=1)
        _init_conv(self.init_conv, scale=1.0)

        # ---- Time embedding ---------------------------------------------- #
        temb_dim = first_conv_channels * 4
        self.time_emb = TimeEmbedding(temb_dim)
        self.time_mlp = TimeMLP(temb_dim, depth=time_mlp_depth, activation=activation)

        # ---- Down path --------------------------------------------------- #
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = first_conv_channels
        skip_channels = [ch]  # track skip shapes for the up path

        for i, (w, use_attn) in enumerate(zip(widths, has_attention)):
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(DenoiserResBlock(ch, w, temb_dim, norm_groups, activation))
                if use_attn:
                    level_blocks.append(AttentionBlock(w, norm_groups))
                ch = w
                skip_channels.append(ch)

            self.down_blocks.append(level_blocks)

            if i < len(widths) - 1:
                ds = Downsample(ch)
                self.down_samples.append(ds)
                skip_channels.append(ch)
            else:
                self.down_samples.append(None)

        # ---- Middle ------------------------------------------------------ #
        self.mid_res1 = DenoiserResBlock(ch, widths[-1], temb_dim, norm_groups, activation)
        self.mid_attn = AttentionBlock(widths[-1], norm_groups)
        self.mid_res2 = DenoiserResBlock(widths[-1], widths[-1], temb_dim, norm_groups, activation)

        # ---- Up path ----------------------------------------------------- #
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i in reversed(range(len(widths))):
            w = widths[i]
            use_attn = has_attention[i]
            level_blocks = nn.ModuleList()

            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                level_blocks.append(DenoiserResBlock(ch + skip_ch, w, temb_dim, norm_groups, activation))
                if use_attn:
                    level_blocks.append(AttentionBlock(w, norm_groups))
                ch = w

            self.up_blocks.append(level_blocks)

            if i > 0:
                self.up_samples.append(Upsample(ch, interpolation))
            else:
                self.up_samples.append(None)

        # ---- End block --------------------------------------------------- #
        self.final_gn = nn.GroupNorm(norm_groups, ch)
        self.final_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z:    (B, latent_channels, H, W) — noisy latent
            cond: (B, cond_channels, H, W)   — image-encoder embedding
            t:    (B,)                        — diffusion timestep
        Returns:
            (B, out_channels, H, W) — predicted noise
        """
        # Concatenate latent + conditioning
        x = torch.cat([z, cond], dim=1)
        x = self.init_conv(x)

        # Time embedding
        temb = self.time_emb(t)
        temb = self.time_mlp(temb)

        # ---- Down -------------------------------------------------------- #
        # ---- Down -------------------------------------------------------- #
        skips = [x]
        for level_blocks, ds in zip(self.down_blocks, self.down_samples):
            idx = 0
            while idx < len(level_blocks):
                # Always starts with DenoiserResBlock
                block = level_blocks[idx]
                x = block(x, temb)
                idx += 1

                # Optional AttentionBlock
                if idx < len(level_blocks) and isinstance(level_blocks[idx], AttentionBlock):
                    x = level_blocks[idx](x)
                    idx += 1
                
                skips.append(x)

            if ds is not None:
                x = ds(x)
                skips.append(x)

        # ---- Middle ------------------------------------------------------ #
        x = self.mid_res1(x, temb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, temb)

        # ---- Up ---------------------------------------------------------- #
        for level_blocks, us in zip(self.up_blocks, self.up_samples):
            idx = 0
            while idx < len(level_blocks):
                block = level_blocks[idx]
                if isinstance(block, DenoiserResBlock):
                    x = torch.cat([x, skips.pop()], dim=1)
                    x = block(x, temb)
                else:  # AttentionBlock
                    x = block(x)
                idx += 1

            if us is not None:
                x = us(x)

        # ---- End --------------------------------------------------------- #
        x = self.act_fn(self.final_gn(x))
        return self.final_conv(x)


# ========================================================================== #
#  Convenience: build all 4 models with default parameters                   #
# ========================================================================== #

def build_models(
    image_size: tuple = (512, 512),
    image_channels: int = 3,
    num_classes: int = 2,
    filter_num: int = 16,
    elayers: Sequence[int] = (1, 2, 4, 4, 2),
    dlayers: Sequence[int] = (2, 4, 4, 2),
    denoiser_widths: Sequence[int] = (16, 32, 64),
    denoiser_attention: Sequence[bool] = (False, True, True),
    denoiser_res_blocks: int = 2,
    norm_groups: int = 4,
):
    """Build all four LDSeg model components with matching shapes.

    Returns:
        (LabelEncoder, LabelDecoder, ImageEncoder, Denoiser)
    """
    H, W = image_size

    label_encoder = LabelEncoder(
        in_channels=1,
        elayers=elayers,
        filter_num=filter_num,
    )
    # Determine latent spatial dims
    num_pools = len(elayers) - 1
    lat_h, lat_w = H // (2 ** num_pools), W // (2 ** num_pools)

    label_decoder = LabelDecoder(
        in_channels=1,
        dlayers=dlayers,
        filter_num=filter_num,
        num_classes=num_classes,
    )

    image_encoder = ImageEncoder(
        in_channels=image_channels,
        filter_size=filter_num,
        out_channels=1,
    )

    denoiser = Denoiser(
        latent_channels=1,
        cond_channels=1,
        first_conv_channels=filter_num,
        widths=denoiser_widths,
        has_attention=denoiser_attention,
        num_res_blocks=denoiser_res_blocks,
        norm_groups=norm_groups,
    )

    return label_encoder, label_decoder, image_encoder, denoiser


# ========================================================================== #
#  Build from config file                                                     #
# ========================================================================== #

def build_models_from_config(
    config_path: str = "model_config.ini",
):
    """Build all four LDSeg model components from an INI config file.

    Parameters
    ----------
    config_path : str
        Path to the ``model_config.ini`` file.

    Returns
    -------
    (LabelEncoder, LabelDecoder, ImageEncoder, Denoiser)
    """
    import configparser

    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    # ---- helpers --------------------------------------------------------- #
    def _int(section, key):
        return cfg.getint(section, key)

    def _float(section, key):
        return cfg.getfloat(section, key)

    def _bool(section, key):
        return cfg.getboolean(section, key)

    def _str(section, key):
        return cfg.get(section, key).strip()

    def _int_list(section, key):
        return [int(x.strip()) for x in cfg.get(section, key).split(",")]

    def _bool_list(section, key):
        return [x.strip().lower() in ("true", "1", "yes") for x in cfg.get(section, key).split(",")]

    # ---- Label Encoder --------------------------------------------------- #
    label_encoder = LabelEncoder(
        in_channels=_int("LabelEncoder", "InChannels"),
        elayers=_int_list("LabelEncoder", "EncoderLayers"),
        filter_num=_int("LabelEncoder", "FilterNum"),
        filter_size=_int("LabelEncoder", "FilterSize"),
        dropout=_float("LabelEncoder", "Dropout"),
        use_bn=_bool("LabelEncoder", "UseBatchNorm"),
        activation=_str("LabelEncoder", "Activation"),
    )

    # ---- Label Decoder --------------------------------------------------- #
    label_decoder = LabelDecoder(
        in_channels=_int("LabelDecoder", "InChannels"),
        dlayers=_int_list("LabelDecoder", "DecoderLayers"),
        filter_num=_int("LabelDecoder", "FilterNum"),
        filter_size=_int("LabelDecoder", "FilterSize"),
        dropout=_float("LabelDecoder", "Dropout"),
        use_bn=_bool("LabelDecoder", "UseBatchNorm"),
        num_classes=_int("LabelDecoder", "NumClasses"),
        activation=_str("LabelDecoder", "Activation"),
    )

    # ---- Image Encoder --------------------------------------------------- #
    image_encoder = ImageEncoder(
        in_channels=_int("ImageEncoder", "InChannels"),
        filter_size=_int("ImageEncoder", "FilterSize"),
        kernel_size=_int("ImageEncoder", "KernelSize"),
        dropout=_float("ImageEncoder", "Dropout"),
        groups=_int("ImageEncoder", "Groups"),
        out_channels=_int("ImageEncoder", "OutChannels"),
        block_mults=_int_list("ImageEncoder", "BlockMults"),
        attention_after=_int_list("ImageEncoder", "AttentionAfter"),
        activation=_str("ImageEncoder", "Activation"),
    )

    # ---- Denoiser -------------------------------------------------------- #
    denoiser = Denoiser(
        latent_channels=_int("Denoiser", "LatentChannels"),
        cond_channels=_int("Denoiser", "CondChannels"),
        first_conv_channels=_int("Denoiser", "FirstConvChannels"),
        widths=_int_list("Denoiser", "Widths"),
        has_attention=_bool_list("Denoiser", "HasAttention"),
        num_res_blocks=_int("Denoiser", "NumResBlocks"),
        norm_groups=_int("Denoiser", "NormGroups"),
        interpolation=_str("Denoiser", "Interpolation"),
        activation=_str("Denoiser", "Activation"),
    )

    return label_encoder, label_decoder, image_encoder, denoiser
