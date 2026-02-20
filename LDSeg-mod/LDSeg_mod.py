"""
End-to-end LDSeg model (PyTorch).

Combines:
  - The four segmentation model components from ``utilities.model_torch``
    (LabelEncoder, LabelDecoder, ImageEncoder, Denoiser)
  - The probabilistic distribution modules from ``guided_diffusion/distribution.py``
    (AxisAlignedConvGaussian for prior & posterior)

All hyperparameters are loaded from ``model_config.ini``.
"""

import configparser
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, kl

from utilities.model_torch import (
    LabelEncoder,
    LabelDecoder,
    ImageEncoder,
    Denoiser,
)

# ========================================================================== #
#  Utility functions (from guided_diffusion/utils.py)                         #
# ========================================================================== #

def truncated_normal_(tensor: torch.Tensor, mean: float = 0, std: float = 1):
    """Fill *tensor* with values drawn from a truncated normal (±2σ)."""
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m: nn.Module):
    """Kaiming-normal init for Conv2d / ConvTranspose2d layers."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m: nn.Module):
    """Orthogonal init for Conv2d / ConvTranspose2d layers."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)


def l2_regularisation(m: nn.Module) -> torch.Tensor:
    """Compute the L2 norm of all parameters in *m*."""
    l2_reg = None
    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


# ========================================================================== #
#  Distribution modules (from guided_diffusion/distribution.py)               #
# ========================================================================== #

class DistEncoder(nn.Module):
    """CNN that progressively downsamples via AvgPool2d blocks.

    Each block consists of ``no_convs_per_block`` Conv2d + ReLU layers.
    When ``posterior=True`` the input channel count is increased by 1 to
    accommodate a concatenated segmentation mask.
    """

    def __init__(
        self,
        input_channels: int,
        num_filters: Sequence[int],
        no_convs_per_block: int,
        padding: bool = True,
        posterior: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_filters = list(num_filters)

        if posterior:
            self.input_channels += 1

        layers = []
        for i, out_dim in enumerate(self.num_filters):
            in_dim = self.input_channels if i == 0 else self.num_filters[i - 1]

            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.BatchNorm2d(out_dim))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.BatchNorm2d(out_dim))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AxisAlignedConvGaussian(nn.Module):
    """Convolutional network that parametrises a diagonal Gaussian.

    Returns a ``torch.distributions.Independent(Normal(...), 1)`` whose
    ``loc`` and ``scale`` have shape ``(B, latent_dim)``.

    Parameters
    ----------
    input_channels : int
        Image channels (before optional mask concatenation).
    num_filters : list[int]
        Filter widths per encoder block.
    no_convs_per_block : int
        Conv layers per block.
    latent_dim : int
        Dimensionality of μ and log σ.
    posterior : bool
        If ``True`` the encoder expects ``(image, segm)`` concatenated along
        the channel axis and adds +1 to its input channels.
    """

    def __init__(
        self,
        input_channels: int,
        num_filters: Sequence[int],
        no_convs_per_block: int,
        latent_dim: int,
        posterior: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_filters = list(num_filters)
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior

        self.encoder = DistEncoder(
            self.input_channels,
            self.num_filters,
            self.no_convs_per_block,
            posterior=self.posterior,
        )
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1, 1), stride=1)

        nn.init.kaiming_normal_(self.conv_layer.weight, mode="fan_in", nonlinearity="relu")
        nn.init.normal_(self.conv_layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        segm: Optional[torch.Tensor] = None,
    ) -> Independent:
        """
        Args:
            x:    (B, input_channels, H, W)  — image patch.
            segm: (B, 1, H, W) or ``None``  — segmentation mask (posterior only).

        Returns:
            A diagonal multivariate Normal distribution with ``event_shape = (latent_dim,)``.
        """
        if segm is not None:
            x = torch.cat((x, segm), dim=1)

        encoding = self.encoder(x)

        # Global average pool → (B, C, 1, 1)
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        # Project to 2 × latent_dim → split into μ and log σ
        mu_log_sigma = self.conv_layer(encoding)
        mu_log_sigma = mu_log_sigma.squeeze(-1).squeeze(-1)  # (B, 2*latent_dim)

        mu = mu_log_sigma[:, : self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim :]

        return Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)


# ========================================================================== #
#  End-to-end LDSeg model                                                    #
# ========================================================================== #

class LDSeg(nn.Module):
    """End-to-end Latent Diffusion Segmentation model.

    Bundles the four LDSeg components (LabelEncoder, LabelDecoder,
    ImageEncoder, Denoiser) together with prior and posterior distribution
    networks (AxisAlignedConvGaussian).

    Parameters
    ----------
    label_encoder : LabelEncoder
    label_decoder : LabelDecoder
    image_encoder : ImageEncoder
    denoiser : Denoiser
    prior : AxisAlignedConvGaussian
    posterior : AxisAlignedConvGaussian
    """

    def __init__(
        self,
        label_encoder: LabelEncoder,
        label_decoder: LabelDecoder,
        image_encoder: ImageEncoder,
        denoiser: Denoiser,
        prior: AxisAlignedConvGaussian,
        posterior: AxisAlignedConvGaussian,
    ):
        super().__init__()
        self.label_encoder = label_encoder
        self.label_decoder = label_decoder
        self.image_encoder = image_encoder
        self.denoiser = denoiser
        self.prior = prior
        self.posterior = posterior

    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        timestep: torch.Tensor,
        noisy_encoded: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """Run the full forward pass.

        Args:
            image:         (B, C_img, H, W)      — input image.
            mask:          (B, 1, H, W)           — ground-truth binary segmentation mask.
            timestep:      (B,)                   — diffusion timestep for each sample.
            noisy_encoded: (B, 1, H_lat, W_lat)   — pre-noised latent from a diffusion
                           schedule (e.g. ``q_sample``).  If ``None``, simple additive
                           Gaussian noise is used as a fallback (useful for smoke tests
                           only — real training should supply this from the scheduler).

        Returns:
            dict with keys:
              - ``encoded``       : label encoder output (B, 1, H_lat, W_lat)
              - ``noisy_encoded`` : noisy latent fed to the denoiser
              - ``decoded``       : label decoder reconstruction (B, num_classes, H, W)
              - ``img_embedding`` : image encoder conditioning (B, 1, H_lat, W_lat)
              - ``denoiser_out``  : predicted noise (B, 1, H_lat, W_lat)
              - ``prior_dist``    : prior distribution (Independent Normal)
              - ``posterior_dist``: posterior distribution (Independent Normal)
              - ``kl_div``        : KL(posterior || prior), scalar per sample (B,)
        """
        # 1. Encode the ground-truth mask into latent space
        encoded = self.label_encoder(mask)  # (B, 1, H_lat, W_lat)

        # 2. Obtain the noisy latent
        #    In a real training loop the caller should compute this via the
        #    diffusion schedule's q_sample(encoded, t, noise).  The fallback
        #    below is only for simple testing purposes.
        if noisy_encoded is None:
            noise = torch.randn_like(encoded)
            noisy_encoded = encoded + noise

        # 3. Encode the image into a conditioning embedding
        img_embedding = self.image_encoder(image)  # (B, 1, H_lat, W_lat)

        # 4. Denoise: predict noise from *noisy* latent + image embedding
        denoiser_out = self.denoiser(noisy_encoded, img_embedding, timestep)

        # 5. Decode the clean encoded latent back to a segmentation mask
        decoded = self.label_decoder(encoded)  # (B, num_classes, H, W)

        # 6. Prior & Posterior distributions
        #    Both operate at latent resolution so that spatial dims match.
        #    Prior  sees (image_embedding, denoiser_prediction)
        #    Posterior sees (image_embedding, clean_encoded_mask)
        prior_dist = self.prior(img_embedding, denoiser_out)
        posterior_dist = self.posterior(img_embedding, encoded)

        # 7. KL divergence
        kl_div = kl.kl_divergence(posterior_dist, prior_dist)  # (B,)

        return {
            "encoded": encoded,
            "noisy_encoded": noisy_encoded,
            "decoded": decoded,
            "img_embedding": img_embedding,
            "denoiser_out": denoiser_out,
            "prior_dist": prior_dist,
            "posterior_dist": posterior_dist,
            "kl_div": kl_div,
        }


# ========================================================================== #
#  Config helpers                                                             #
# ========================================================================== #

def _int(cfg, section, key):
    return cfg.getint(section, key)

def _float(cfg, section, key):
    return cfg.getfloat(section, key)

def _bool(cfg, section, key):
    return cfg.getboolean(section, key)

def _str(cfg, section, key):
    return cfg.get(section, key).strip()

def _int_list(cfg, section, key):
    return [int(x.strip()) for x in cfg.get(section, key).split(",")]

def _bool_list(cfg, section, key):
    return [x.strip().lower() in ("true", "1", "yes") for x in cfg.get(section, key).split(",")]


def build_ldseg_from_config(config_path: str = "model_config.ini") -> LDSeg:
    """Build the complete LDSeg model from an INI config file.

    Parameters
    ----------
    config_path : str
        Path to ``model_config.ini``.

    Returns
    -------
    LDSeg
        The end-to-end model with all six sub-modules.
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    # ---- Label Encoder --------------------------------------------------- #
    label_encoder = LabelEncoder(
        in_channels=_int(cfg, "LabelEncoder", "InChannels"),
        elayers=_int_list(cfg, "LabelEncoder", "EncoderLayers"),
        filter_num=_int(cfg, "LabelEncoder", "FilterNum"),
        filter_size=_int(cfg, "LabelEncoder", "FilterSize"),
        dropout=_float(cfg, "LabelEncoder", "Dropout"),
        use_bn=_bool(cfg, "LabelEncoder", "UseBatchNorm"),
        activation=_str(cfg, "LabelEncoder", "Activation"),
        blocks_per_stage=_int_list(cfg, "LabelEncoder", "BlocksPerStage") if cfg.has_option("LabelEncoder", "BlocksPerStage") else None,
    )

    # ---- Label Decoder --------------------------------------------------- #
    label_decoder = LabelDecoder(
        in_channels=_int(cfg, "LabelDecoder", "InChannels"),
        dlayers=_int_list(cfg, "LabelDecoder", "DecoderLayers"),
        filter_num=_int(cfg, "LabelDecoder", "FilterNum"),
        filter_size=_int(cfg, "LabelDecoder", "FilterSize"),
        dropout=_float(cfg, "LabelDecoder", "Dropout"),
        use_bn=_bool(cfg, "LabelDecoder", "UseBatchNorm"),
        num_classes=_int(cfg, "LabelDecoder", "NumClasses"),
        activation=_str(cfg, "LabelDecoder", "Activation"),
        blocks_per_stage=_int_list(cfg, "LabelDecoder", "BlocksPerStage") if cfg.has_option("LabelDecoder", "BlocksPerStage") else None,
    )

    # ---- Image Encoder --------------------------------------------------- #
    image_encoder = ImageEncoder(
        in_channels=_int(cfg, "ImageEncoder", "InChannels"),
        filter_size=_int(cfg, "ImageEncoder", "FilterSize"),
        kernel_size=_int(cfg, "ImageEncoder", "KernelSize"),
        dropout=_float(cfg, "ImageEncoder", "Dropout"),
        groups=_int(cfg, "ImageEncoder", "Groups"),
        out_channels=_int(cfg, "ImageEncoder", "OutChannels"),
        block_mults=_int_list(cfg, "ImageEncoder", "BlockMults"),
        attention_after=_int_list(cfg, "ImageEncoder", "AttentionAfter"),
        activation=_str(cfg, "ImageEncoder", "Activation"),
        blocks_per_stage=_int_list(cfg, "ImageEncoder", "BlocksPerStage") if cfg.has_option("ImageEncoder", "BlocksPerStage") else None,
        no_downsample_at=_int_list(cfg, "ImageEncoder", "NoDownsampleAt") if cfg.has_option("ImageEncoder", "NoDownsampleAt") else None,
    )

    # ---- Denoiser -------------------------------------------------------- #
    denoiser = Denoiser(
        latent_channels=_int(cfg, "Denoiser", "LatentChannels"),
        cond_channels=_int(cfg, "Denoiser", "CondChannels"),
        first_conv_channels=_int(cfg, "Denoiser", "FirstConvChannels"),
        widths=_int_list(cfg, "Denoiser", "Widths"),
        has_attention=_bool_list(cfg, "Denoiser", "HasAttention"),
        num_res_blocks=_int(cfg, "Denoiser", "NumResBlocks"),
        norm_groups=_int(cfg, "Denoiser", "NormGroups"),
        interpolation=_str(cfg, "Denoiser", "Interpolation"),
        activation=_str(cfg, "Denoiser", "Activation"),
        time_mlp_depth=cfg.getint("Denoiser", "TimeMlpDepth", fallback=2),
    )

    # ---- Distribution (prior & posterior) -------------------------------- #
    dist_in_ch = _int(cfg, "Distribution", "InputChannels")
    dist_filters = _int_list(cfg, "Distribution", "NumFilters")
    dist_convs = _int(cfg, "Distribution", "NoConvsPerBlock")
    dist_latent = _int(cfg, "Distribution", "LatentDim")

    # NOTE: Both prior and posterior always receive two concatenated inputs
    # (img_embedding + something), so both need posterior=True to account
    # for the extra channel from concatenation.
    prior = AxisAlignedConvGaussian(
        input_channels=dist_in_ch,
        num_filters=dist_filters,
        no_convs_per_block=dist_convs,
        latent_dim=dist_latent,
        posterior=True,
    )

    posterior = AxisAlignedConvGaussian(
        input_channels=dist_in_ch,
        num_filters=dist_filters,
        no_convs_per_block=dist_convs,
        latent_dim=dist_latent,
        posterior=True,
    )

    model = LDSeg(
        label_encoder=label_encoder,
        label_decoder=label_decoder,
        image_encoder=image_encoder,
        denoiser=denoiser,
        prior=prior,
        posterior=posterior,
    )

    # Store latent size on the model for use by sampling code
    if cfg.has_section("Latent") and cfg.has_option("Latent", "LatentSize"):
        model.latent_size = cfg.getint("Latent", "LatentSize")
    else:
        model.latent_size = None

    return model
