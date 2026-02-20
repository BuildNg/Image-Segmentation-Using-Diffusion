"""
Detailed parameter breakdown for each LDSeg sub-module.

For each component, shows:
  - Total parameters
  - Per-block breakdown (ResConvBlock / ConvBlock / DenoiserResBlock)
  - Parameters in all other layers (stem, head, attention, norm, time embedding, etc.)
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch.nn as nn
from LDSeg_mod import build_ldseg_from_config
from utilities.model_torch import (
    ResConvBlock, ConvBlock, DenoiserResBlock,
    AttentionBlock, MultiHeadAttentionBlock,
    TimeEmbedding, TimeMLP,
    Downsample, Upsample,
)

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_config.ini")
model = build_ldseg_from_config(config_path)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def print_sep(char="─", width=72):
    print(char * width)


def block_type_label(block):
    if isinstance(block, ResConvBlock):       return "ResConvBlock"
    if isinstance(block, ConvBlock):          return "ConvBlock"
    if isinstance(block, DenoiserResBlock):   return "DenoiserResBlock"
    if isinstance(block, AttentionBlock):     return "AttentionBlock (single-head)"
    if isinstance(block, MultiHeadAttentionBlock): return "MultiHeadAttentionBlock"
    if isinstance(block, Downsample):         return "Downsample"
    if isinstance(block, Upsample):           return "Upsample"
    if isinstance(block, TimeEmbedding):      return "TimeEmbedding"
    if isinstance(block, TimeMLP):            return "TimeMLP"
    return type(block).__name__


def report_blocks(named_blocks, indent=4):
    """Print a table of named blocks with their param counts."""
    pad = " " * indent
    total_block_params = 0
    for label, block in named_blocks:
        p = nparams(block)
        total_block_params += p
        print(f"{pad}{label:<52} {p:>12,}")
    return total_block_params


# ─────────────────────────────────────────────────────────────────────────────
# 1. LABEL ENCODER
# ─────────────────────────────────────────────────────────────────────────────
enc = model.label_encoder
print_sep("═")
print(f"  LABEL ENCODER   total params: {nparams(enc):,}")
print_sep("═")

# Collect ResConvBlocks and MaxPool layers from enc.blocks
res_blocks = [(f"ResConvBlock [{i}]", b) for i, b in enumerate(enc.blocks)
              if isinstance(b, ResConvBlock)]
other_params = nparams(enc) - sum(nparams(b) for _, b in res_blocks)

print(f"  {'Layer':<52} {'Params':>12}")
print_sep()
block_total = report_blocks(res_blocks)
print_sep()
print(f"    {'Other (proj 1×1 + LayerNorm)':<50} {other_params:>12,}")
print(f"    {'MaxPool2d layers':<50} {'0':>12}  (no params)")
print_sep()
print(f"    {'TOTAL':<50} {nparams(enc):>12,}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. LABEL DECODER
# ─────────────────────────────────────────────────────────────────────────────
dec = model.label_decoder
print_sep("═")
print(f"  LABEL DECODER   total params: {nparams(dec):,}")
print_sep("═")

res_blocks_dec = [(f"ResConvBlock [{i}]", b) for i, b in enumerate(dec.resblocks)]
up_blocks_dec  = [(f"ConvTranspose2d [{i}]", b) for i, b in enumerate(dec.upblocks)]
head_params    = nparams(dec.head_conv) + nparams(dec.head_bn)

print(f"  {'Layer':<52} {'Params':>12}")
print_sep()
report_blocks(up_blocks_dec)
report_blocks(res_blocks_dec)
print_sep()
print(f"    {'Head (Conv1×1 + BN)':<50} {head_params:>12,}")
print_sep()
print(f"    {'TOTAL':<50} {nparams(dec):>12,}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 3. IMAGE ENCODER
# ─────────────────────────────────────────────────────────────────────────────
img_enc = model.image_encoder
print_sep("═")
print(f"  IMAGE ENCODER   total params: {nparams(img_enc):,}")
print_sep("═")

stage_blocks = []
for i, stage in enumerate(img_enc.stages):
    cb = stage["conv_block"]
    stage_blocks.append((f"ConvBlock [{i}] (includes Downsample + GN)", cb))
    if "attn" in stage:
        stage_blocks.append((f"  └─ MultiHeadAttentionBlock [{i}]", stage["attn"]))

stem_params  = nparams(img_enc.init_conv)
final_params = nparams(img_enc.final_conv) + nparams(img_enc.final_attn) + nparams(img_enc.final_bn)

print(f"  {'Layer':<52} {'Params':>12}")
print_sep()
print(f"    {'Stem Conv2d (init_conv)':<50} {stem_params:>12,}")
report_blocks(stage_blocks)
print_sep()
print(f"    {'Final (Conv1×1 + MHAttn + BN)':<50} {final_params:>12,}")
print_sep()
print(f"    {'TOTAL':<50} {nparams(img_enc):>12,}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. DENOISER
# ─────────────────────────────────────────────────────────────────────────────
den = model.denoiser
print_sep("═")
print(f"  DENOISER (U-Net)   total params: {nparams(den):,}")
print_sep("═")

print(f"  {'Layer':<52} {'Params':>12}")
print_sep()

# Stem
stem_p = nparams(den.init_conv)
print(f"    {'Stem Conv2d (init_conv)':<50} {stem_p:>12,}")

# Time embedding
temb_p = nparams(den.time_emb) + nparams(den.time_mlp)
print(f"    {'TimeEmbedding + TimeMLP':<50} {temb_p:>12,}")
print(f"      {'└─ TimeEmbedding (sinusoidal, no params)':<48} {'0':>12}")
print(f"      {'└─ TimeMLP':<48} {nparams(den.time_mlp):>12,}")

print_sep("·")

# Down path
print(f"    {'── DOWN PATH ──':<50}")
for lvl_idx, (level_blocks, ds) in enumerate(zip(den.down_blocks, den.down_samples)):
    for blk in level_blocks:
        label = f"  Level {lvl_idx} {block_type_label(blk)}"
        print(f"    {label:<50} {nparams(blk):>12,}")
    if ds is not None:
        print(f"    {'  Level ' + str(lvl_idx) + ' Downsample (strided Conv)':<50} {nparams(ds):>12,}")

print_sep("·")

# Middle
mid_p = nparams(den.mid_res1) + nparams(den.mid_attn) + nparams(den.mid_res2)
print(f"    {'── MIDDLE ──':<50}")
print(f"    {'  mid_res1 (DenoiserResBlock)':<50} {nparams(den.mid_res1):>12,}")
print(f"    {'  mid_attn (AttentionBlock)':<50} {nparams(den.mid_attn):>12,}")
print(f"    {'  mid_res2 (DenoiserResBlock)':<50} {nparams(den.mid_res2):>12,}")

print_sep("·")

# Up path
print(f"    {'── UP PATH ──':<50}")
for lvl_idx, (level_blocks, us) in enumerate(zip(den.up_blocks, den.up_samples)):
    for blk in level_blocks:
        label = f"  Level {lvl_idx} {block_type_label(blk)}"
        print(f"    {label:<50} {nparams(blk):>12,}")
    if us is not None:
        print(f"    {'  Level ' + str(lvl_idx) + ' Upsample (interp + Conv)':<50} {nparams(us):>12,}")

print_sep("·")

# Final head
final_p = nparams(den.final_gn) + nparams(den.final_conv)
print(f"    {'Final (GroupNorm + Conv3×3)':<50} {final_p:>12,}")

print_sep()
print(f"    {'TOTAL':<50} {nparams(den):>12,}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 5. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
components = {
    "LabelEncoder":  model.label_encoder,
    "LabelDecoder":  model.label_decoder,
    "ImageEncoder":  model.image_encoder,
    "Denoiser":      model.denoiser,
    "Prior":         model.prior,
    "Posterior":     model.posterior,
}

print_sep("═")
print(f"  {'SUMMARY':<30} {'Total Params':>14} {'% of Total':>10}")
print_sep("═")
total = sum(nparams(m) for m in components.values())
for name, mod in components.items():
    p = nparams(mod)
    print(f"  {name:<30} {p:>14,} {100*p/total:>9.1f}%")
print_sep()
print(f"  {'GRAND TOTAL':<30} {total:>14,} {'100.0%':>10}")
print_sep("═")
