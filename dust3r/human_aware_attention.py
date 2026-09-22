"""Human-aware temporal cross-attention helpers for DUSt3R."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def masked_softmax(logits: torch.Tensor, key_keep: torch.Tensor) -> torch.Tensor:
    """Softmax over allowed keys, returning zeros when a row has no valid key."""
    key_keep = key_keep.to(device=logits.device, dtype=torch.bool)
    allowed = key_keep[:, None, None, :]
    masked_logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
    probabilities = masked_logits.softmax(dim=-1)
    probabilities = probabilities.masked_fill(~allowed, 0.0)
    has_valid_key = key_keep.any(dim=-1)[:, None, None, None]
    return probabilities * has_valid_key.to(probabilities.dtype)


def masked_cross_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_position: torch.Tensor,
    key_position: torch.Tensor,
    query_keep: torch.Tensor,
    key_keep: torch.Tensor,
) -> torch.Tensor:
    """Run a CroCo CrossAttention module using background-only token exchange."""
    batch, query_tokens, channels = query.shape
    key_tokens = key.shape[1]
    value_tokens = value.shape[1]
    heads = module.num_heads

    q = module.projq(query).reshape(batch, query_tokens, heads, channels // heads).permute(0, 2, 1, 3)
    k = module.projk(key).reshape(batch, key_tokens, heads, channels // heads).permute(0, 2, 1, 3)
    v = module.projv(value).reshape(batch, value_tokens, heads, channels // heads).permute(0, 2, 1, 3)
    if module.rope is not None:
        q = module.rope(q, query_position)
        k = module.rope(k, key_position)

    logits = (q @ k.transpose(-2, -1)) * module.scale
    attention = module.attn_drop(masked_softmax(logits, key_keep))
    output = (attention @ v).transpose(1, 2).reshape(batch, query_tokens, channels)
    output = module.proj_drop(module.proj(output))
    return output * query_keep.to(output.dtype).unsqueeze(-1)


def _as_batch_vector(value: Any, device: torch.device) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(device=device).reshape(-1)


def _background_patch_mask(
    dynamic_mask: torch.Tensor,
    patch_size: int | Tuple[int, int],
    token_count: int,
) -> torch.Tensor:
    if dynamic_mask.ndim == 3:
        dynamic_mask = dynamic_mask.unsqueeze(1)
    if dynamic_mask.ndim != 4:
        raise ValueError(f"Expected Bx1xHxW dynamic mask, got {tuple(dynamic_mask.shape)}")
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    human_patches = F.max_pool2d(
        dynamic_mask.float(), kernel_size=patch_size, stride=patch_size
    ).flatten(1)
    if human_patches.shape[1] != token_count:
        raise ValueError(
            f"Dynamic mask produced {human_patches.shape[1]} patches, expected {token_count}"
        )
    return human_patches < 0.5


def temporal_background_masks(
    view1: Dict[str, Any],
    view2: Dict[str, Any],
    patch_size: int | Tuple[int, int],
    token_count1: int,
    token_count2: int,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Build per-view keep masks only for pairs from different timestamps."""
    timestamp1 = _as_batch_vector(view1.get("timestamp"), device)
    timestamp2 = _as_batch_vector(view2.get("timestamp"), device)
    dynamic1 = view1.get("dynamic_mask")
    dynamic2 = view2.get("dynamic_mask")
    if timestamp1 is None or timestamp2 is None or dynamic1 is None or dynamic2 is None:
        return None
    cross_time = timestamp1 != timestamp2
    if not bool(cross_time.any()):
        return None

    background1 = _background_patch_mask(dynamic1.to(device), patch_size, token_count1)
    background2 = _background_patch_mask(dynamic2.to(device), patch_size, token_count2)
    background1 = torch.where(cross_time[:, None], background1, torch.ones_like(background1))
    background2 = torch.where(cross_time[:, None], background2, torch.ones_like(background2))
    return background1, background2
