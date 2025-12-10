import torch
import torch.nn.functional as F
from contextlib import nullcontext
import copy
import comfy
from comfy.ldm.modules.attention import optimized_attention

def _safe_interpolate_nchw(x: torch.Tensor, size):
    return F.interpolate(x, size=size, mode="nearest-exact")

def _match_len(cond: torch.Tensor, target_len: int) -> torch.Tensor:
    cur = cond.shape[1]
    if cur == target_len:
        return cond
    if cur > target_len:
        return cond[:, :target_len, :]
    reps = (target_len + cur - 1) // cur
    return cond.repeat(1, reps, 1)[:, :target_len, :]

def _q_spatial_from_original(q: torch.Tensor, original_shape):
    _, S, _ = q.shape
    H, W = int(original_shape[2]), int(original_shape[3])
    if H <= 0 or W <= 0 or S <= 0:
        return 1, S

    for ds in (1, 2, 4, 8, 16, 32):
        h, w = H // ds, W // ds
        if h > 0 and w > 0 and h * w == S:
            return h, w

    # last-resort ratio fit
    target = W / max(1.0, H)
    best = None
    lim = int(S ** 0.5) + 1
    for h in range(1, lim):
        if S % h: continue
        w = S // h
        err = abs((w / max(1.0, h)) - target)
        if best is None or err < best[0]:
            best = (err, h, w)

    return (best[1], best[2]) if best else (1, S)


def _mask_to_q_layout(mask_any, q: torch.Tensor, original_shape) -> torch.Tensor:
    """
    Convert an arbitrary mask into [B,S,1] aligned with q.
    - None/Scalar -> broadcast
    - HxW / 1xHxW / CxHxW -> downsample to q's spatial h*w and flatten
    """
    B, S, _ = q.shape

    if mask_any is None:
        return torch.ones((B, S, 1), dtype=q.dtype, device=q.device)

    m = mask_any
    if not torch.is_tensor(m):
        m = torch.as_tensor(m, dtype=q.dtype, device=q.device)
    else:
        m = m.to(dtype=q.dtype, device=q.device)

    if m.ndim == 0 or (m.ndim == 1 and m.numel() == 1):
        return torch.ones((B, S, 1), dtype=q.dtype, device=q.device) * m.clamp(0.0, 1.0)

    # Expect HxW-like
    if m.ndim == 3:
        if m.shape[0] in (1, 3, 4):
            # robust: any nonzero pixel across channels is "on"
            m = (m.any(dim=0)).to(m.dtype)
        else:
            m = m.squeeze()
    if m.ndim != 2:
        raise RuntimeError(f"Expected HxW-like mask, got {tuple(m.shape)}")

    hds, wds = _q_spatial_from_original(q, original_shape)
    m = m.unsqueeze(0).unsqueeze(0)                 # [1,1,H,W]
    m = _safe_interpolate_nchw(m, size=(hds, wds))  # [1,1,hds,wds]
    m = m.view(1, 1, hds * wds, 1).repeat(B, 1, 1, 1).squeeze(1)  # [B,S,1]
    return m.clamp(0.0, 1.0)

def _to(x: torch.Tensor, device, dtype):
    if x.device != device or x.dtype != dtype:
        return x.to(device=device, dtype=dtype)
    return x

def _fp32_autocast_for(t: torch.Tensor):
    """
    Disable autocast for the device of tensor t (cuda/mps/cpu) so we run attention in fp32.
    """
    dev = t.device.type
    try:
        if dev == "cuda":
            return torch.cuda.amp.autocast(enabled=False)
        elif dev == "mps":
            return torch.autocast("mps", enabled=False)
        elif dev == "cpu":
            return torch.autocast("cpu", enabled=False)
    except Exception:
        pass
    return nullcontext()

def set_model_patch_replace(model, patch, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    to["patches_replace"]["attn2"][key] = patch

def iter_attn2_modules(unet):
    """
    Yields (key_tuple, attn2_module) for every cross-attn found anywhere in the UNet.
    key_tuple is a stable identifier you can use in patches_replace.
    """
    # Walk known trees (input/middle/output) but don't assume exact indices/counts.
    roots = [
        ("input", getattr(unet, "input_blocks", [])),
        ("middle", getattr(unet, "middle_block", [])),
        ("output", getattr(unet, "output_blocks", [])),
    ]
    for root_name, seq in roots:
        for i, sub in enumerate(seq):
            # Stages can be lists/Modules; find SpatialTransformers inside
            modules = sub if isinstance(sub, (list, tuple)) else [sub]
            for j, mod in enumerate(modules):
                # Some stages have the SpatialTransformer at fixed slot (often j==1),
                # but we won't assume that. Scan recursively.
                for name, m in mod.named_modules():
                    # A TransformerBlock typically has .attn2
                    if hasattr(m, "attn2"):
                        # Try to recover which transformer block index it is
                        # by parsing path segments like 'transformer_blocks.N'.
                        tbi = None
                        parts = name.split(".")
                        if "transformer_blocks" in parts:
                            k = parts.index("transformer_blocks")
                            if k + 1 < len(parts) and parts[k + 1].isdigit():
                                tbi = int(parts[k + 1])
                        key = (root_name, i) if tbi is None else (root_name, i, tbi)
                        yield key, m.attn2

def apply_patches(new_model, make_patch):
    unet = new_model.model.diffusion_model
    for key, attn2 in iter_attn2_modules(unet):
        set_model_patch_replace(new_model, make_patch(attn2), key)

class AttentionCouple:
    """
    Cross-attn mixing:
      • Mapping: cond_or_uncond aligns with q_list; 0=cond, 1=uncond.
      • Base positive per pixel: 1 outside union, global_default_weight inside union.
      • Region positives: mask*strength, renorm so sum ≤ union per pixel.
      • Region negatives: renorm so sum ≤ 1 per pixel; global negative full-field.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "mode": (["Attention", "Latent"], ),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "attention_couple"
    CATEGORY = "loaders"

    def attention_couple(self, model, positive, negative, mode):
        if mode == "Latent":
            return (model, positive, negative)

        self.raw_positive = copy.deepcopy(positive)
        self.raw_negative = copy.deepcopy(negative)

        new_model = model.clone()

        # Clear prior patches each call (avoid accumulation across runs)
        to = new_model.model_options.setdefault("transformer_options", {})
        to["patches_replace"] = {"attn2": {}}

        apply_patches(new_model, self.make_patch)

        # Return only the first pooled items like original behavior
        return new_model, [self.raw_positive[0]], [self.raw_negative[0]]

    def make_patch(self, module):
        def patch(q, k, v, extra_options):
            cond_or_uncond = extra_options["cond_or_uncond"]  # list of 0/1 aligned with q_list
            q_list = q.chunk(len(cond_or_uncond), dim=0)  # same order as cond_or_uncond
            b = q_list[0].shape[0]
            original_shape = extra_options["original_shape"]

            # Choose a stable anchor for resampling masks: prefer the first uncond chunk (value==1), else 0
            q_anchor = q_list[0]
            for j, val in enumerate(cond_or_uncond):
                if int(val) == 1:
                    q_anchor = q_list[j]
                    break

            # ---- Build masks & contexts per side ----
            def build_side(entries):
                conds_list, opts_list = [], []
                for ent in entries:
                    conds_list.append(ent[0])
                    opts_list.append(ent[1] if isinstance(ent[1], dict) else {})

                # Identify entries with/without masks
                region_indices, raw_masks, strengths = [], [], []
                base_indices = []
                for idx, opts in enumerate(opts_list):
                    if "mask" in opts:
                        region_indices.append(idx)
                        m = _mask_to_q_layout(opts["mask"], q_anchor, original_shape).to(dtype=torch.float32)
                        raw_masks.append(m)
                        strengths.append(float(opts.get("mask_strength", 1.0)))
                    else:
                        base_indices.append(idx)

                masks_per_entry = []

                union = torch.clamp(torch.sum(torch.stack(raw_masks, dim=0), dim=0), 0.0, 1.0)

                w_stack = torch.stack([m * s for m, s in zip(raw_masks, strengths)],
                                      dim=0)  # [R,B,S,1]
                sum_w = torch.sum(w_stack, dim=0)
                eps = 1e-6
                ratio = torch.minimum(torch.ones_like(sum_w), union / (sum_w + eps))
                w_stack = w_stack * ratio
                torch.nan_to_num_(w_stack, nan=0.0, posinf=1.0, neginf=0.0)
                region_weights = [w_stack[j] for j in range(w_stack.shape[0])]

                base_weight = 1.0 - union
                base_weight = base_weight.clamp(0.0, 1.0)
                torch.nan_to_num_(base_weight, nan=0.0, posinf=1.0, neginf=0.0)

                # Assign weights back to entries in original order
                ridx_map = {entry_idx: j for j, entry_idx in enumerate(region_indices)}
                for entry_idx in range(len(conds_list)):
                    if entry_idx in ridx_map:
                        masks_per_entry.append(region_weights[ridx_map[entry_idx]])
                    else:
                        masks_per_entry.append(base_weight)

                # Token-length unify across entries
                if len(conds_list) == 1:
                    ctx = conds_list[0]
                else:
                    max_len = max(t.shape[1] for t in conds_list)
                    ctx = torch.cat([_match_len(t, max_len) for t in conds_list], dim=0)

                # Move to attention device & proj dtype
                ctx = _to(ctx, device=q_list[0].device, dtype=module.to_k.weight.dtype)
                # Masks to q dtype on device (attention upcasts later)
                masks_per_entry = [_to(m, q_list[0].device, q_list[0].dtype) for m in masks_per_entry]

                return masks_per_entry, ctx

            masks_uncond, ctx_uncond = build_side(self.raw_negative)
            masks_cond, ctx_cond = build_side(self.raw_positive)

            # Project contexts (module precision), move to device
            k_uncond = module.to_k(ctx_uncond).to(device=q_list[0].device)  # shape [Nneg, S, D]
            v_uncond = module.to_v(ctx_uncond).to(device=q_list[0].device)
            k_cond = module.to_k(ctx_cond).to(device=q_list[0].device)  # shape [Npos, S, D]
            v_cond = module.to_v(ctx_cond).to(device=q_list[0].device)

            len_pos = len(self.raw_positive)
            len_neg = len(self.raw_negative)

            out = []
            for i, c in enumerate(cond_or_uncond):
                if int(c) == 0:
                    masks_bank, k_bank, v_bank, length = masks_cond, k_cond, v_cond, len_pos
                else:
                    masks_bank, k_bank, v_bank, length = masks_uncond, k_uncond, v_uncond, len_neg
                q_src = q_list[i]
                q_target = q_src.repeat(length, 1, 1)  # [length*b, S, C]
                k_rep = k_bank.repeat_interleave(b, dim=0)  # [length*b, S, D]
                v_rep = v_bank.repeat_interleave(b, dim=0)

                with _fp32_autocast_for(q_src):
                    q32 = q_target.float()
                    k32 = k_rep.float()
                    v32 = v_rep.float()
                    qkv32 = optimized_attention(q32, k32, v32, extra_options["n_heads"])

                m_stack = torch.stack(masks_bank, dim=0)  # [length, B, S, 1]
                torch.nan_to_num_(m_stack, nan=0.0, posinf=1.0, neginf=0.0)
                m_flat = m_stack.reshape(length * b, m_stack.shape[2], 1)  # [length*b, S, 1]
                m_flat = m_flat.to(device=qkv32.device, dtype=qkv32.dtype)

                qkv32 = qkv32 * m_flat
                qkv32 = qkv32.view(length, b, qkv32.shape[1], qkv32.shape[2]).sum(dim=0)

                torch.nan_to_num_(qkv32, nan=0.0, posinf=0.0, neginf=0.0)
                qkv32 = torch.clamp(qkv32, min=-1e4, max=1e4)
                out.append(qkv32.to(dtype=q_src.dtype))

            y = torch.cat(out, dim=0)
            # Final safety net
            torch.nan_to_num_(y, nan=0.0, posinf=0.0, neginf=0.0)
            y = torch.clamp(y, min=-1e4, max=1e4)
            return y

        return patch

