from __future__ import annotations
import torch
from nodes import ConditioningCombine
from .attention_couple import AttentionCouple

def _combine(a, b):
    return b if a is None else ConditioningCombine().combine(a, b)[0]

class MultiMaskCouple:
    CATEGORY = "conditioning/MultiMaskCouple"
    FUNCTION = "process"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "inputcount": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                # Region 1
                "region_1_positive": ("CONDITIONING",),
                "region_1_negative": ("CONDITIONING",),
            }
        }

    def process(self, model, clip, inputcount: int, **kwargs):

        pos_combined = None
        neg_combined = None

        # Collect region conditionings (pre-masked by MaskedRegionCond)
        for i in range(1, inputcount + 1):
            rp = kwargs.get(f"region_{i}_positive", None)
            rn = kwargs.get(f"region_{i}_negative", None)
            if rp is not None:
                pos_combined = _combine(pos_combined, rp)
            if rn is not None:
                neg_combined = _combine(neg_combined, rn)

        new_model, pos_out, neg_out = AttentionCouple().attention_couple(
            model=model,
            clip=clip,
            positive=pos_combined,
            negative=neg_combined,
            mode="Attention",
        )
        return new_model, pos_out, neg_out

NODE_CLASS_MAPPINGS = {"MultiMaskCouple": MultiMaskCouple}
NODE_DISPLAY_NAME_MAPPINGS = {"MultiMaskCouple": "MultiMaskCouple"}
