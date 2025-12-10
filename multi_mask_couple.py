# comfy_couple.py — minimal combine + per-pixel global default weight in AttentionCouple
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
                "default_positive": ("CONDITIONING",),
                "default_negative": ("CONDITIONING",),
                "inputcount": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                # Region 1
                "region_1_positive": ("CONDITIONING",),
                "region_1_negative": ("CONDITIONING",),
            }
        }

    def process(self, model, default_positive, default_negative, inputcount: int, **kwargs):

        pos_combined = default_positive
        neg_combined = default_negative

        # Collect region conditionings (pre-masked by MaskedRegionCond)
        for i in range(1, inputcount + 1):
            rp = kwargs.get(f"region_{i}_positive", None)
            rn = kwargs.get(f"region_{i}_negative", None)
            if rp is not None:
                pos_combined = _combine(pos_combined, rp)
                print(f"Region {i} positive combined.")
            else:
                print("No region positive for region", i)
            if rn is not None:
                neg_combined = _combine(neg_combined, rn)
                print(f"Region {i} negative combined.")
            else:
                print("No region negative for region", i)

        if pos_combined is None:
            raise ValueError("No positive conditioning provided (need default_positive or region positives).")

        new_model, pos_out, neg_out = AttentionCouple().attention_couple(
            model=model,
            positive=pos_combined,
            negative=neg_combined,
            mode="Attention",
        )
        return new_model, pos_out, neg_out

NODE_CLASS_MAPPINGS = {"MultiMaskCouple": MultiMaskCouple}
NODE_DISPLAY_NAME_MAPPINGS = {"MultiMaskCouple": "MultiMaskCouple"}
