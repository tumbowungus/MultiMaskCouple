# masked_region_cond.py
import torch
from nodes import ConditioningSetMask

class MaskedRegionCond:
    """
    Attach the SAME mask to a region's positive and negative CONDITIONING.
    Strength is per side so you can emphasize/suppress independently.
    """
    CATEGORY = "conditioning/MultiMaskCouple"
    FUNCTION = "apply"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("region_positive", "region_negative")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "mask": ("MASK",),
                "pos_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "neg_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    @staticmethod
    def _append_mask(cond, mask, strength: float):
        # ConditioningSetMask.append returns (conditioning_out,)
        return ConditioningSetMask().append(cond, mask, "default", float(strength))[0]

    def apply(self, positive, negative, mask, pos_strength: float, neg_strength: float):
        pos_out = self._append_mask(positive, mask, pos_strength)
        neg_out = self._append_mask(negative, mask, neg_strength)
        return (pos_out, neg_out)

NODE_CLASS_MAPPINGS = {"MaskedRegionCond": MaskedRegionCond}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskedRegionCond": "MaskedRegionCond"}
