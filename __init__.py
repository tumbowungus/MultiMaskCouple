from .multi_mask_couple import MultiMaskCouple
from .masked_region_cond import MaskedRegionCond

NODE_CLASS_MAPPINGS = {
    "MultiMaskCouple": MultiMaskCouple,
    "MaskedRegionCond": MaskedRegionCond,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiMaskCouple": "MultiMaskCouple",
    "MaskedRegionCond": "MaskedRegionCond",
}

# Serve the frontend assets from ./web
WEB_DIRECTORY = "./web"
