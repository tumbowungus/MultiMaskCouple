# MultiMaskCouple

This is simple custom node for [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) which simplifies the process of masking your prompts, i.e. applying them to only part of the image.

MultiMaskCouple makes it a lot easier to manage scenes with multiple distinct characters. You can also heavily influence pose and composition.

This began as some minor improvements to [ComfyCouple](https://github.com/Danand/ComfyUI-ComfyCouple), but it's now pretty much its own separate thing.

## Installation

Clone this repo inside the custom_nodes directory in your ComfyUI install location.

## Features

- Arbitrary number of masks
- Any image resolution
- Fast attention coupling
- Convenient interface

## Nodes

MaskedRegionCond is a convenience node to reduce the number of sockets that need to be connected. It's just two ConditioningSetMask outputs bundled together.

MultiMaskCouple is where the action is, it applies the masks and does a process called "attention coupling." 
I can't fully explain attention coupling, but basically it applies the prompts to the appropriate regions while also allowing them to interact and combine.

The nodes can be found in conditioning > MultiMaskCouple.

## Usage

Usage can seem a little complicated at first. A basic outline is provided here, but the example workflow might be more helpful.

1. Create a mask image using pure RGB colors
2. Load the mask(s) using the built in "Load Image" and "Convert Image to Mask" nodes
3. For each mask, create a positive and negative prompt and CLIP encode them as normal
4. Feed those into a "Masked Region Conditioning" node.
5. Connect the outputs to the main "MultiMaskCouple" node.
6. You will also need a default positive and default negative prompt, this is applied to any unmasked regions, and is also mixed in when strength < 1.
7. The outputs of MultiMaskCouple hook into KSampler as normal.

## Example Output

This image:

![Example Trio](examples/example-trio.png)

was made with this mask:

![Example Mask](examples/trio-mask.png)

and [this workflow](examples/multimask-example.json).

## Tips and Notes

- This is not strict masking. The regions can interact through the attention coupling process. This is a positive feature, but it can get in the way sometimes. Phrases like "left of trio" can help a lot in keeping things separate. Black regions are also useful for enforcing spacing.
- It can be a hassle to manage the intersection of the different prompts (what they have in common, like setting and camera angle). The string manipulation nodes like Concat can be used to make this more convenient.
- Keeping the masks simple is usually a good idea. I've included a few examples.
- Developing an intuition for how to design masks takes time and is tricky to explain. Experiment!
- This README is a work in progress. I'll write a more full-fledged guide soon.
- Remember that the default mask is a fallback, not something applied globally. It will only be used in black regions or places where total strength < 1.
