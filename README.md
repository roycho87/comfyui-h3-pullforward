# ComfyUI H3 Pull-Forward

Keeps the **still background of a locked-off shot from drifting** when you chain MiniMax H3 continuation clips.

Each continuation re-generates the background from the tail of the previous clip, and every pass drifts it a little
(colour, brightness, texture - the "photocopy effect"). These nodes find which parts of the picture are *not* moving, take one
clean value for them, and paste that value back over the following frames in the latent, so the background is carried
forward instead of re-generated. Only the cells that are still are touched; the moving subject is left alone.

**Result so far:** a 10-clip chain on one locked-off scene, with the background held to clip 1 and a resample pass on the moving parts.
Background drift, brightness and saturation stay close to clip 1 instead of walking away clip by clip.

> Status: working prototype. Tested on one static-camera scene, a chain of 10 clips. It needs a **locked-off camera**.

---

## What is in the box

| Node | Where to find it | What it does |
|---|---|---|
| **Motion Mask** | `mask/motion` | Finds what moves in the decoded video and outputs a motion mask and a **stable mask** |
| **H3 Stable Mask To Latent** | `mask/motion` | Converts the stable mask from pixels to the H3 latent grid (like MVEx Mask To Latent Space, built for stable masks) |
| **H3 Latent Pull-Forward** | `mask/motion` | Pastes each still cell's anchor over the frames where the mask says it is stable |
| **H3 Moving Mask** | `mask/motion` | Inverts the stable mask into a moving-cells mask for a resample pass and keeps the first 22 video frames (7 latent frames) out of it |
| **H3 Latent Tail / Tail Merge** | `latent/h3` | Optional, for the resample template: cut the last frames off a latent for a short second pass, then put them back |
| **H3 Mask To Video Image** | `mask/motion` | Optional, for debugging: turns a latent mask (for example `replaced_cells`) into black & white frames you can look at |

`workflows/MiniMaxH3_PullForward_Template.json` is a complete, minimal MiniMax H3 workflow with the nodes wired in (group 8).

`workflows/MiniMaxH3_PullForward_Resample_Template.json` is the same workflow plus an optional resample stage (group 9, see below). It needs the `H3BlendLatentsByFrames` node from [xyzDist/H3-LongTakeNoCuts](https://github.com/xyzDist/H3-LongTakeNoCuts).

---

## Install

1. Copy this folder into `ComfyUI/custom_nodes/`.
2. **Restart ComfyUI fully** (a browser refresh is not enough - Python nodes only load at startup).
3. If you already have a node called `MotionMask` from another pack or an old single file, remove it: two nodes with the same
   internal name clash.

No extra Python packages are needed (numpy only).

The template workflow also uses nodes from other packs (Power Lora Loader from rgthree, VideoHelperSuite, ComfyUI-Easy-Use,
LoadImageCrop, a Wildcard Processor, and KJNodes inside some subgraphs). ComfyUI's *Install Missing Custom Nodes* will list
what you still need.

---

## How it works, in the order the data flows

```
Sampler  ->  latent
              |-> decode (diagnostic) -> Motion Mask --stable_mask--> H3 Stable Mask To Latent --+
              |                                                                                  v
              +----------------------------------------------------------------->  H3 Latent Pull-Forward -> latent -> decode / save
```

### 1. Motion Mask (works on decoded pixels)

Compares each frame with its neighbours, cleans the result, and outputs two masks:

* **`motion_mask`** - white where something moves, at that frame.
* **`stable_mask`** - white where a pixel has been still for **`hold`** frames in a row. It starts white, turns black the
  moment the pixel moves, stays black while it moves, and turns white again only after `hold` still frames. It then stays
  white until the pixel moves again. This is the mask Pull-Forward uses.

Defaults (the values this was tuned with on a locked-off AI video):

| Setting | Default | Meaning |
|---|---|---|
| `threshold` | 0.01 | How much a pixel must change to count as motion. Lower = more sensitive (it catches slow motion, so less background is called "stable") |
| `compare` | previous and next frame | What each frame is compared with |
| `frame_gap` | 1 | Compare with the frame this many steps away |
| `blur` | 2.0 | Pre-blur in pixels, to hide compression noise |
| `despeckle` | 2 | Removes isolated specks |
| `fill_holes` | 80 | Fills the inside of moving shapes (flat areas barely change, only their edges light up) |
| `grow` | 16 | Safety margin around moving areas, in pixels |
| `linger` | 22 | Frames the motion mask stays white after motion stops |
| `lookahead` | 3 | Frames the mask turns white before motion starts |
| `feather` | 2.0 | Edge softness |
| `camera_compensation` | off | Leave off for a locked-off camera |
| `invert` | off | |
| **`hold`** | **22** | Frames a pixel must stay still before it counts as stable (about 3.4 video frames per latent frame) |

### 2. H3 Stable Mask To Latent

The latent is much smaller than the video (one cell is 16 x 16 pixels), and H3 packs video frames into latent frames in an
uneven repeating pattern. This node reads that pattern from the VAE and reduces the mask correctly:

* A latent cell counts as **stable only if every pixel it covers is stable, in every frame of its group** (the safe direction).
* **`grow_moving_px`** (default **36**) and **`grow_moving_frames`** (default **1**) push the stable area *away* from anything
  that moves, so the edge of a moving subject never ends up inside a pasted area.
* The result is snapped to the model's 2 x 2 token grid.

Connect the same VAE you decode with, and the latent you are about to paste into (only its size is read).

### 3. H3 Latent Pull-Forward

For every latent cell the mask calls stable:

1. When a stable stretch **begins**, take the cell's own value at that frame as the **anchor**.
2. For every following frame while the mask stays white, **paste the anchor over the cell**.
3. When the mask turns black (the cell moves), forget the anchor. When it turns white again, take a new one.

Settings:

| Setting | Default | Meaning |
|---|---|---|
| `mode` | pull forward | `probe only` changes nothing and just reports how many cells would be replaced |
| `anchor_source` | same clip: first frame of stability | **same clip** = paste this clip's own picture, taken at the first frame the mask calls the cell stable. **external reference clip** = paste the typical picture of the latent you connect to `reference` (for example an earlier clean clip) |
| `block_latents` | 5 | See below. Leave it at 5 for MiniMax H3 |
| `clip_index` | 1 | The shot number (wire it to your Shot Number). Only used by the auto reference anchor |
| `reference_name` | clip1_reference | File name of the saved reference in `output/h3_pullforward` |
| `min_stable_fraction` | 0.05 | A clip is not saved as the reference if less than this share of the picture is stable overall |
| `min_stable_per_frame` | 0.02 | A clip is not saved if at any moment less than this share is stable (motion covers the whole screen) |
| `ignore_first_latent_frames` | 7 | Those first latent frames (22 video frames, the Motion Context overlap) are not looked at by the two checks above |
| `reference_min_stable` | 0.8 | A cell of the saved reference is only used as the anchor if it was stable in at least this share of the reference clip's frames |

`anchor_source` also has **auto: clip 1 reference** (the default): see "hold the background to clip 1" below.

**Why `block_latents` matters.** H3 builds the video in blocks of 5 latent frames (about 17 video frames), and a frame's
position in its block changes its values. If you paste one anchor over every position, the latent becomes perfectly flat and
the decoder rebuilds each block from flat input, which shows up as a **brightness pulse every 17 frames**. With
`block_latents` = 5 the node keeps a separate anchor for each position in the block, so the block pattern is preserved and the
pulse does not appear. Set it to 0 to see the pulse for yourself.

Outputs: the modified `latent`, `replaced_cells` (a mask of the cells that were pasted into, one frame per latent frame) and a short `report`.

Because anchors for each block position need to be collected first, the **first 5 latent frames (about 17 video frames) of each
stable stretch are not replaced**, on top of the `hold` delay.

### 4. H3 Mask To Video Image (debug)

Scales a latent-grid mask up to video size and spreads it over the video frames, as black & white images. Wire `replaced_cells`
into it to see exactly where Pull-Forward pasted, next to your video.

---

## Using it in a workflow

1. After the sampler, decode the video part of the latent (a diagnostic decode) and feed those frames to **Motion Mask**.
2. Send `stable_mask`, the VAE and the sampler latent to **H3 Stable Mask To Latent**.
3. Send the sampler latent and the latent-grid mask to **H3 Latent Pull-Forward**.
4. Use the output latent for everything after this point: the save-latent node that the next continuation loads, and the final decode.

The template workflow does exactly this inside a normal Motion Context chain, with the shot number and the wildcard seed
set to *increment* so each queued run is the next clip.

### Optional: resample the moving cells
`workflows/MiniMaxH3_PullForward_Resample_Template.json` resamples the whole clip (the same idea as H3-LongTakeNoCuts, but only moving cells are re-noised): Pull-Forward, then a second pass (denoise 0.55, 5 steps), then `H3BlendLatentsByFrames` with `0:0, 22:0, 44:1`. The blended latent feeds both the export and the saved context.

`workflows/MiniMaxH3_PullForward_Resample_LastFrames_Template.json` is a shorter variant. It adds a short second sampler pass after Pull-Forward (denoise 0.55, 5 steps) on the **end of the clip only**.
**H3 Latent Tail** cuts the last `tail_video_frames` (default 39) off the latent and the moving-cells mask; the cut snaps back to a
5-latent-frame block, so you get about 41 video frames. The stable mask is inverted into a moving mask, only those cells are
re-noised and denoised, and **H3 Latent Tail Merge** puts the tail back. `H3BlendLatentsByFrames` fades from the Pull-Forward latent
to the resampled one between video frames 202 and 220; if you change the clip length or the tail size, move those two numbers to the
start of the tail (the Tail node's report says where it starts). Everything before the tail is untouched, which is what keeps the
pass short.

Experimental and not yet run on real latents: the audio latent is cut proportionally along its time axis (dimension 2), and the
tail is sampled as a stand-alone short clip. Check the first result for audio sync and a visible seam at the start of the tail.

### Optional: hold the background to clip 1 (auto reference)
`workflows/MiniMaxH3_PullForward_Resample_AutoClip1_Template.json` sets Pull-Forward's `anchor_source` to **auto: clip 1 reference**.
Shot 1 pastes its own picture as usual and saves itself as the reference (`output/h3_pullforward/clip1_reference.npz`). Every later
shot pastes clip 1's picture over the cells that were still in clip 1, so the background cannot drift away from clip 1 over the chain.
Cells that were moving in clip 1 (where the character was) are not trusted and use the clip's own first stable frame.
If clip 1 fails the checks (motion covers the whole picture, too little stable), no reference is saved and later clips use their own first stable frame, but every later shot checks itself again: the first clip that passes the same checks is saved as the reference for all clips after it.
If a clip is fully in motion (under 0.1% of its cells are ever stable), Pull-Forward skips itself: the latent passes through untouched and that clip is not checked as a reference.
The checks use `min_stable_fraction`, `min_stable_per_frame` and `ignore_first_latent_frames` (see the settings table).
Set `clip_index` from your Shot Number. Running shot 1 again replaces the reference; use a different `reference_name` for a different scene.

### Debug: where does it paste, where does it regenerate?
`workflows/MiniMaxH3_PullForward_Resample_AutoClip1_Debug_Template.json` is the auto-reference workflow plus group 10, which saves one extra video
(`output/minimax/debug/`) with three panels side by side: the decoded clip, `replaced_cells` (white = Pull-Forward pasted that cell, black = left to the
sampler) and the resample mask (white = re-noised by the resample pass). It uses **H3 Mask To Video Image** and ComfyUI's core **Image Stitch** node.
If an artifact in the left panel sits in a black area of the middle panel, the sampler regenerated it; if it sits in a white area, the paste is not protecting it.
The first ~17 video frames of each stable stretch are black on purpose (anchors are still being collected).

### Extra workflow
* `workflows/Video_Combiner_1-10_Template.json` joins up to 10 exported clips. Switch **Join end to end** on for clips whose context frames were already trimmed on export (the default Motion Context workflow does this); switch it off to use the **Overlap frames** value instead. **Audio blend (ms)** crossfades the audio across each seam in overlap mode (stepped volume fades built from core audio nodes, the audio keeps the same length as the video).

## Tuning

* **Background still flickers or drifts:** the mask is calling too little of it stable. Raise `threshold` a little, or lower
  `hold`, or lower `grow_moving_px`.
* **Pasted areas touch the moving subject (ghosting, frozen hands):** the mask is calling moving parts stable. Lower
  `threshold` (more sensitive), raise `grow_moving_px`, raise `hold`.
* **A brightness pulse every 17 frames:** check `block_latents` is 5.
* Use **`probe only`** and the `report` output to see how much would be replaced before you switch it on.

## Limitations

* Locked-off camera only. A moving camera makes everything "moving", so nothing is pasted.
* If the background really changes (a light is switched on), the mask turns black there and the new look becomes the new anchor
  after `hold` frames; it is not held to the old one.
* Written for the MiniMax H3 latent layout (24 channels, 16x spatial compression, blocks of 5 latent frames).
* Prototype: tested on one scene. Expect to tune the Motion Mask for yours.

## Credits

The resample idea (a second, lower-denoise pass over the clip, blended in over frames 22-44) comes from
[xyzDist/H3-LongTakeNoCuts](https://github.com/xyzDist/H3-LongTakeNoCuts), including the `H3BlendLatentsByFrames` node the resample templates use.
What this repo adds is the stable-background hold and a mask that keeps the resample off the still background.

The frame grouping and 2 x 2 token snap in *H3 Stable Mask To Latent* follow the geometry used by
[MaskVidExperiments](https://github.com/drozbay/MaskVidExperiments) (GPL-3.0) by drozbay. This repository is released under the
same licence (GPL-3.0, see `LICENSE`).
