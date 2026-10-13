"""H3 Pull-Forward  -  ComfyUI custom nodes for MiniMax H3 (part of comfyui-h3-pullforward)

Holds the still parts of a locked-off shot (the background) to one clean value across a chain of continuation clips, so the
"photocopy" drift of repeated generations does not build up in the background.

Nodes
  H3 Stable Mask To Latent  - Motion Mask `stable_mask` (pixels) -> the same mask on the H3 latent grid
  H3 Latent Pull-Forward    - pastes each still cell's anchor over the frames where the mask says it is stable
  H3 Mask To Video Image    - (optional, debug) latent mask -> black & white frames you can look at
  H3 Latent Tail / Tail Merge - (optional) cut the last frames off a latent so a resample pass only samples those, then put them back

Frame grouping and the 2x2 token snap follow the geometry used by MaskVidExperiments (GPL-3.0), which these nodes are modelled on.
"""
import numpy as np


def _starts(n_src, n_dst):
    """Start index in a source axis of length n_src for each of n_dst equal groups (a partition of the axis)."""
    return np.minimum((np.arange(n_dst) * n_src) // n_dst, n_src - 1).astype(np.intp)


def _unpack(latent):
    s = latent["samples"]
    if hasattr(s, "is_nested") and s.is_nested:
        video, audio = s.unbind()
        return video, audio, True
    if isinstance(s, (list, tuple)):
        return s[0], s[1], True
    return s, None, False


RAW_MASK_CUTOFF = 0.5      # a mask value above this counts as white (stable)


def stable_to_latent(mask, F, H, W):
    """stable mask (1 = stable) -> bool [F,H,W].  Already latent-sized masks are used as they are; a pixel mask is reduced so a latent
    cell counts as stable only if EVERY pixel it covers is stable in EVERY pixel frame of its group."""
    m = np.asarray(mask, np.float32)
    if m.ndim == 2:
        m = m[None]
    if m.shape == (F, H, W):
        return m > RAW_MASK_CUTOFF
    b = (m > RAW_MASK_CUTOFF).astype(np.uint8)
    Bp, Hp, Wp = b.shape
    if Hp < H or Wp < W:
        raise ValueError(f"the stable mask ({Hp}x{Wp}) is smaller than the latent grid ({H}x{W})")
    b = np.minimum.reduceat(b, _starts(Hp, H), axis=1)
    b = np.minimum.reduceat(b, _starts(Wp, W), axis=2)
    b = np.minimum.reduceat(b, _starts(Bp, F), axis=0)
    return b.astype(bool)


AUTO_ANCHOR = "auto: clip 1 reference"
SAME_ANCHOR = "same clip: first frame of stability"
EXT_ANCHOR = "external reference clip"


def stable_pull_forward(video, stable, reference=None, anchor_source=SAME_ANCHOR, block_latents=5, ref_ok=None):
    """video [C,F,H,W]; stable bool [F,H,W].  Wherever a cell is stable, its value is replaced by the anchor of that stable stretch.
    The H3 encoder works in blocks of `block_latents` latent frames and the frame's position in the block changes its values
    (that is the repeating ~17-frame pulse), so each position in the block gets its OWN anchor: the cell's value at the first frame of
    the stretch that sits at that position. Pasting one anchor over all positions would make the block pattern show up as flicker.
    block_latents 0 or 1 = one anchor for all frames.
    reference [C,Fr,H,W] (optional): the anchors come from it instead. ref_ok [H,W] bool says which cells of the reference can be
    trusted (cells that were still in the reference clip); the other cells use this clip's own first stable frame.
    Returns (out, replaced[F,H,W])."""
    C, F, H, W = video.shape
    P = max(1, int(block_latents))
    out = video.copy()
    replaced = np.zeros((F, H, W), np.float32)
    anchor = np.zeros((P, C, H, W), np.float32)
    have = np.zeros((P, H, W), bool)
    ref = None
    if str(anchor_source).startswith("external") and reference is None:
        raise ValueError("anchor_source is 'external reference clip' but no reference is connected")
    if reference is not None:
        if reference.shape[0] != C or reference.shape[2:] != (H, W):
            raise ValueError("reference latent has a different channel count or grid size")
        ref = np.stack([np.median(reference[:, ph::P], axis=1) if reference[:, ph::P].shape[1] else np.median(reference, axis=1)
                        for ph in range(P)]).astype(np.float32)          # [P,C,H,W]
    ok = np.ones((H, W), bool) if ref_ok is None else np.asarray(ref_ok, bool)
    for g in range(F):
        st = stable[g]
        ph = g % P
        have[:, ~st] = False                                    # a cell that moved forgets its anchors
        if ref is not None:
            use = st & ok                                       # trusted reference cells: the reference is the anchor
            anchor[ph][:, use] = ref[ph][:, use]
            have[ph] |= use
            own = st & ~ok
        else:
            use = np.zeros((H, W), bool)
            own = st
        paste_own = own & have[ph]                              # this position already has an anchor from this stretch
        new = own & ~have[ph]
        anchor[ph][:, new] = video[:, g][:, new]                # first time this position is seen in the stretch: it IS the anchor
        have[ph] |= new
        paste = use | paste_own
        out[:, g][:, paste] = anchor[ph][:, paste]
        replaced[g] = paste
    return out, replaced


def check_reference_candidate(stable, ignore_first=7, min_stable=0.05, min_stable_frame=0.02):
    """Decide whether this clip's stable mask is good enough to be saved as the clip-1 reference.
    stable: bool [F,H,W].  The first `ignore_first` latent frames (the Motion Context overlap) are not looked at.
    Fails if too little of the picture is stable overall, or if at any moment motion covers (almost) the whole picture.
    Returns (ok, reason, stable_fraction_map [H,W])."""
    F = stable.shape[0]
    ig = min(max(int(ignore_first), 0), max(F - 1, 0))
    content = stable[ig:]
    frac_map = content.mean(0).astype(np.float32)
    overall = float(content.mean())
    per_frame = content.mean(axis=(1, 2))
    worst = float(per_frame.min())
    if overall < float(min_stable):
        return False, f"only {overall * 100:.1f}% of the picture is stable (needs {float(min_stable) * 100:.0f}%)", frac_map
    if worst < float(min_stable_frame):
        k = int(per_frame.argmin()) + ig
        return False, f"motion covers the whole picture at latent frame {k} (only {worst * 100:.1f}% stable, needs {float(min_stable_frame) * 100:.0f}%)", frac_map
    return True, f"{overall * 100:.0f}% of the picture stable, never below {worst * 100:.0f}% in any frame", frac_map


def _reference_path(name):
    import os
    import re
    import folder_paths
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name or "clip1_reference")) or "clip1_reference"
    d = os.path.join(folder_paths.get_output_directory(), "h3_pullforward")
    return d, os.path.join(d, safe + ".npz")


def save_reference(name, video, stable_frac):
    import os
    d, p = _reference_path(name)
    os.makedirs(d, exist_ok=True)
    np.savez(p, video=np.asarray(video, np.float32), stable_frac=np.asarray(stable_frac, np.float32))
    return p


def load_reference(name):
    import os
    _, p = _reference_path(name)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["video"], z["stable_frac"]


def delete_reference(name):
    import os
    _, p = _reference_path(name)
    if os.path.exists(p):
        os.remove(p)
        return True
    return False


class H3LatentPullForward:
    """Pastes each still cell's anchor over the frames where the Motion Mask's stable_mask says it is stable.
    anchor 'auto: clip 1 reference': clip 1 saves its own result as the reference and every later clip pastes clip 1's picture."""
    CATEGORY = "mask/motion"
    FUNCTION = "run"
    RETURN_TYPES = ("LATENT", "MASK", "STRING")
    RETURN_NAMES = ("latent", "replaced_cells", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "The clip's video+audio latent."}),
                "stable_mask": ("MASK", {"tooltip": "The stable_mask output of Motion Mask (white = stable). Pixel-sized, or latent-sized (e.g. through H3 Stable Mask To Latent)."}),
                "mode": (["probe only", "pull forward"], {"default": "pull forward",
                                                          "tooltip": "probe only = change nothing, just report."}),
                "anchor_source": ([AUTO_ANCHOR, SAME_ANCHOR, EXT_ANCHOR], {"default": AUTO_ANCHOR,
                                  "tooltip": "auto: clip 1 reference = clip 1 pastes its own picture and saves it; every later clip pastes clip 1's picture (cells that were never still in clip 1 use the clip's own picture). "
                                             "same clip: first frame of stability = what gets pasted is this clip's own picture, taken at the first frame the mask calls that cell stable (after it was last moving). "
                                             "external reference clip = what gets pasted is the typical picture of the clip connected to `reference`."}),
                "block_latents": ("INT", {"default": 5, "min": 0, "max": 64,
                                          "tooltip": "H3 builds the video in blocks of this many latent frames (5), and a frame's place in its block changes its values (the repeating ~17-frame pulse). "
                                                     "Each place in the block gets its own anchor so the pulse is not pasted into the background. 0 = off (one anchor for everything)."}),
            },
            "optional": {
                "reference": ("LATENT", {"tooltip": "Only used when anchor_source is 'external reference clip' (or to override the saved clip 1 in auto mode)."}),
                "clip_index": ("INT", {"default": 1, "min": 1, "max": 100000,
                                       "tooltip": "The shot number (1, 2, 3 ...). Used by 'auto: clip 1 reference': shot 1 saves the reference; later shots use it, or, if there is none, the first clip that passes the checks saves it."}),
                "reference_name": ("STRING", {"default": "clip1_reference",
                                              "tooltip": "File name of the saved clip 1 reference (in output/h3_pullforward). Use a different name for a different scene; running shot 1 again replaces it."}),
                "min_stable_fraction": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                                                  "tooltip": "A clip is NOT saved as the reference if less than this share of the picture is stable overall."}),
                "min_stable_per_frame": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01,
                                                   "tooltip": "A clip is NOT saved if at any moment less than this share of the picture is stable (motion covers the whole screen, for example a camera move or a cut)."}),
                "ignore_first_latent_frames": ("INT", {"default": 7, "min": 0, "max": 64,
                                                       "tooltip": "These first latent frames are not looked at by the checks above. 7 latent frames = the first 22 video frames (the Motion Context overlap)."}),
                "reference_min_stable": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                                                   "tooltip": "A cell of the saved clip 1 is only used as the anchor if it was stable in at least this share of clip 1's frames. Other cells (where the character was) use the current clip's own picture."}),
            },
        }

    def run(self, latent, stable_mask, mode, anchor_source, block_latents=5, reference=None, clip_index=1, reference_name="clip1_reference",
            min_stable_fraction=0.05, min_stable_per_frame=0.02, ignore_first_latent_frames=7, reference_min_stable=0.8):
        import torch
        import comfy.nested_tensor
        video, audio, nested = _unpack(latent)
        v = video.unsqueeze(0) if video.ndim == 4 else video
        arr = v.detach().float().cpu().numpy()                       # [B,C,F,H,W]
        mk = stable_mask.detach().float().cpu().numpy()
        ref_arr = None
        if reference is not None:
            rv, _, _ = _unpack(reference)
            rv = rv.unsqueeze(0) if rv.ndim == 4 else rv
            ref_arr = rv.detach().float().cpu().numpy()
        notes = []
        ref_ok = None
        anchor_eff = anchor_source
        is_auto = anchor_source == AUTO_ANCHOR
        if is_auto:
            anchor_eff = SAME_ANCHOR
            if ref_arr is not None:
                anchor_eff = EXT_ANCHOR
                notes.append("auto: used the connected reference clip.")
            elif int(clip_index) > 1:                                   # shot 1 never reuses an old file
                try:
                    loaded = load_reference(reference_name)
                except Exception as e:                                # unreadable file: act as if there is none
                    loaded = None
                    notes.append(f"auto: could not read the saved clip 1 ({e}).")
                if loaded is None:
                    notes.append(f"auto: no saved reference '{reference_name}' yet, so this clip used its own first stable frame.")
                else:
                    rvid, rfrac = loaded
                    if rvid.shape[0] != arr.shape[1] or rvid.shape[2:] != arr.shape[3:]:
                        notes.append("auto: the saved clip 1 has a different size, so this clip used its own first stable frame.")
                    else:
                        ref_arr = rvid[None]
                        ref_ok = rfrac >= float(reference_min_stable)
                        anchor_eff = EXT_ANCHOR
                        notes.append(f"auto: pasting clip 1's picture where clip 1 was still ({float(ref_ok.mean()) * 100:.0f}% of the picture); the rest uses this clip's own.")
        # Fully in motion: nothing is still anywhere, so there is nothing to paste or to save. Pass the clip through untouched.
        F0, H0, W0 = arr.shape[2:]
        stable_share = float(stable_to_latent(mk, F0, H0, W0).mean())
        if stable_share < 0.001:
            msg = (f"skipped: the clip is fully in motion ({stable_share * 100:.2f}% of cell-frames stable), so the latent passes through untouched "
                   "and this clip is not checked as a reference.")
            return (latent, torch.zeros((F0, H0, W0), dtype=torch.float32), msg)
        outs, reps = [], []
        st0 = None
        for b in range(arr.shape[0]):
            C, F, H, W = arr[b].shape
            st = stable_to_latent(mk, F, H, W)
            if b == 0:
                st0 = st
            r = None if ref_arr is None else ref_arr[min(b, len(ref_arr) - 1)]
            o, rep = stable_pull_forward(arr[b], st, r, anchor_eff, block_latents, ref_ok if r is not None else None)
            outs.append(o); reps.append(rep)
        report = (f"stable cells {float(stable_to_latent(mk, *arr.shape[2:]).mean()) * 100:.1f}% of all cell-frames; "
                  f"pasted into {float((reps[0] > 0).mean()) * 100:.1f}%. anchor = " +
                  ("clip 1's saved picture" if ref_ok is not None else f"the cell's value at the first frame its stable stretch began") +
                  f" (anchor_source: {anchor_source}); pasted over every frame until the mask says it moved.")
        # Save a reference: shot 1 always tries (and replaces/deletes the old one); a later shot tries only while no reference exists,
        # so the first clip that stays stable the whole way becomes the reference for everything after it.
        no_ref_used = ref_ok is None and ref_arr is None
        if is_auto and mode == "pull forward" and (int(clip_index) == 1 or no_ref_used):
            first = int(clip_index) == 1
            ok, why, frac = check_reference_candidate(st0, ignore_first_latent_frames, min_stable_fraction, min_stable_per_frame)
            if ok:
                try:
                    save_reference(reference_name, outs[0], frac)
                    notes.append(f"auto: clip {int(clip_index)} saved as the reference for the clips after it ({why}).")
                except Exception as e:
                    notes.append(f"auto: could not save the reference ({e}).")
            else:
                if first:
                    try:
                        delete_reference(reference_name)
                    except Exception:
                        pass
                notes.append(f"auto: clip {int(clip_index)} NOT saved as the reference, {why}. "
                             "Later clips will use their own first stable frame until a clip passes the check.")
        if notes:
            report += "\n" + "\n".join(notes)
        replaced = torch.from_numpy(np.ascontiguousarray(reps[0]))
        if mode != "pull forward":
            return (latent, replaced, report)
        out = torch.from_numpy(np.ascontiguousarray(np.stack(outs))).to(device=v.device, dtype=v.dtype)
        if video.ndim == 4:
            out = out.squeeze(0)
        new = dict(latent)
        new["samples"] = comfy.nested_tensor.NestedTensor((out, audio)) if nested else out
        new.pop("noise_mask", None)
        return (new, replaced, report)


def _h3_frame_groups(pattern, t, drop=0):
    """Pixel-frame range (start, end) feeding each latent frame, for an encoder that turns frames in a repeating cycle
    (`pattern` = frames per latent frame) and, if `drop` > 0, pads to whole chunks and trims `drop` latents at the end (MiniMax H3).
    Same geometry as MaskVidExperiments (GPL-3.0), which this node is modelled on."""
    starts = [0]
    for p in pattern:
        starts.append(starts[-1] + p)
    chunk, n = starts[-1], len(pattern)
    total = n * int(np.ceil(t / chunk))
    kept = max(1, total - drop) if drop > 0 else sum(1 for k in range(total) if (k // n) * chunk + starts[k % n] < t)
    groups = []
    for k in range(kept):
        a = (k // n) * chunk + starts[k % n]
        groups.append((min(a, t - 1), min(a + pattern[k % n], t)))
    if drop > 0:
        groups[-1] = (groups[-1][0], t)
    return groups


def stable_pixels_to_latent(stable, F, H, W, shrink_px=0, shrink_frames=0, pattern=None, drop=0, token=2):
    """stable: [B,Hp,Wp] 0..1 (1 = stable).  Returns (float32 [F,H,W] 1 = stable, note).
    Works on the MOVING side (1 - stable): moving areas are grown by shrink_px pixels (cross steps, like MVEx) and shrink_frames
    frames both ways, then a latent cell is moving if ANY pixel in it is, in ANY frame of its group, snapped to 2x2 tokens.
    So a cell is stable only if everything it covers is stable."""
    mv = (np.asarray(stable, np.float32) <= 0.5)
    if mv.ndim == 2:
        mv = mv[None]
    B, Hp, Wp = mv.shape
    if Hp < H or Wp < W:
        raise ValueError(f"the stable mask ({Hp}x{Wp}) is smaller than the latent grid ({H}x{W})")
    for _ in range(int(shrink_px)):
        o = mv.copy()
        o[:, 1:] |= mv[:, :-1]; o[:, :-1] |= mv[:, 1:]; o[:, :, 1:] |= mv[:, :, :-1]; o[:, :, :-1] |= mv[:, :, 1:]
        mv = o
    for _ in range(int(shrink_frames)):
        o = mv.copy()
        o[1:] |= mv[:-1]; o[:-1] |= mv[1:]
        mv = o
    m = np.maximum.reduceat(mv.astype(np.uint8), _starts(Hp, H), axis=1)
    m = np.maximum.reduceat(m, _starts(Wp, W), axis=2)
    if token and token > 1:
        pad = ((0, 0), (0, -H % token), (0, -W % token))
        q = np.pad(m, pad, mode="edge")
        q = q.reshape(q.shape[0], q.shape[1] // token, token, q.shape[2] // token, token).max(axis=(2, 4))
        m = q.repeat(token, axis=1).repeat(token, axis=2)[:, :H, :W]
    note = ""
    groups = _h3_frame_groups(pattern, B, drop) if pattern else None
    if groups is None or len(groups) != F:
        if groups is not None:
            note = f"(frame grouping gave {len(groups)} latent frames but the latent has {F}; used an even split instead) "
        groups = [(int(a), max(int(a) + 1, int(b))) for a, b in zip(_starts(B, F), list(_starts(B, F)[1:]) + [B])]
    out = np.stack([m[a:b].max(axis=0) for a, b in groups]).astype(bool)
    return (~out).astype(np.float32), note


class H3StableMaskToLatent:
    """Motion Mask's stable_mask (white = still) -> latent-grid stable mask for H3 Pull-Forward.
    Works like MVEx Mask To Latent Space (VAE-aware frame grouping, 2x2 token snap) but is built for a STABLE mask:
    a cell is stable only if every pixel it covers is stable in every frame of its group, and `grow_moving_*` pushes the stable
    areas away from anything that moves (the safe direction)."""
    CATEGORY = "mask/motion"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("stable_mask_latent", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "stable_mask": ("MASK", {"tooltip": "The stable_mask output of Motion Mask (white = still)."}),
            "latent": ("LATENT", {"tooltip": "The latent this mask will be used on (only its size is read)."}),
            "vae": ("VAE", {"tooltip": "The video VAE (gives the real frames-per-latent pattern for H3)."}),
            "grow_moving_px": ("INT", {"default": 36, "min": 0, "max": 256,
                                       "tooltip": "Pushes the stable area this many pixels away from anything that moves (a safety margin around moving parts)."}),
            "grow_moving_frames": ("INT", {"default": 1, "min": 0, "max": 64,
                                           "tooltip": "Same, in frames: a cell also counts as moving this many frames before and after the motion."}),
        }}

    def run(self, stable_mask, latent, vae, grow_moving_px, grow_moving_frames):
        import torch
        video, _, _ = _unpack(latent)
        F, H, W = int(video.shape[-3]), int(video.shape[-2]), int(video.shape[-1])
        pattern, drop, token, why = None, 0, 2, "no H3 frame pattern found on the VAE: used an even split of frames"
        inner = getattr(vae, "first_stage_model", None)
        ratio_t = getattr(inner, "vae_ratio_t", None)
        chunk_latents = getattr(inner, "tokens_chunk_size", None)
        if ratio_t is not None and chunk_latents is not None and ratio_t > 1:
            pre = getattr(inner, "frame_pre_padding", 0) or 0
            drop = getattr(inner, "token_drop", 0) or 0
            pattern = [ratio_t - pre] + [ratio_t] * (chunk_latents - 1)
            why = f"H3 frame pattern {pattern} per chunk, {drop} trailing latents trimmed"
        mk = stable_mask.detach().float().cpu().numpy()
        out, note = stable_pixels_to_latent(mk, F, H, W, grow_moving_px, grow_moving_frames, pattern, drop, token)
        report = (f"{mk.shape[0]} video frames {mk.shape[2]}x{mk.shape[1]} -> {F} latent frames {W}x{H}; {why}. {note}"
                  f"stable cells: {float(out.mean()) * 100:.1f}% of cell-frames.")
        return (torch.from_numpy(np.ascontiguousarray(out)), report)


class H3MaskToVideoImage:
    """Turns a latent-grid mask [F,h,w] (e.g. replaced_cells) into a black & white IMAGE batch the same size and length as `images`
    (nearest-neighbour in space, each video frame takes the latent frame it belongs to), so it can be stacked next to the video."""
    CATEGORY = "mask/motion"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK", {"tooltip": "Latent-grid mask, e.g. replaced_cells."}),
                             "images": ("IMAGE", {"tooltip": "Only its frame count and size are used."})}}

    def run(self, mask, images):
        import torch
        m = mask.detach().float().cpu()
        if m.ndim == 2:
            m = m[None]
        B, Hp, Wp = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        F = m.shape[0]
        idx = torch.clamp((torch.arange(B) * F) // B, max=F - 1)
        m = m[idx]
        m = torch.nn.functional.interpolate(m[:, None], size=(Hp, Wp), mode="nearest")[:, 0]
        return ((m > 0.5).float()[..., None].repeat(1, 1, 1, 3),)


def _pack(like, video, audio, nested):
    if not nested:
        return video
    import comfy.nested_tensor
    return comfy.nested_tensor.NestedTensor((video, audio))


def tail_geometry(F, tail_video_frames, block_latents=5, frames_per_block=17):
    """Latent frame where the tail starts.  The tail covers at least `tail_video_frames` video frames and always starts on a block boundary
    (a multiple of block_latents), because that is where the VAE's frame packing repeats."""
    per_latent = frames_per_block / float(block_latents)
    want = int(np.ceil(tail_video_frames / per_latent))
    start = ((F - want) // block_latents) * block_latents
    return max(int(start), 0)


class H3LatentTail:
    """Cuts the END of a video+audio latent (and, optionally, a latent-grid mask) so a resample pass only has to sample those frames.
    Use H3 Latent Tail Merge afterwards to put the resampled tail back."""
    CATEGORY = "latent/h3"
    FUNCTION = "run"
    RETURN_TYPES = ("LATENT", "MASK", "STRING")
    RETURN_NAMES = ("tail_latent", "tail_mask", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT", {"tooltip": "The video+audio latent (for example the Pull-Forward output)."}),
            "tail_video_frames": ("INT", {"default": 39, "min": 1, "max": 4096,
                                          "tooltip": "Resample at least this many video frames at the end. The cut snaps back to a block boundary, so you get a few more."}),
            "block_latents": ("INT", {"default": 5, "min": 1, "max": 64, "tooltip": "Latent frames per VAE block (5 for H3)."}),
        },
            "optional": {"mask": ("MASK", {"tooltip": "A latent-grid mask [frames,h,w] cut the same way (for example the moving-cells mask)."})}}

    def run(self, latent, tail_video_frames, block_latents, mask=None):
        video, audio, nested = _unpack(latent)
        F = int(video.shape[-3])
        start = tail_geometry(F, tail_video_frames, block_latents)
        v = video[..., start:, :, :]
        a = audio
        a_note = ""
        if audio is not None:
            A = int(audio.shape[2])
            a_start = int(round(A * start / float(F)))
            a = audio[:, :, a_start:]
            a_note = f"; audio time steps {a_start}..{A}"
        new = dict(latent)
        new["samples"] = _pack(latent["samples"], v, a, nested)
        new.pop("noise_mask", None)
        m = mask
        if m is not None:
            m = m.detach()
            if m.ndim == 2:
                m = m[None]
            m = m[start:]
        report = (f"tail = latent frames {start}..{F} ({F - start} of {F}), about {int(round((F - start) * 17 / 5.0))} video frames{a_note}")
        return (new, m, report)


class H3LatentTailMerge:
    """Puts a resampled tail (from H3 Latent Tail, after the sampler) back on the end of the full latent."""
    CATEGORY = "latent/h3"
    FUNCTION = "run"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT", {"tooltip": "The full latent the tail was cut from."}),
                             "tail_latent": ("LATENT", {"tooltip": "The same tail after resampling."})}}

    def run(self, latent, tail_latent):
        import torch
        video, audio, nested = _unpack(latent)
        tv, ta, _ = _unpack(tail_latent)
        F, Ft = int(video.shape[-3]), int(tv.shape[-3])
        v = torch.cat([video[..., :F - Ft, :, :], tv.to(video.dtype)], dim=-3)
        a = audio
        if audio is not None and ta is not None:
            At = int(ta.shape[2])
            a = torch.cat([audio[:, :, :int(audio.shape[2]) - At], ta.to(audio.dtype)], dim=2)
        new = dict(latent)
        new["samples"] = _pack(latent["samples"], v, a, nested)
        new.pop("noise_mask", None)
        return (new,)


class H3MovingMask:
    """Stable mask on the latent grid (white = still) -> 'moving' mask for the resample pass (white = resample), with the first latent
    frames switched off so the carried-over context frames are never re-noised. 7 latent frames = the first 22 video frames."""
    CATEGORY = "mask/motion"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("moving_mask",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "stable_mask": ("MASK", {"tooltip": "The latent-grid stable mask from H3 Stable Mask To Latent (white = still)."}),
            "protect_first_latent_frames": ("INT", {"default": 7, "min": 0, "max": 64,
                                                    "tooltip": "These first latent frames are never resampled. 7 latent frames = the first 22 video frames, which is the Motion Context overlap."}),
        }}

    def run(self, stable_mask, protect_first_latent_frames):
        m = 1.0 - stable_mask.detach().float().clamp(0.0, 1.0)
        if m.ndim == 2:
            m = m[None]
        n = int(protect_first_latent_frames)
        if n > 0:
            m[:n] = 0.0
        return (m,)


NODE_CLASS_MAPPINGS = {"H3LatentPullForward": H3LatentPullForward, "H3StableMaskToLatent": H3StableMaskToLatent, "H3MovingMask": H3MovingMask,
                       "H3MaskToVideoImage": H3MaskToVideoImage,
                       "H3LatentTail": H3LatentTail, "H3LatentTailMerge": H3LatentTailMerge}
NODE_DISPLAY_NAME_MAPPINGS = {"H3MovingMask": "H3 Moving Mask (invert + protect first frames)",
                              "H3LatentPullForward": "H3 Latent Pull-Forward",
                              "H3StableMaskToLatent": "H3 Stable Mask To Latent",
                              "H3MaskToVideoImage": "H3 Mask To Video Image (debug)",
                              "H3LatentTail": "H3 Latent Tail (cut end for resample)",
                              "H3LatentTailMerge": "H3 Latent Tail Merge"}
