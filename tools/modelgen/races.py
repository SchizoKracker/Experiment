"""Race / subrace body presets and palettes for the humanoid sculptor.

Each *model base* is race + body type (masc / fem).  Subraces mostly differ in
palette and small proportion tweaks, which the game applies as blend shapes
and material parameters, so they share a base mesh with their parent race.
"""
from humanoid import P

# body-shape parameters per race (masculine form); feminine form derives from these
RACE_BODY = {
    "human":      dict(height=1.78, muscle=0.62),
    "elf":        dict(height=1.80, muscle=0.35, fat=0.15, limb_thick=0.9, shoulder=0.94, arm=1.03, leg=1.05, neck=1.08,
                       jaw=0.88, chin=0.95, brow=0.7, cheek=1.15, nose_len=1.02, nose_w=0.85, eye=1.06, eye_tilt=0.6,
                       face_long=1.04, ears="long", ear_size=1.0),
    "half_elf":   dict(height=1.76, muscle=0.45, fat=0.25, limb_thick=0.96, jaw=0.95, brow=0.85, cheek=1.07, eye_tilt=0.3,
                       ears="pointed", ear_size=0.95),
    "dwarf":      dict(height=1.35, head_scale=1.26, muscle=0.75, fat=0.45, limb_thick=1.32, shoulder=1.2, hip=1.12,
                       leg=0.86, arm=0.94, hand=1.18, foot=1.12, neck=0.8, jaw=1.15, brow=1.35, nose_len=1.18, nose_w=1.25,
                       cheek=1.05, eye=0.95),
    "halfling":   dict(height=0.98, head_scale=1.28, muscle=0.35, fat=0.4, limb_thick=1.08, shoulder=0.96, leg=0.95,
                       hand=1.04, foot=1.28, jaw=0.9, chin=0.92, brow=0.8, cheek=1.15, nose_len=0.92, eye=1.1,
                       ears="pointed", ear_size=0.9),
    "gnome":      dict(height=1.02, head_scale=1.4, muscle=0.3, fat=0.35, limb_thick=1.0, shoulder=0.95, leg=0.92,
                       hand=1.1, jaw=0.88, chin=1.1, brow=0.9, cheek=1.2, nose_len=1.45, nose_w=1.3, eye=1.18,
                       ears="pointed", ear_size=1.2),
    "dragonborn": dict(height=1.95, muscle=0.8, fat=0.3, limb_thick=1.18, shoulder=1.16, hip=1.02, neck=1.3, hand=1.1,
                       foot=1.08, snout=1.0, ears="none", tail=0.0),
    "half_orc":   dict(height=1.86, muscle=0.85, fat=0.35, limb_thick=1.2, shoulder=1.14, neck=1.25, jaw=1.25, chin=1.05,
                       brow=1.55, nose_len=0.92, nose_w=1.35, nose_bridge=0.8, lips=1.15, eye=0.92, ears="pointed",
                       ear_size=0.9, tusks=1.0),
    "tiefling":   dict(height=1.78, muscle=0.5, fat=0.22, cheek=1.12, jaw=0.98, brow=1.05, eye_tilt=0.3, horns="ram",
                       tail=0.95, ears="pointed", ear_size=0.95),
    # --- originals ---
    "tallowborn": dict(height=1.74, muscle=0.35, fat=0.4, limb_thick=1.05, brow=0.6, nose_len=0.75, nose_w=0.9,
                       nose_bridge=0.7, lips=0.75, cheek=0.85, jaw=0.95, ears="none", wick=1.0),
    "mothkin":    dict(height=1.72, muscle=0.25, fat=0.1, limb_thick=0.85, shoulder=0.92, arm=1.06, leg=1.06,
                       neck=1.1, jaw=0.78, chin=0.85, brow=0.6, nose_len=0.55, nose_w=0.7, eye=1.75, ears="none",
                       antennae=1.0, lips=0.7),
}

FEM_ADJUST = dict(fem=1.0, muscle_mul=0.75, fat_add=0.08, height_mul=0.93, shoulder_mul=1.0, jaw_mul=0.92, brow_mul=0.75)

# default skin colours (linear-ish RGB for Blender previews; game uses sRGB palettes from data)
SKIN = {
    "human": (0.62, 0.42, 0.32), "elf": (0.78, 0.62, 0.52), "half_elf": (0.7, 0.5, 0.38), "dwarf": (0.66, 0.43, 0.33),
    "halfling": (0.7, 0.5, 0.38), "gnome": (0.72, 0.52, 0.42), "dragonborn": (0.55, 0.12, 0.1),
    "half_orc": (0.36, 0.45, 0.3), "tiefling": (0.62, 0.18, 0.16), "tallowborn": (0.86, 0.8, 0.62),
    "mothkin": (0.55, 0.5, 0.45),
}


def preset(name: str):
    """'human_m', 'elf_f', ... -> parameter dict."""
    race, _, sex = name.rpartition("_")
    if sex not in ("m", "f"):
        race, sex = name, "m"
    body = dict(RACE_BODY[race])
    p = P(**body)
    if sex == "f":
        p["fem"] = 1.0
        p["muscle"] *= FEM_ADJUST["muscle_mul"]
        p["fat"] += FEM_ADJUST["fat_add"]
        p["height"] *= FEM_ADJUST["height_mul"]
        p["jaw"] *= FEM_ADJUST["jaw_mul"]
        p["brow"] *= FEM_ADJUST["brow_mul"]
        p["tusks"] *= 0.75
    p["race"] = race
    p["sex"] = sex
    return p


def skin_color(p):
    return SKIN.get(p.get("race", "human"), SKIN["human"])


ALL_BASES = [f"{r}_{s}" for r in RACE_BODY for s in ("m", "f")]
