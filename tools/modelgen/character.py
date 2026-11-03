"""Assemble a full character (body + eyes + outfit + hair) into meshed parts."""
from __future__ import annotations

import numpy as np

import humanoid as hu
import anatomy
import mesher
import outfits as of
from sdf import F

# outfit presets: list of (builder, kwargs, material)
OUTFITS = {
    "commoner": [
        ("shirt", dict(sleeve="short", collar="v"), "cloth_a"),
        ("trousers", dict(length="knee"), "cloth_b"),
        ("belt", dict(), "leather"),
        ("wraps", dict(), "cloth_c"),
        ("boots", dict(height="low"), "leather_dark"),
    ],
    "traveler": [
        ("shirt", dict(sleeve="long", collar="round"), "cloth_a"),
        ("tunic", dict(length=0.2), "cloth_a"),
        ("trousers", dict(length="full"), "cloth_b"),
        ("belt", dict(), "leather"),
        ("boots", dict(height="mid"), "leather_dark"),
        ("gloves", dict(), "leather"),
    ],
}


def build_parts(p, outfit="commoner", hair_style="short", beard_style="none", h=0.004, head_only=False):
    J = hu.build_skeleton(p)
    clip = (np.array([-1, -1, J["neck"][2] - 0.08], F), np.array([1, 1, 3], F)) if head_only else None
    body = anatomy.build_body(p, J)
    parts = {}

    bv, bt = mesher.mesh_shape(body, h=h, clip=clip)
    tags = mesher.vertex_tags(body, bv)
    parts["body"] = dict(v=bv, t=bt, mat="skin", tags=tags)

    ev, et = mesher.mesh_shape(anatomy.eye_shape(p, J), h=0.0012)
    parts["eyes"] = dict(v=ev, t=et, mat="eye", colors=eye_colors(p, J, ev))

    garments = []
    for kind, kw, mat in OUTFITS[outfit]:
        builder = {"shirt": of.shirt, "trousers": of.trousers, "belt": of.belt, "boots": of.boots,
                   "wraps": of.sandals_wraps, "gloves": of.gloves}.get(kind)
        if kind == "tunic":
            shp = of.tunic_skirt(body, J, p, **kw)
        else:
            shp = builder(body, J, p, **kw)
        gv, gt = mesher.mesh_shape(shp, h=h, clip=clip)
        if len(gt) == 0:
            continue
        parts[kind] = dict(v=gv, t=gt, mat=mat)
        garments.append(shp)

    hair_shapes = []
    hs = of.hair(body, J, p, hair_style)
    if hs is not None:
        hair_shapes.append(hs)
    bs = of.beard(body, J, p, beard_style)
    if bs is not None:
        hair_shapes.append(bs)
    eb = of.eyebrows(J, p)
    if eb is not None:
        hair_shapes.append(eb)
    for i, s in enumerate(hair_shapes):
        if s is None:
            continue
        v, t = mesher.mesh_shape(s, h=min(h, 0.003), clip=clip)
        if len(t):
            parts[f"hair{i}"] = dict(v=v, t=t, mat="hair")

    # cull skin fully hidden under garments (saves triangles, prevents poke-through)
    if garments:
        inside = np.zeros(len(bv), bool)
        for g in garments:
            inside |= g.eval(bv) < -0.0025
        tri_hidden = inside[bt].all(axis=1)
        parts["body"]["t"] = bt[~tri_hidden]
    return J, parts


def eye_colors(p, J, verts, iris=(0.25, 0.45, 0.55)):
    h = hu.HeadSpace(p, J)
    cols = np.zeros((len(verts), 3), F)
    for sx in (-1, 1):
        c = anatomy.eye_centers(p, J)[0 if sx < 0 else 1] if p["snout"] <= 0 else h(sx * 0.046, -0.05, 0.012)
        m = (np.sign(verts[:, 0]) == sx)
        d = verts[m] - c
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        fwd = np.array([0, -1, 0], F) if p["snout"] <= 0 else np.array([sx * 0.5, -0.86, 0], F)
        cosang = d @ fwd
        col = np.tile(np.array([0.92, 0.9, 0.86], F), (m.sum(), 1))
        ir = cosang > 0.875
        col[ir] = np.array(iris, F) * (0.7 + 0.3 * ((cosang[ir] - 0.875) / 0.125))[:, None]
        col[cosang > 0.972] = (0.02, 0.02, 0.02)
        cols[m] = col
    return cols
