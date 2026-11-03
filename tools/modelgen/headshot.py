"""Beauty headshot: sculpted head + strand hair/brows/lashes + painted makeup (vertex colour), Cycles.
usage: python3 headshot.py <out.png> [preset=human_f] [samples]"""
import sys, time
import numpy as np
from PIL import Image
import races, humanoid as hu, body, mesher, anatomy, grooms, face, character
import hair as hr
from headfit import LANDMARKS as LMK
from sdf import Shape, F
import bpy
import blendutil as bu
import os
OUT_DIR = os.environ.get("MODELGEN_OUT", "/tmp/claude-0/prev")
os.makedirs(OUT_DIR, exist_ok=True)
from hair_preview import hair_material, add_strands

out = sys.argv[1]
preset = sys.argv[2] if len(sys.argv) > 2 else "human_f"
samples = int(sys.argv[3]) if len(sys.argv) > 3 else 96
t0 = time.time()
hu.ARM_DROP_DEG = 72
p = races.preset(preset)
J = hu.build_skeleton(p)
s = body.build_body(p, J)
hf = hr.HeadFrame(J)
hh = hf.hh
lo = np.array([-0.2, hf.yc - 0.2, hf.top - 1.75 * hh], F)
hi = np.array([0.2, hf.yc + 0.2, hf.top + 0.02], F)
import os
CACHE = f"{OUT_DIR}/hs_cache_{preset}.npz"
use_cache = os.path.exists(CACHE) and "--rebuild" not in sys.argv
if use_cache:
    C = np.load(CACHE)
    v, f, ev, ef = C["v"], C["f"], C["ev"], C["ef"]
else:
    v, f = mesher.mesh_shape(s, h=0.0013, clip=(lo, hi), verbose=False)
    ev, ef = mesher.mesh_shape(anatomy.eye_shape(p, J), h=0.0005, verbose=False)
print("meshed", round(time.time() - t0, 1), len(f))


# ------------------------------------------------------------------ makeup (vertex colour)
def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def mix(c, target, w):
    return c * (1 - w[:, None]) + np.asarray(target)[None, :] * w[:, None]


def poly_dist(P, pts, n=80):
    pts = np.asarray(pts)
    tt = np.linspace(0, len(pts) - 1, n)
    i = np.clip(tt.astype(int), 0, len(pts) - 2)
    a = (tt - i)[:, None]
    dense = pts[i] * (1 - a) + pts[i + 1] * a
    d = np.full(len(P), np.inf)
    for q in dense:
        d = np.minimum(d, np.linalg.norm(P - q, axis=1))
    return d


u, fw, zn = hf.local(v)
au = np.abs(u)
skin = np.array([0.74, 0.55, 0.46])
cols = np.tile(skin, (len(v), 1))
rng = np.random.default_rng(2)
# natural variation: warmer/redder on cheeks, nose, ears; slightly cooler under the jaw
cols = mix(cols, (0.78, 0.5, 0.44), 0.35 * np.exp(-(((au - 0.17) / 0.08) ** 2 + ((zn - 0.66) / 0.07) ** 2)))   # blush
cols = mix(cols, (0.76, 0.5, 0.44), 0.25 * np.exp(-((au / 0.05) ** 2 + ((zn - 0.67) / 0.06) ** 2)) * (fw > 0.3))  # nose tip
cols = mix(cols, (0.7, 0.5, 0.44), 0.3 * smooth(0.3, 0.36, au) * smooth(0.4, 0.45, zn) * smooth(0.72, 0.66, zn))  # ears
# lips (vermilion): between the measured borders, tapering to the corners
mcl = LMK["mouth_corner_lat"]
uz, sz, lz = LMK["upper_lip_top_zn"], LMK["stomion_zn"], LMK["lower_lip_bottom_zn"]
tl = np.clip(au / mcl, 0, 1.2)
hsx = np.sqrt(np.clip(1 - tl ** 2.2, 0, 1)) * 0.8 + 0.2
bow = 0.004 * np.exp(-((au - 0.026) / 0.014) ** 2) - 0.003 * np.exp(-(au / 0.01) ** 2)
top_b = sz - (sz - uz + bow) * hsx
bot_b = sz + (lz - sz) * hsx
lipw = smooth(top_b - 0.003, top_b + 0.002, zn) * smooth(bot_b + 0.004, bot_b - 0.002, zn) * smooth(mcl * 1.04, mcl * 0.92, au) * (fw > 0.2)
cols = mix(cols, (0.62, 0.26, 0.27), 0.82 * lipw)
cols = mix(cols, (0.45, 0.16, 0.17), 0.35 * lipw * np.exp(-((zn - sz) / 0.006) ** 2))   # deeper at the mouth line
# eyes: shadow on the lid up to the crease (warm taupe), liner at the upper lash line, soft lower smudge
for sx, (up, lo_, c) in face.LID_ARCS.items():
    m = (np.sign(u) == sx) & (np.abs(au - LMK["pupil_lat"]) < 0.14) & (np.abs(zn - LMK["pupil_zn"]) < 0.12)
    P = v[m]
    du = poly_dist(P, up)
    dl = poly_dist(P, lo_)
    above = (P[:, 2] - c[2]) > -0.002
    shadow = np.exp(-(du / (0.006 * hh / 0.234 * 1.0 + 0.004)) ** 2) * above
    liner = np.exp(-(du / 0.0018) ** 2) * above
    outer = smooth(0.13, 0.21, np.abs(P[:, 0]) / hh)
    cc = cols[m]
    cc = mix(cc, (0.5, 0.36, 0.32), 0.55 * shadow * (0.6 + 0.4 * outer))
    cc = mix(cc, (0.1, 0.07, 0.06), 0.9 * liner)
    cc = mix(cc, (0.35, 0.25, 0.23), 0.35 * np.exp(-(dl / 0.0022) ** 2) * outer)
    cols[m] = cc
# caruncle + inner lid rim: moist pink
for sx, cp in face.CARUNCLE.items():
    dcar = np.linalg.norm(v - cp, axis=1)
    cols = mix(cols, (0.78, 0.45, 0.45), 0.85 * np.exp(-(dcar / (0.0075 * hh)) ** 2))
# brows: skin under the hair a touch darker
bc = np.interp(au, face.BROW_HAIR_LAT, face.BROW_HAIR_ZN)
cols = mix(cols, (0.55, 0.4, 0.33), 0.35 * np.exp(-((zn - bc) / 0.012) ** 2) * smooth(0.05, 0.07, au) * smooth(0.25, 0.23, au) * (fw > 0.1))
# scalp under the hair takes the hair colour (so the scalp never reads as thinning), feathered at the hairline
hd = grooms.hairline_density(hf, v, feather=0.035)
cols = mix(cols, (0.42, 0.2, 0.1), 0.85 * hd)
cols *= (1 + rng.normal(0, 0.012, (len(v), 1)))
cols = np.clip(cols, 0, 1) ** 2.2

# ------------------------------------------------------------------ grooms
if use_cache:
    gs = [dict(name=str(n), strands=C["S_" + str(n)], r_root=float(C["rr_" + str(n)]), r_tip=float(C["rt_" + str(n)])) for n in C["names"]
          if str(n) not in ("brows", "lashes")]
    # brows + lashes are regrown every run (cheap) on the full sculpt, including the brow ridge displacement
    blo = np.array([-0.08, hf.yc - 0.16, hf.top - 0.62 * hh], F)
    bhi = np.array([0.08, hf.yc - 0.02, hf.top - 0.36 * hh], F)
    bfield = hr.Field(s, blo, bhi, h=0.0012)
    gs.append(grooms.brows(s, J, bfield, density=0.9))
    gs.append(grooms.lashes(J, p, shape=s))
else:
    base = Shape(prims=s.prims)
    glo = np.array([-0.26, hf.yc - 0.22, hf.top - 0.9], F)
    ghi = np.array([0.26, hf.yc + 0.32, hf.top + 0.04], F)
    field = hr.Field(base, glo, ghi, h=0.0035)
    gs = list(grooms.female_braid(base, J, field)) if p["fem"] > 0.5 else [grooms.male_swept_back(base, J, field)]
    blo = np.array([-0.08, hf.yc - 0.16, hf.top - 0.62 * hh], F)
    bhi = np.array([0.08, hf.yc - 0.02, hf.top - 0.36 * hh], F)
    gs.append(grooms.brows(s, J, hr.Field(s, blo, bhi, h=0.0012), density=0.9))
    gs.append(grooms.lashes(J, p, shape=s))
    extra = {}
    for g in gs:
        extra["S_" + g["name"]] = g["strands"]; extra["rr_" + g["name"]] = g["r_root"]; extra["rt_" + g["name"]] = g["r_tip"]
    np.savez(CACHE, v=v, f=f, ev=ev, ef=ef, names=np.array([g["name"] for g in gs]), **extra)


def face_clear(S, keep_u=0.0):
    """drop strands that hang in front of the face (below the hairline, inside the face outline)"""
    Sm = S.reshape(-1, 3)
    uu, ff, zz = hf.local(Sm)
    bad = (zz > 0.30) & (np.abs(uu) < 0.27) & (ff > 0.12)
    bad = bad.reshape(S.shape[:2]).any(1)
    return S[~bad]


for g in gs:
    if g["name"] in ("scalp", "tendrils"):
        g["strands"] = face_clear(g["strands"])
    if g["name"] == "tendrils":
        T = g["strands"]
        roots_t = T[:, 0]
        g["strands"] = hr.clump_and_frizz(T, roots_t, 8, clump=0.85, frizz=0.0003, length_var=0.1, seed=4)
print("groomed", round(time.time() - t0, 1), {g["name"]: g["strands"].shape for g in gs})

# ------------------------------------------------------------------ skin shader
def skin_material():
    """Painted vertex colour + procedural skin detail: pores (Voronoi), micro-texture and fine creases (noise),
    colour mottling, oily/dry roughness variation, red-weighted subsurface scattering, sheen for vellus hair."""
    m = bpy.data.materials.new("skin")
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    b = N.get("Principled BSDF")
    tc = N.new("ShaderNodeTexCoord")
    col = N.new("ShaderNodeVertexColor"); col.layer_name = "Col"

    def noise(scale, detail=4.0, rough=0.5):
        n = N.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        L.new(tc.outputs["Object"], n.inputs["Vector"])
        return n

    def maprange(sock, a, b_, c, d):
        r = N.new("ShaderNodeMapRange")
        r.inputs["From Min"].default_value, r.inputs["From Max"].default_value = a, b_
        r.inputs["To Min"].default_value, r.inputs["To Max"].default_value = c, d
        L.new(sock, r.inputs["Value"])
        return r.outputs["Result"]

    def math(op, a, b_=None, v=None):
        mm = N.new("ShaderNodeMath"); mm.operation = op
        L.new(a, mm.inputs[0])
        if b_ is not None:
            L.new(b_, mm.inputs[1])
        elif v is not None:
            mm.inputs[1].default_value = v
        return mm.outputs[0]

    # colour: blotchy mottling (capillaries / melanin unevenness) at two scales
    mot1 = maprange(noise(45, 6, 0.6).outputs["Fac"], 0.3, 0.7, 0.93, 1.06)
    mot2 = maprange(noise(220, 3, 0.5).outputs["Fac"], 0.35, 0.65, 0.97, 1.03)
    mot = math("MULTIPLY", mot1, mot2)
    red = N.new("ShaderNodeMix"); red.data_type = "RGBA"; red.blend_type = "MULTIPLY"
    red.inputs["Factor"].default_value = 1.0
    L.new(col.outputs["Color"], red.inputs["A"])
    tint = N.new("ShaderNodeCombineColor")
    L.new(mot, tint.inputs[0]); L.new(math("POWER", mot, v=1.4), tint.inputs[1]); L.new(math("POWER", mot, v=1.6), tint.inputs[2])
    L.new(tint.outputs[0], red.inputs["B"])
    L.new(red.outputs["Result"], b.inputs["Base Color"])
    # subsurface
    b.inputs["Subsurface Weight"].default_value = 0.45
    b.inputs["Subsurface Radius"].default_value = (1.0, 0.38, 0.22)
    b.inputs["Subsurface Scale"].default_value = 0.004
    L.new(red.outputs["Result"], b.inputs["Subsurface Color"]) if "Subsurface Color" in b.inputs else None
    # roughness: oilier/dryer patches
    L.new(maprange(noise(18, 3, 0.5).outputs["Fac"], 0.3, 0.7, 0.36, 0.55), b.inputs["Roughness"])
    b.inputs["Specular IOR Level"].default_value = 0.45
    b.inputs["Sheen Weight"].default_value = 0.12
    b.inputs["Sheen Roughness"].default_value = 0.4
    # pores: Voronoi cells ~0.35 mm, squashed into small pits; micro-texture + faint creases
    vor = N.new("ShaderNodeTexVoronoi"); vor.feature = "F1"
    vor.inputs["Scale"].default_value = 2600
    vor.inputs["Randomness"].default_value = 1.0
    L.new(tc.outputs["Object"], vor.inputs["Vector"])
    pores = maprange(vor.outputs["Distance"], 0.0, 0.32, 0.0, 1.0)
    micro = maprange(noise(1400, 2, 0.6).outputs["Fac"], 0.3, 0.7, 0.0, 1.0)
    crease = maprange(noise(160, 8, 0.75).outputs["Fac"], 0.42, 0.58, 0.0, 1.0)
    h = math("ADD", math("MULTIPLY", pores, v=0.6), math("MULTIPLY", micro, v=0.3))
    h = math("ADD", h, math("MULTIPLY", crease, v=0.25))
    bump = N.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.00012
    L.new(h, bump.inputs["Height"])
    L.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def eye_material():
    """Procedural eye in object space (origin = eye centre, -Y = gaze): sclera, dark limbal ring,
    hazel-green iris with radial fibres, pupil; glossy wet coat."""
    m = bpy.data.materials.new("eye")
    m.use_nodes = True
    N, L = m.node_tree.nodes, m.node_tree.links
    b = N.get("Principled BSDF")
    tc = N.new("ShaderNodeTexCoord")
    nrm = N.new("ShaderNodeVectorMath"); nrm.operation = "NORMALIZE"
    L.new(tc.outputs["Object"], nrm.inputs[0])
    dot = N.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"
    L.new(nrm.outputs[0], dot.inputs[0]); dot.inputs[1].default_value = (0, -1, 0)
    ramp = N.new("ShaderNodeValToRGB")
    E = ramp.color_ramp.elements
    stops = [(0.0, (0.55, 0.42, 0.40)), (0.70, (0.80, 0.76, 0.72)), (0.862, (0.78, 0.75, 0.70)),
             (0.874, (0.05, 0.06, 0.05)), (0.886, (0.10, 0.17, 0.10)), (0.93, (0.20, 0.30, 0.13)),
             (0.958, (0.34, 0.27, 0.09)), (0.968, (0.12, 0.08, 0.03)), (0.972, (0.005, 0.005, 0.005)), (1.0, (0.0, 0.0, 0.0))]
    E[0].position, E[0].color = stops[0][0], (*stops[0][1], 1)
    E[1].position, E[1].color = stops[-1][0], (*stops[-1][1], 1)
    for pos, c in stops[1:-1]:
        e = E.new(pos); e.color = (*c, 1)
    L.new(dot.outputs["Value"], ramp.inputs["Fac"])
    # radial iris fibres: noise stretched along the radius
    sep = N.new("ShaderNodeSeparateXYZ"); L.new(nrm.outputs[0], sep.inputs[0])
    ang = N.new("ShaderNodeMath"); ang.operation = "ARCTAN2"
    L.new(sep.outputs["Z"], ang.inputs[0]); L.new(sep.outputs["X"], ang.inputs[1])
    nz = N.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 18; nz.inputs["Detail"].default_value = 6
    comb = N.new("ShaderNodeCombineXYZ"); L.new(ang.outputs[0], comb.inputs["X"]); L.new(dot.outputs["Value"], comb.inputs["Y"])
    L.new(comb.outputs[0], nz.inputs["Vector"])
    fib = N.new("ShaderNodeMapRange"); L.new(nz.outputs["Fac"], fib.inputs["Value"])
    fib.inputs["From Min"].default_value, fib.inputs["From Max"].default_value = 0.3, 0.7
    fib.inputs["To Min"].default_value, fib.inputs["To Max"].default_value = 0.6, 1.4
    irm = N.new("ShaderNodeMapRange"); L.new(dot.outputs["Value"], irm.inputs["Value"])
    irm.inputs["From Min"].default_value, irm.inputs["From Max"].default_value = 0.872, 0.88
    irm.inputs["To Min"].default_value, irm.inputs["To Max"].default_value = 0.0, 1.0
    w = N.new("ShaderNodeMath"); w.operation = "MULTIPLY"; L.new(irm.outputs[0], w.inputs[0])
    sub = N.new("ShaderNodeMath"); sub.operation = "SUBTRACT"; L.new(fib.outputs[0], sub.inputs[0]); sub.inputs[1].default_value = 1.0
    L.new(sub.outputs[0], w.inputs[1])
    add = N.new("ShaderNodeMath"); add.operation = "ADD"; add.inputs[0].default_value = 1.0; L.new(w.outputs[0], add.inputs[1])
    mul = N.new("ShaderNodeMix"); mul.data_type = "RGBA"; mul.blend_type = "MULTIPLY"; mul.inputs["Factor"].default_value = 1.0
    L.new(ramp.outputs["Color"], mul.inputs["A"])
    cc = N.new("ShaderNodeCombineColor")
    for i in range(3):
        L.new(add.outputs[0], cc.inputs[i])
    L.new(cc.outputs[0], mul.inputs["B"])
    L.new(mul.outputs["Result"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.25
    b.inputs["Coat Weight"].default_value = 1.0
    b.inputs["Coat Roughness"].default_value = 0.0
    b.inputs["Subsurface Weight"].default_value = 0.15
    b.inputs["Subsurface Scale"].default_value = 0.002
    return m


# ------------------------------------------------------------------ scene
bu.reset()
ob = bu.mesh_object("head", v, f)
bu.set_vertex_colors(ob, cols)
ob.data.materials.append(skin_material())
for poly in ob.data.polygons:
    poly.use_smooth = True
em = eye_material()
for k, cen in enumerate(anatomy.eye_centers(p, J)):
    side = np.sign(ev[:, 0]) == np.sign(cen[0])
    idx = np.where(side)[0]
    remap = -np.ones(len(ev), int); remap[idx] = np.arange(len(idx))
    fk = ef[np.all(side[ef], axis=1)]
    eo = bu.mesh_object(f"eye{k}", ev[idx] - cen, remap[fk])
    eo.location = tuple(cen)
    eo.data.materials.append(em)
    for poly in eo.data.polygons:
        poly.use_smooth = True
mats = dict(brows=hair_material("brow", melanin=0.82, red=0.65, rough=0.35),
            lashes=hair_material("lash", melanin=0.98, red=0.2, rough=0.3))
hm = hair_material("hair", melanin=0.55, red=0.9, rough=0.28)
for g in gs:
    add_strands(g["name"], g["strands"], g["r_root"], g["r_tip"], mats.get(g["name"], hm))

W, H = 1000, 1250
bu.setup_render((W, H), samples)
sc = bpy.context.scene
sc.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.09, 0.085, 0.08, 1)
sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
hc = np.array([0, hf.yc, hf.top - 0.62 * hh])
bu.add_light("AREA", tuple(hc + np.array([-0.7, -0.9, 0.55])), 32, (1.0, 0.93, 0.85), 1.2, tuple(hc))    # key
bu.add_light("AREA", tuple(hc + np.array([0.9, -0.5, 0.1])), 7, (0.85, 0.9, 1.0), 1.6, tuple(hc))       # fill
bu.add_light("AREA", tuple(hc + np.array([0.6, 0.9, 0.6])), 60, (1.0, 0.85, 0.7), 0.7, tuple(hc))        # rim
bu.add_light("AREA", tuple(hc + np.array([-0.5, 0.8, 0.4])), 40, (0.9, 0.9, 1.0), 0.7, tuple(hc))        # rim 2
cam = bu.camera(tuple(hc), tuple(hc), lens=100)
views = [("34", (0.38, -0.82, 0.04)), ("front", (0.0, -0.9, 0.03)), ("profile", (-0.9, -0.05, 0.03))]
ims = []
for name, d in views:
    cam.location = tuple(hc + np.array(d)); bu.look_at(cam, tuple(hc + np.array([0, 0, 0.005])))
    pth = f"{OUT_DIR}/_hs_{name}.png"; bu.render(pth); ims.append(Image.open(pth).convert("RGB"))
    ims[-1].save(out.replace(".png", f"_{name}.png"))
sheet = Image.new("RGB", (W * len(ims), H))
for i, im in enumerate(ims):
    sheet.paste(im, (W * i, 0))
sheet.save(out)
print("saved", out, round(time.time() - t0, 1))
