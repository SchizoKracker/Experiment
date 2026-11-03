"""Head close-ups (3/4 left, front, 3/4 right + profile) in skin-clay, stacked above the reference.
usage: python3 face_preview.py <out.png> <preset> [ref_image]"""
import sys, time
import numpy as np
from PIL import Image
import races, humanoid as hu, body, mesher, anatomy, character
import hair as hr
from sdf import F
import bpy
import blendutil as bu

out, preset = sys.argv[1], sys.argv[2]
ref = sys.argv[3] if len(sys.argv) > 3 else None
t = time.time()
p = races.preset(preset)
J = hu.build_skeleton(p)
s = body.build_body(p, J)
hf = hr.HeadFrame(J)
lo = np.array([-0.16, hf.yc - 0.17, hf.top - 1.45 * hf.hh], F)
hi = np.array([0.16, hf.yc + 0.17, hf.top + 0.02], F)
v, f = mesher.mesh_shape(s, h=0.0017, clip=(lo, hi), verbose=False)
ev, ef = mesher.mesh_shape(anatomy.eye_shape(p, J), h=0.0007, verbose=False)
tags = mesher.vertex_tags(s, v)
print("meshed", round(time.time() - t, 1), len(f))
bu.reset()
skin = np.array((0.66, 0.5, 0.42))
cols = np.tile(skin, (len(v), 1))
u_, f_, z_ = hr.HeadFrame(J).local(v)
lm_ = anatomy.REF_HEAD["landmarks"]
lipband = (((u_ / 0.1) ** 2 + ((z_ - (lm_["mouth"] + 0.004)) / 0.038) ** 2) < 1) & (f_ > 0.2)
cols[lipband] = skin * np.array([1.0, 0.74, 0.72])
ob = bu.mesh_object("head", v, f)
bu.set_vertex_colors(ob, np.clip(cols, 0, 1))
ob.data.materials.append(bu.vc_material("skin", rough=0.5, sss=0.18))
eo = bu.mesh_object("eyes", ev, ef)
bu.set_vertex_colors(eo, character.eye_colors(p, J, ev, iris=(0.22, 0.35, 0.45)))
eo.data.materials.append(bu.vc_material("eye", rough=0.04))
bu.setup_render((600, 600), 32)
bpy.context.scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.28, 0.29, 0.31, 1)
bpy.context.scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
hc = np.array([0, hf.yc, hf.top - 0.55 * hf.hh])
bu.add_light("AREA", tuple(hc + np.array([-0.9, -1.1, 1.0])), 70, (1.0, 0.95, 0.88), 0.9, tuple(hc))
bu.add_light("AREA", tuple(hc + np.array([1.2, -0.4, 0.3])), 18, (0.85, 0.9, 1.0), 1.5, tuple(hc))
bu.add_light("AREA", tuple(hc + np.array([0.3, 1.2, 0.8])), 30, (1, 1, 1), 1.0, tuple(hc))
cam = bu.camera(tuple(hc), tuple(hc), lens=85)
paths = []
for i, ang in enumerate((-38, 0, 38, 90)):
    a = np.radians(ang)
    d = 0.6
    cam.location = tuple(hc + np.array([np.sin(a) * d, -np.cos(a) * d, 0.03]))
    bu.look_at(cam, tuple(hc + np.array([0, 0, 0.0])))
    pth = f"/tmp/claude-0/prev/_face_{i}.png"
    bu.render(pth)
    paths.append(pth)
ims = [Image.open(x).convert("RGB") for x in paths]
W = 600 * len(ims)
rows = [ims]
sheet_h = 600
refim = None
if ref:
    refim = Image.open(ref).convert("RGB")
    refim = refim.resize((int(refim.width * 600 / refim.height), 600))
    sheet_h += 600
    W = max(W, refim.width)
sheet = Image.new("RGB", (W, sheet_h), (20, 20, 20))
for i, im in enumerate(ims):
    sheet.paste(im, (600 * i, 0))
if refim:
    sheet.paste(refim, (0, 600))
sheet.save(out)
print("saved", out, round(time.time() - t, 1))

# ---------------- orthographic cross-reference against 4.webp (front + side), aligned by top/chin/midline
if "--ortho" in sys.argv:
    refimg = Image.open("/tmp/claude-0/-home-user/b59d01a1-ddc9-50bf-96aa-6e7dbd729d22/images/4.webp").convert("RGB")
    hh = hf.hh
    S = 1.6 * hh                        # ortho width (m) for an 800 px render
    bpy.context.scene.render.resolution_x = 800
    bpy.context.scene.render.resolution_y = 800
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = S
    zc = hf.top - 0.62 * hh
    outs = []
    for view, pos in (("front", (0, -2.0, 0)), ("side", (2.0, 0, 0))):
        tgt = np.array([0, hf.yc, zc])
        cam.location = tuple(tgt + np.array(pos)); bu.look_at(cam, tuple(tgt))
        pth = f"/tmp/claude-0/prev/_ortho_{view}.png"; bu.render(pth)
        im = Image.open(pth).convert("RGB")
        ppm = 800 / S
        top_px = 400 - (hf.top - zc) * ppm
        scale = 552 / (hh * ppm)                      # ref: top y=10, chin y=562
        im = im.resize((int(800 * scale), int(800 * scale)))
        top_px *= scale
        if view == "front":
            mid_x = 400 * scale; ref_mid = 386.5; ref_crop = refimg.crop((0, 0, 600, 790))
        else:
            # side: align the head centre plane (x0 = 887 in the ref, i.e. 277 in the side crop)
            mid_x = 400 * scale; ref_mid = 887 - 610; ref_crop = refimg.crop((610, 0, 1216, 790))
        dx, dy = int(ref_mid - mid_x), int(10 - top_px)
        canvas = Image.new("RGB", ref_crop.size, (40, 40, 40))
        canvas.paste(im, (dx, dy))
        blend = Image.blend(ref_crop, canvas, 0.5)
        row = Image.new("RGB", (ref_crop.width * 3, ref_crop.height))
        row.paste(ref_crop, (0, 0)); row.paste(canvas, (ref_crop.width, 0)); row.paste(blend, (ref_crop.width * 2, 0))
        outs.append(row)
    sh = Image.new("RGB", (outs[0].width, outs[0].height + outs[1].height))
    sh.paste(outs[0], (0, 0)); sh.paste(outs[1], (0, outs[0].height))
    sh.save(out.replace(".png", "_ortho.png"))
    print("saved ortho", out.replace(".png", "_ortho.png"))
