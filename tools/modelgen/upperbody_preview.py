"""Clay preview of head + upper body (to the waist): front, 3/4, side, back.
usage: python3 upperbody_preview.py <out.png> [preset=human_f]"""
import sys, time, os
import numpy as np
from PIL import Image
import races, humanoid as hu, body, mesher, anatomy, character
import hair as hr
from sdf import F
import bpy
import blendutil as bu

out = sys.argv[1]
preset = sys.argv[2] if len(sys.argv) > 2 else "human_f"
t0 = time.time()
hu.ARM_DROP_DEG = 72
p = races.preset(preset)
J = hu.build_skeleton(p)
s = body.build_body(p, J)
hf = hr.HeadFrame(J)
hh = hf.hh
H = p["height"]
lo = np.array([-0.42, hf.yc - 0.25, hf.top - 3.9 * hh], F)
hi = np.array([0.42, hf.yc + 0.25, hf.top + 0.02], F)
v, f = mesher.mesh_shape(s, h=0.0022, clip=(lo, hi), verbose=False)
ev, ef = mesher.mesh_shape(anatomy.eye_shape(p, J), h=0.0008, verbose=False)
print("meshed", round(time.time() - t0, 1), len(f))

bu.reset()
ob = bu.mesh_object("body", v, f)
ob.data.materials.append(bu.material("clay", (0.66, 0.5, 0.42), rough=0.5, sss=0.15))
bu.set_true_normals(ob, mesher.gradient(s, v.astype(F)))
eo = bu.mesh_object("eyes", ev, ef)
bu.set_vertex_colors(eo, character.eye_colors(p, J, ev, iris=(0.22, 0.35, 0.45)))
eo.data.materials.append(bu.vc_material("eye", rough=0.04))
W, Hh = 700, 900
bu.setup_render((W, Hh), 32)
sc = bpy.context.scene
sc.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.28, 0.29, 0.31, 1)
sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
c = np.array([0, hf.yc + 0.01, hf.top - 1.75 * hh])
bu.add_light("AREA", tuple(c + np.array([-1.2, -1.5, 1.4])), 160, (1.0, 0.95, 0.88), 1.2, tuple(c))
bu.add_light("AREA", tuple(c + np.array([1.6, -0.6, 0.4])), 40, (0.85, 0.9, 1.0), 1.8, tuple(c))
bu.add_light("AREA", tuple(c + np.array([0.4, 1.6, 1.0])), 80, (1, 1, 1), 1.4, tuple(c))
cam = bu.camera(tuple(c), tuple(c), lens=60)
ims = []
for name, ang in (("front", 0), ("34", 38), ("side", 90), ("back", 180)):
    a = np.radians(ang)
    d = 1.55
    cam.location = tuple(c + np.array([np.sin(a) * d, -np.cos(a) * d, 0.05]))
    bu.look_at(cam, tuple(c))
    pth = os.path.join(os.path.dirname(out) or ".", f"_ub_{name}.png")
    bu.render(pth)
    ims.append(Image.open(pth).convert("RGB"))
sheet = Image.new("RGB", (W * len(ims), Hh))
for i, im in enumerate(ims):
    sheet.paste(im, (W * i, 0))
sheet.save(out)
print("saved", out, round(time.time() - t0, 1))
